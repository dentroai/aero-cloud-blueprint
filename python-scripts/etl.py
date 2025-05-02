import os
import sys
import argparse
import tempfile
import subprocess
import uuid
import base64
from pathlib import Path
import requests
from pdf2image import convert_from_path
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging
import time
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Document Handling ---


def find_documents(folder_path: str) -> list[Path]:
    """Finds all PDF and DOCX files recursively in the given folder."""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("**/*.pdf"))
    docx_files = list(folder.glob("**/*.docx"))
    logging.info(
        f"Found {len(pdf_files)} PDF files and {len(docx_files)} DOCX files in {folder_path}"
    )
    return pdf_files + docx_files


def convert_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Converts a PDF file to PNG images, one image per page."""
    image_paths = []
    try:
        images = convert_from_path(pdf_path, dpi=300)  # Using 300 dpi as a balance
        for i, image in enumerate(images):
            image_filename = f"page_{i + 1}.png"
            image_filepath = output_dir / image_filename
            image.save(image_filepath, "PNG")
            image_paths.append(image_filepath)
        logging.info(
            f"Converted {pdf_path.name} to {len(image_paths)} images in {output_dir}"
        )
    except Exception as e:
        logging.error(f"Error converting PDF {pdf_path.name} to images: {e}")
    return image_paths


def convert_docx_to_images(docx_path: Path, output_dir: Path) -> list[Path]:
    """Converts a DOCX file to PNG images via a temporary PDF."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        temp_pdf_path = None
        try:
            # Try docx2pdf first (on any platform)
            try:
                from docx2pdf import convert as convert_docx

                logging.info(f"Using docx2pdf to convert {docx_path.name} to PDF.")
                temp_pdf_path_str = str(tmpdir / (docx_path.stem + ".pdf"))
                convert_docx(str(docx_path), temp_pdf_path_str)
                temp_pdf_path = Path(temp_pdf_path_str)
                if not temp_pdf_path.exists():
                    logging.error(
                        f"Converted PDF {temp_pdf_path} not found after docx2pdf conversion."
                    )
                    raise FileNotFoundError(f"docx2pdf did not create {temp_pdf_path}")
            except (ImportError, Exception) as e:
                logging.warning(f"docx2pdf failed or not installed: {e}")

                # Fallback to LibreOffice if available
                if sys.platform.startswith("linux"):
                    logging.info(
                        f"Trying LibreOffice to convert {docx_path.name} to PDF."
                    )
                    result = subprocess.run(
                        [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "pdf",
                            str(docx_path),
                            "--outdir",
                            str(tmpdir),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode != 0:
                        logging.error(
                            f"LibreOffice conversion failed for {docx_path.name}: {result.stderr}"
                        )
                        return []

                    # Find the converted PDF (LibreOffice names it based on the original)
                    pdf_name = docx_path.stem + ".pdf"
                    temp_pdf_path = tmpdir / pdf_name
                    if not temp_pdf_path.exists():
                        logging.error(
                            f"Converted PDF {temp_pdf_path} not found after LibreOffice conversion."
                        )
                        return []
                else:
                    logging.error(
                        "No available method to convert DOCX to PDF. Please install docx2pdf or use LibreOffice."
                    )
                    return []

            # Convert the temporary PDF to images
            logging.info(f"Converting temporary PDF {temp_pdf_path.name} to images.")
            return convert_pdf_to_images(temp_pdf_path, output_dir)

        except Exception as e:
            logging.error(f"Error processing DOCX {docx_path.name}: {e}")
            return []


# --- AI Model Interactions ---


def encode_image_to_base64(image_path: Path) -> str | None:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None


def transcribe_image_vllm(
    image_path: Path, vllm_endpoint: str, model_name: str
) -> str | None:
    """Transcribes an image using a vLLM endpoint (OpenAI compatible API)."""
    logging.info(f"Transcribing image {image_path.name} using vLLM model {model_name}")
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    headers = {"Content-Type": "application/json"}
    # Assuming OpenAI compatible chat completions endpoint /v1/chat/completions
    payload = {
        "model": "vllm-model",  # Using the default served_model_name from vLLM, regardless of the actual model loaded
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe the content of this document image accurately and completely.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 4000,  # Adjust as needed
    }
    api_url = f"{vllm_endpoint.rstrip('/')}/v1/chat/completions"

    try:
        response = requests.post(
            api_url, headers=headers, json=payload, timeout=300
        )  # 5 min timeout
        response.raise_for_status()
        data = response.json()
        transcription = data["choices"][0]["message"]["content"]
        logging.info(f"Successfully transcribed {image_path.name}")
        return transcription.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling vLLM API at {api_url}: {e}")
    except (KeyError, IndexError) as e:
        logging.error(
            f"Error parsing vLLM response for {image_path.name}: {e}. Response: {response.text[:200]}"
        )
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during transcription for {image_path.name}: {e}"
        )
    return None


def get_embedding_ollama(
    text: str, ollama_endpoint: str, model_name: str
) -> list[float] | None:
    """Gets text embedding from an Ollama API endpoint."""
    logging.debug(
        f"Getting embedding for text snippet (length: {len(text)}) using Ollama model {model_name}"
    )
    api_url = f"{ollama_endpoint.rstrip('/')}/api/embeddings"  # Corrected endpoint
    payload = {"model": model_name, "prompt": text}  # Use 'prompt' for embeddings

    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        else:
            logging.error(
                f"Ollama embedding response missing 'embedding' key or not a list. Keys: {list(data.keys())}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Ollama embedding API at {api_url}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during embedding: {e}")
    return None


# --- Vector Store Interaction ---


def setup_qdrant_client(
    host: str,
    port: int,
    collection_name: str,
    vector_size: int,
    distance=models.Distance.COSINE,
) -> QdrantClient | None:
    """Initializes Qdrant client and ensures the collection exists."""
    try:
        client = QdrantClient(host=host, port=port)
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logging.info(
                f"Created Qdrant collection '{collection_name}' with vector size {vector_size}"
            )
        else:
            # Collection exists, but don't attempt to verify parameters to avoid potential API compatibility issues
            logging.info(f"Using existing Qdrant collection '{collection_name}'")
            logging.info(
                f"Assuming it's compatible with vector size {vector_size} and {distance} distance"
            )
        return client
    except Exception as e:
        logging.error(
            f"Error connecting to or setting up Qdrant collection '{collection_name}': {e}"
        )
        return None


def upsert_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    doc_path: Path,
    page_num: int,
    transcription: str,
    embedding: list[float],
):
    """Upserts the document page data to Qdrant."""
    if not embedding:
        logging.warning(
            f"Skipping upsert for {doc_path.name} page {page_num} due to missing embedding."
        )
        return

    # Use a deterministic ID based on file path and page number
    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_path.resolve()}_page_{page_num}")
    )

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "file_path": str(doc_path.resolve()),
            "file_name": doc_path.name,
            "page_number": page_num,
            "text": transcription,
            "timestamp": time.time(),  # Add timestamp for potential filtering
        },
    )

    try:
        client.upsert(
            collection_name=collection_name, points=[point], wait=False
        )  # wait=False for async upsert
        logging.debug(f"Upserted point {point_id} for {doc_path.name} page {page_num}")
    except Exception as e:
        logging.error(f"Error upserting point {point_id} to Qdrant: {e}")


# --- Parallel Processing ---


def process_page(
    doc_path: Path, image_path: Path, page_num: int, args, qdrant_client, vector_size
):
    """Process a single page of a document."""
    logging.info(
        f"Processing page {page_num} of document {doc_path.name} ({image_path.name})"
    )

    # a. Transcribe image
    transcription = transcribe_image_vllm(
        image_path, args.vllm_endpoint, args.vllm_model
    )
    if not transcription:
        logging.warning(f"Failed to transcribe {image_path.name}. Skipping page.")
        return False

    # b. Get embedding
    embedding = get_embedding_ollama(
        transcription, args.ollama_endpoint, args.ollama_model
    )
    if not embedding:
        logging.warning(
            f"Failed to get embedding for transcription of {image_path.name}. Skipping page."
        )
        return False

    # Ensure embedding is the correct size
    if len(embedding) != vector_size:
        logging.warning(
            f"Embedding size mismatch for {image_path.name} (Expected {vector_size}, Got {len(embedding)}). Skipping page."
        )
        return False

    # c. Upsert to Qdrant
    upsert_to_qdrant(
        qdrant_client,
        args.qdrant_collection,
        doc_path,
        page_num,
        transcription,
        embedding,
    )
    logging.info(f"Successfully processed page {page_num} of {doc_path.name}")
    return True


# --- Main Orchestration ---


def main(args):
    """Main function to orchestrate the document processing pipeline."""
    start_time = time.time()
    logging.info("Starting document processing pipeline...")

    # 1. Find documents
    documents = find_documents(args.docs_folder)
    if not documents:
        logging.warning("No documents found. Exiting.")
        return

    # 2. Setup Qdrant client
    # Get embedding dimension from Ollama first (assuming it's consistent)
    logging.info("Determining embedding vector size from Ollama model...")
    sample_embedding = get_embedding_ollama(
        "test", args.ollama_endpoint, args.ollama_model
    )
    if not sample_embedding:
        logging.error(
            "Could not get sample embedding from Ollama to determine vector size. Exiting."
        )
        return
    vector_size = len(sample_embedding)
    logging.info(f"Determined embedding vector size: {vector_size}")

    qdrant_client = setup_qdrant_client(
        args.qdrant_host, args.qdrant_port, args.qdrant_collection, vector_size
    )
    if not qdrant_client:
        logging.error("Failed to initialize Qdrant client. Exiting.")
        return

    processed_files = 0
    processed_pages = 0
    errors_occurred = 0

    # 3. Process each document sequentially
    for doc_path in documents:
        logging.info(f"--- Starting processing document: {doc_path.name} ---")
        doc_images_output_dir = Path(args.images_output) / doc_path.stem
        doc_images_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert document to images
        image_paths = []
        if doc_path.suffix.lower() == ".pdf":
            image_paths = convert_pdf_to_images(doc_path, doc_images_output_dir)
        elif doc_path.suffix.lower() == ".docx":
            image_paths = convert_docx_to_images(doc_path, doc_images_output_dir)

        if not image_paths:
            logging.warning(f"No images generated for {doc_path.name}. Skipping.")
            errors_occurred += 1
            continue

        # 4. Process pages in parallel (5 at a time)
        logging.info(
            f"Processing {len(image_paths)} pages from {doc_path.name} with maximum 5 pages in parallel"
        )

        doc_pages_processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create a list to store the futures
            future_to_page = {}

            # Submit each page for processing
            for i, image_path in enumerate(image_paths):
                page_num = i + 1
                future = executor.submit(
                    process_page,
                    doc_path,
                    image_path,
                    page_num,
                    args,
                    qdrant_client,
                    vector_size,
                )
                future_to_page[future] = (image_path, page_num)

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                image_path, page_num = future_to_page[future]
                try:
                    success = future.result()
                    if success:
                        doc_pages_processed += 1
                    else:
                        errors_occurred += 1
                except Exception as e:
                    logging.error(
                        f"Error processing page {page_num} of {doc_path.name}: {e}"
                    )
                    errors_occurred += 1

        processed_pages += doc_pages_processed
        processed_files += 1
        logging.info(
            f"--- Finished document: {doc_path.name} - Processed {doc_pages_processed} of {len(image_paths)} pages ---"
        )

    end_time = time.time()
    duration = end_time - start_time
    logging.info("--- Pipeline Summary ---")
    logging.info(f"Total documents found: {len(documents)}")
    logging.info(f"Successfully processed documents: {processed_files}")
    logging.info(
        f"Successfully processed pages (transcribed, embedded, upserted): {processed_pages}"
    )
    logging.info(f"Errors encountered (skipped files/pages): {errors_occurred}")
    logging.info(f"Total processing time: {duration:.2f} seconds")
    logging.info(f"Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline: PDF/DOCX -> Images -> vLLM Transcription -> Ollama Embedding -> Qdrant Upsert"
    )

    # Input/Output Folders
    parser.add_argument(
        "--docs-folder",
        type=str,
        required=True,
        help="Path to the folder containing input PDF and DOCX files.",
    )
    parser.add_argument(
        "--images-output",
        type=str,
        required=True,
        help="Path to the base folder where page images will be saved (subfolders per document).",
    )

    # vLLM Configuration
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default="http://localhost:8000",
        help="URL of the vLLM server (OpenAI API compatible endpoint base).",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        required=True,
        help="Name of the vision model served by vLLM for transcription.",
    )

    # Ollama Configuration
    parser.add_argument(
        "--ollama-endpoint",
        type=str,
        default="http://localhost:11434",
        help="URL of the Ollama server.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        required=True,
        help="Name of the embedding model served by Ollama.",
    )

    # Qdrant Configuration
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Hostname of the Qdrant server.",
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=6333, help="Port of the Qdrant server."
    )
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="document_embeddings",
        help="Name of the Qdrant collection to use.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
