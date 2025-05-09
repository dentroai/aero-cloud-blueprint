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
import logging
import time
import concurrent.futures
from threading import Lock
import numpy as np
import easyocr
import re
import math
import datetime

# SQLAlchemy and pgvector imports
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    UniqueConstraint,
    LargeBinary,
)
from sqlalchemy.orm import sessionmaker, Session as DBSession
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add this near the top of the file with other imports
reader_lock = Lock()
ocr_readers = {}

# --- SQLAlchemy Setup ---
Base = declarative_base()
engine = None
# SessionLocal will be our session factory, defined after engine setup
SessionLocal = None
# DocumentChunk model will be defined dynamically once vector_size is known
DocumentChunk = None


def define_document_chunk_model(vector_size: int):
    """
    Dynamically defines the DocumentChunk SQLAlchemy model class,
    aligning with Flowise's expected 'documents' table structure.
    """
    global DocumentChunk
    if (
        DocumentChunk is not None
        and hasattr(DocumentChunk.embedding.type, "dim")
        and DocumentChunk.embedding.type.dim == vector_size
        and DocumentChunk.__tablename__ == "documents"
    ):
        logging.debug(
            f"DocumentChunk model (as 'documents') already defined with vector size {vector_size}."
        )
        return DocumentChunk

    logging.info(
        f"Defining DocumentChunk model (as 'documents') with vector size {vector_size}."
    )

    class FlowiseDocument(Base):
        __tablename__ = "documents"

        id = Column(String, primary_key=True, index=True)
        content = Column(Text, nullable=False)
        doc_metadata = Column(JSONB, nullable=True)
        embedding = Column(Vector(vector_size), nullable=False)
        image_data = Column(LargeBinary, nullable=True)
        timestamp = Column(
            DateTime(timezone=True),
            default=lambda: datetime.datetime.now(datetime.UTC),
            onupdate=lambda: datetime.datetime.now(datetime.UTC),
        )

        def __repr__(self):
            return (
                f"<FlowiseDocument(id={self.id}, doc_metadata='{self.doc_metadata}')>"
            )

    DocumentChunk = FlowiseDocument
    return DocumentChunk


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


# --- EasyOCR to Markdown Conversion ---

# Configuration Constants for OCR to Markdown conversion
OCR_MD_CFG = {
    # Line building
    "y_tolerance_factor": 0.6,  # Max vertical distance factor (rel. to avg_h) for same line
    "merge_gap_factor": 0.15,  # Max horizontal gap factor (rel. to avg_char_w) to merge boxes
    # Semantic tagging
    "rule_height_factor": 0.5,  # Max height factor (rel. to avg_h) for a box to be a rule
    "rule_width_factor": 0.75,  # Min width factor (rel. to page_w) for a box to be a rule
    "heading_height_factor": 1.3,  # Min height factor (rel. to avg_h) for a line to be a heading
    "indent_chars": 4,  # Number of avg char widths per indentation level
    # Table detection
    "table_align_factor": 1.0,  # Max horizontal alignment diff factor (rel. to avg_char_w)
    "table_min_rows": 3,  # Minimum consecutive rows to form a table
    # Markdown rendering
    "md_indent_spaces": 2,  # Spaces per indentation level in Markdown output
}


def norm_box(box, text, conf):
    """Return a flat dict with useful geometry."""
    pts = np.asarray(box, dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    return dict(
        text=text.strip(),
        conf=conf,
        xmin=x0,
        xmax=x1,
        ymin=y0,
        ymax=y1,
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
        w=w,
        h=h,
        box=pts,
    )


def char_width(records):
    """Rough average character width on the page."""
    tot_w = tot_c = 0
    for r in records:
        if r["text"]:
            tot_w += r["w"]
            tot_c += len(r["text"])
    return (tot_w / tot_c) if tot_c else 10


def build_lines(records, avg_h, c_w, cfg):
    """Return list[Line]; each Line = dict(segments=[…], …)."""
    recs = sorted(records, key=lambda r: r["cy"])
    y_tol = cfg["y_tolerance_factor"] * avg_h
    merge_tol = cfg["merge_gap_factor"] * c_w

    lines, cur = [], []
    last_y = None
    for r in recs:
        if last_y is None or abs(r["cy"] - last_y) <= y_tol:
            cur.append(r)
        else:
            lines.append(cur)
            cur = [r]
        last_y = r["cy"]
    if cur:
        lines.append(cur)

    # sort (and optionally merge touching boxes) inside each line
    out = []
    for segs in lines:
        segs.sort(key=lambda r: r["xmin"])
        merged = [segs[0]]
        for r in segs[1:]:
            prev = merged[-1]
            gap = r["xmin"] - prev["xmax"]
            if gap < merge_tol:  # very narrow gap → merge
                prev["text"] += " " + r["text"]
                prev["xmax"] = r["xmax"]
                prev["w"] = prev["xmax"] - prev["xmin"]
            else:
                merged.append(r)
        text = " ".join(s["text"] for s in merged)
        out.append(
            dict(
                segments=merged,
                text=text,
                xmin=min(s["xmin"] for s in merged),
                xmax=max(s["xmax"] for s in merged),
                cy=np.mean([s["cy"] for s in merged]),
                hmax=max(s["h"] for s in merged),
            )
        )
    return out


def tag_lines(lines, avg_h, c_w, page_w, cfg):
    bullet_rx = re.compile(r"^\s*(?:[\-\u2022\*]|(\[.?]))")
    indent_tab = cfg["indent_chars"] * c_w
    for L in lines:
        txt = L["text"]
        L["indent_lvl"] = round(L["xmin"] / indent_tab) if indent_tab > 0 else 0
        L["rule"] = (
            L["hmax"] < cfg["rule_height_factor"] * avg_h
            and (L["xmax"] - L["xmin"]) > cfg["rule_width_factor"] * page_w
            and re.fullmatch(r"[ \-\u2013\u2014_]+", txt)
        )
        L["heading"] = L["hmax"] > cfg["heading_height_factor"] * avg_h or (
            txt.isupper() and len(txt) > 2
        )
        L["bullet"] = bool(bullet_rx.match(txt))
        L["kv"] = ":" in txt and not L["bullet"]
    return lines


def detect_tables(lines, c_w, cfg):
    """Return list of (start_idx, end_idx, n_cols) table blocks."""
    tables = []
    i = 0
    align_tol = cfg["table_align_factor"] * c_w
    while i < len(lines):
        nseg = len(lines[i]["segments"])
        if nseg < 2:
            i += 1
            continue
        # try to extend a group with same col count & aligned x
        ref_x = [s["xmin"] for s in lines[i]["segments"]]
        j = i + 1
        while (
            j < len(lines)
            and len(lines[j]["segments"]) == nseg
            and all(
                abs(lines[j]["segments"][k]["xmin"] - ref_x[k]) < align_tol
                for k in range(nseg)
            )
        ):
            j += 1
        if j - i >= cfg["table_min_rows"]:  # need min rows → table
            tables.append((i, j, nseg))
            i = j
        else:
            i += 1
    return tables


def render_md(lines, tables, cfg):
    tbl_bounds = {idx for s, e, _ in tables for idx in range(s, e)}
    md, i = [], 0
    indent_spaces = cfg["md_indent_spaces"]
    while i < len(lines):
        L = lines[i]
        if i in tbl_bounds:
            # find which table block we are in
            tb = next((t for t in tables if t[0] <= i < t[1]), None)
            s, e, n = tb

            # escape pipes in cell content
            def escape(cell_text):
                return cell_text.replace("|", "\\|")

            # header + separator + rows
            header = (
                "|" + "|".join(escape(l["text"]) for l in lines[s]["segments"]) + "|"
            )
            sep = "|" + "|".join("---" for _ in range(n)) + "|"
            md.append(header)
            md.append(sep)
            for r in range(s + 1, e):
                row = (
                    "|"
                    + "|".join(escape(l["text"]) for l in lines[r]["segments"])
                    + "|"
                )
                md.append(row)
            i = e
            continue

        indent = " " * (indent_spaces * L["indent_lvl"])
        if L["rule"]:
            md.append("\n---\n")
        elif L["heading"]:
            level = 1 if L["indent_lvl"] == 0 else 2
            md.append("#" * level + " " + L["text"])
        elif L["bullet"]:
            body = re.sub(r"^\s*[\-\u2022\*]\s*", "", L["text"])
            md.append(f"{indent}- {body}")
        elif L["kv"]:
            lbl, val = map(str.strip, L["text"].split(":", 1))
            md.append(f"{indent}{lbl}: **{val}**")
        else:
            md.append(f"{indent}{L['text']}")
        i += 1
    return "\n".join(md)


def page_width(records):
    if not records:
        return 0
    return max(r["xmax"] for r in records) - min(r["xmin"] for r in records)


def transcribe_image_easyocr(image_path: Path, language: str) -> str:
    """Transcribes an image using easyOCR and converts to markdown."""
    logging.info(
        f"Transcribing image {image_path.name} using easyOCR with language: {language}"
    )

    # Get or create reader with thread safety
    global ocr_readers
    reader_key = language

    try:
        # First try to get existing reader
        reader = None
        with reader_lock:
            if reader_key in ocr_readers:
                reader = ocr_readers[reader_key]
            else:
                # Create new reader (model download happens here)
                logging.info(
                    f"Initializing new easyOCR reader for language: {language} (with GPU support)"
                )
                reader = easyocr.Reader([language], gpu=True, download_enabled=True)
                ocr_readers[reader_key] = reader

        # Perform OCR
        raw = reader.readtext(str(image_path), detail=1, paragraph=False)
        if not raw:
            logging.warning(f"No text detected in {image_path.name}")
            return ""

        records = [norm_box(b, t, c) for b, t, c in raw if t.strip()]
        if not records:
            logging.warning(f"No non-whitespace text detected in {image_path.name}")
            return ""

        # Process OCR results into markdown
        avg_h = np.mean([r["h"] for r in records if r["h"] > 0]) if records else 10
        c_w = char_width(records)
        pw = page_width(records)

        lines = build_lines(records, avg_h, c_w, OCR_MD_CFG)
        lines = tag_lines(lines, avg_h, c_w, pw, OCR_MD_CFG)
        tables = detect_tables(lines, c_w, OCR_MD_CFG)

        markdown_text = render_md(lines, tables, OCR_MD_CFG)
        logging.info(f"Successfully transcribed {image_path.name} to markdown")

        return markdown_text
    except Exception as e:
        logging.error(f"Error transcribing {image_path.name} with easyOCR: {e}")
        return ""


# --- AI Model Interactions ---


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


def setup_postgres_connection(db_url: str):
    """Initializes SQLAlchemy engine and ensures the table exists."""
    global engine, SessionLocal, Base, DocumentChunk

    if DocumentChunk is None:
        raise RuntimeError(
            "DocumentChunk model is not defined. Call define_document_chunk_model first with the correct vector size."
        )

    try:
        engine = create_engine(db_url)
        Base.metadata.create_all(bind=engine)  # Creates tables if they don't exist
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logging.info(
            f"Connected to PostgreSQL and ensured table '{DocumentChunk.__tablename__}' exists."
        )
        return SessionLocal
    except Exception as e:
        logging.error(f"Error connecting to or setting up PostgreSQL: {e}")
        return None


def upsert_to_postgres(
    db: DBSession,
    doc_path: Path,
    page_num: int,
    transcription: str,
    embedding_vector: list[float],
    image_bytes: bytes | None,
):
    """Upserts the document page data to PostgreSQL into the 'documents' table."""
    global DocumentChunk
    if DocumentChunk is None:
        logging.error(
            "DocumentChunk model (for 'documents' table) not available for upsert."
        )
        return

    if not embedding_vector:
        logging.warning(
            f"Skipping upsert for {doc_path.name} page {page_num} due to missing embedding."
        )
        return

    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_path.resolve()}_page_{page_num}")
    )

    # Prepare metadata
    page_doc_metadata = {
        "file_path": str(doc_path.resolve()),
        "file_name": doc_path.name,
        "page_number": page_num,
        "original_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    try:
        existing_chunk = db.query(DocumentChunk).filter_by(id=point_id).first()

        if existing_chunk:
            logging.debug(
                f"Updating existing document chunk {point_id} for {doc_path.name} page {page_num}"
            )
            existing_chunk.content = transcription
            existing_chunk.embedding = embedding_vector
            existing_chunk.doc_metadata = page_doc_metadata
            existing_chunk.image_data = image_bytes
        else:
            logging.debug(
                f"Inserting new document chunk {point_id} for {doc_path.name} page {page_num}"
            )
            new_chunk = DocumentChunk(
                id=point_id,
                content=transcription,
                embedding=embedding_vector,
                doc_metadata=page_doc_metadata,
                image_data=image_bytes,
            )
            db.add(new_chunk)

        db.commit()
        logging.debug(
            f"Successfully upserted document chunk {point_id} for {doc_path.name} page {page_num}"
        )

    except Exception as e:
        db.rollback()
        logging.error(f"Error upserting point {point_id} to PostgreSQL: {e}")
        # Optionally re-raise or handle more gracefully depending on desired behavior
        # For now, log and continue with other pages/docs.


# --- Parallel Processing ---


def process_page(
    doc_path: Path,
    image_path: Path,
    page_num: int,
    args,
    db_session_factory,
    vector_size,
):
    """Process a single page of a document."""
    logging.info(
        f"Processing page {page_num} of document {doc_path.name} ({image_path.name})"
    )

    db = None  # Initialize db to None for the finally block
    try:
        db = db_session_factory()  # Create a new session for this thread/task

        # a. Transcribe image using easyOCR
        transcription = transcribe_image_easyocr(image_path, args.ocr_language)
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

        # c. Read image file into bytes
        image_bytes = None
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            logging.debug(f"Read {len(image_bytes)} bytes from image {image_path.name}")
        except Exception as e:
            logging.warning(
                f"Could not read image file {image_path.name}: {e}. Storing page without image data."
            )
            # Continue without image_bytes if reading fails, or handle more strictly if needed

        # Ensure embedding is the correct size (already checked against vector_size)
        if len(embedding) != vector_size:
            logging.warning(
                f"Embedding size mismatch for {image_path.name} (Expected {vector_size}, Got {len(embedding)}). Skipping page."
            )
            return False

        # d. Upsert to PostgreSQL
        upsert_to_postgres(
            db,
            doc_path,
            page_num,
            transcription,
            embedding,
            image_bytes,
        )
        logging.info(f"Successfully processed page {page_num} of {doc_path.name}")
        return True
    except Exception as e:
        logging.error(
            f"Unhandled error in process_page for {image_path.name} page {page_num}: {e}"
        )
        if db:
            db.rollback()  # Rollback session on error
        return False
    finally:
        if db:
            db.close()  # Ensure session is closed


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

    # 2. Determine embedding vector size from Ollama model
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

    # 3. Define the DocumentChunk model with the determined vector size
    # This function sets the global 'DocumentChunk' variable.
    define_document_chunk_model(vector_size)

    # 4. Setup PostgreSQL connection
    if not args.postgres_password:
        # Try to get from environment if not provided via CLI
        args.postgres_password = os.getenv("POSTGRES_PASSWORD")
        if not args.postgres_password:
            logging.error(
                "PostgreSQL password is required. Set via --postgres-password or POSTGRES_PASSWORD env var."
            )
            return

    db_url = f"postgresql+psycopg2://{args.postgres_user}:{args.postgres_password}@{args.postgres_host}:{args.postgres_port}/{args.postgres_db}"

    db_session_factory = setup_postgres_connection(db_url)
    if not db_session_factory:
        logging.error("Failed to initialize PostgreSQL connection. Exiting.")
        return

    # Preload easyOCR model before parallel processing
    logging.info(f"Preloading easyOCR model for language: {args.ocr_language}...")
    try:
        # Initialize the OCR reader manually - this will trigger model download if needed
        reader_key = args.ocr_language
        with reader_lock:
            if reader_key not in ocr_readers:
                ocr_readers[reader_key] = easyocr.Reader(
                    [args.ocr_language], gpu=True, download_enabled=True
                )
        logging.info("easyOCR model preloaded successfully.")
    except Exception as e:
        logging.error(f"Error preloading easyOCR model: {e}")
        logging.warning("Continuing with processing, but watch for OCR errors.")

    processed_files = 0
    processed_pages = 0
    errors_occurred = 0

    # 5. Process each document sequentially
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

        # 6. Process pages in parallel (X at a time)
        logging.info(
            f"Processing {len(image_paths)} pages from {doc_path.name} with maximum {args.max_workers} pages in parallel"
        )

        doc_pages_processed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
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
                    db_session_factory,  # Pass the session factory
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
        description="Document Processing Pipeline: PDF/DOCX -> Images -> easyOCR -> Markdown -> Ollama Embedding -> PostgreSQL Upsert"
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

    # OCR Configuration
    parser.add_argument(
        "--ocr-language",
        type=str,
        default="en",
        help="Language for OCR processing (e.g., 'en', 'de', 'fr')",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,  # Default to number of CPUs
        help="Maximum number of worker threads for parallel page processing",
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

    # PostgreSQL Configuration
    parser.add_argument(
        "--postgres-host",
        type=str,
        default="postgres",
        help="Hostname of the PostgreSQL server. Defaults to 'postgres' or POSTGRES_HOST env var.",
    )
    parser.add_argument(
        "--postgres-port",
        type=int,
        default=int(os.getenv("POSTGRES_PORT", 5432)),
        help="Port of the PostgreSQL server. Defaults to 5432 or POSTGRES_PORT env var.",
    )
    parser.add_argument(
        "--postgres-user",
        type=str,
        default=os.getenv("POSTGRES_USER", "rag_user"),
        help="Username for PostgreSQL. Defaults to 'rag_user' or POSTGRES_USER env var.",
    )
    parser.add_argument(
        "--postgres-password",
        type=str,
        default=None,  # Will check env var POSTGRES_PASSWORD if not set
        help="Password for PostgreSQL. Best practice to set via POSTGRES_PASSWORD env var.",
    )
    parser.add_argument(
        "--postgres-db",
        type=str,
        default=os.getenv("POSTGRES_DB", "rag_db"),
        help="Database name for PostgreSQL. Defaults to 'rag_db' or POSTGRES_DB env var.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
