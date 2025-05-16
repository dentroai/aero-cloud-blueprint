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
import io
from PIL import Image

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
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add this near the top of the file with other imports
reader_lock = Lock()
ocr_readers = {}

# Add this after imports
MIME_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# --- Configuration Constants (Add this new constant) ---
MAX_IMAGE_DIMENSION_ETL = 1024  # Max dimension for stored images (longest side)
TARGET_IMAGE_FORMAT_FOR_STORAGE = "JPEG"  # e.g., JPEG, PNG
TARGET_IMAGE_QUALITY = 85  # For JPEG, 1-95 (higher is better quality, larger size)

# --- SQLAlchemy Setup ---
Base = declarative_base()
engine = None
# SessionLocal will be our session factory, defined after engine setup
SessionLocal = None
# DocumentChunk model will be defined dynamically once vector_size is known
DocumentChunk = None


def define_document_chunk_model(text_vector_size: int, image_vector_size: int | None):
    """
    Dynamically defines the DocumentChunk SQLAlchemy model class,
    aligning with Flowise's expected 'documents' table structure,
    and adding fields for image embedding and image data.
    """
    global DocumentChunk
    # Check if model needs redefinition (e.g., different vector sizes)
    # This check is simplified; a more robust check would compare all relevant attributes.
    if (
        DocumentChunk is not None
        and hasattr(DocumentChunk.embedding.type, "dim")
        and DocumentChunk.embedding.type.dim == text_vector_size
        and (
            (
                image_vector_size is None
                and not hasattr(DocumentChunk, "image_embedding")
            )
            or (
                image_vector_size is not None
                and hasattr(DocumentChunk, "image_embedding")
                and DocumentChunk.image_embedding.type.dim == image_vector_size
            )
        )
        and DocumentChunk.__tablename__ == "documents"
    ):
        logging.debug(
            f"DocumentChunk model (as 'documents') already defined with text vector size {text_vector_size} and image vector size {image_vector_size}."
        )
        return DocumentChunk

    logging.info(
        f"Defining DocumentChunk model (as 'documents') with text vector size {text_vector_size} and image vector size {image_vector_size}."
    )

    class FlowiseDocument(Base):
        __tablename__ = "documents"

        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        pageContent = Column("pageContent", Text, nullable=False)
        doc_metadata = Column("metadata", JSONB, nullable=True)
        embedding = Column(Vector(text_vector_size), nullable=False)  # Text embedding
        if image_vector_size is not None:
            image_embedding = Column(
                Vector(image_vector_size), nullable=True
            )  # Image embedding
        image_data = Column(LargeBinary, nullable=True)  # Raw image data

        # Add a unique constraint on 'id' if not already implied by primary_key, though it usually is.
        # __table_args__ = (UniqueConstraint('id', name='uq_documents_id'),)

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


def get_embedding_vllm_text(
    text: str, vllm_endpoint: str, model_name: str
) -> list[float] | None:
    """Gets text embedding from a vLLM OpenAI-compatible API endpoint."""
    logging.debug(
        f"Getting text embedding for snippet (length: {len(text)}) using vLLM model {model_name} at {vllm_endpoint}"
    )
    # Assuming vLLM endpoint is like "http://host:port/v1" and we need "/embeddings"
    api_url = f"{vllm_endpoint.rstrip('/')}/embeddings"
    payload = {"input": text, "model": model_name}

    try:
        response = requests.post(
            api_url, json=payload, timeout=120
        )  # Increased timeout for potentially larger models/payloads
        response.raise_for_status()
        data = response.json()
        if (
            "data" in data
            and isinstance(data["data"], list)
            and len(data["data"]) > 0
            and "embedding" in data["data"][0]
            and isinstance(data["data"][0]["embedding"], list)
        ):
            return data["data"][0]["embedding"]
        else:
            logging.error(
                f"vLLM text embedding response missing 'data[0].embedding' or not a list. Response: {data}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling vLLM text embedding API at {api_url}: {e}")
        logging.error(
            f"Response content: {response.content if 'response' in locals() and hasattr(response, 'content') else 'N/A'}"
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during vLLM text embedding: {e}")
    return None


def get_mime_type(image_path):
    """Determines the MIME type of an image based on its extension."""
    ext = Path(image_path).suffix
    return MIME_TYPE_MAP.get(ext.lower())


def resize_and_format_image_bytes(
    image_bytes: bytes,
    max_dimension: int,
    target_format: str = "JPEG",
    target_quality: int = 85,
) -> bytes | None:
    """
    Resizes image bytes if dimensions exceed max_dimension, converts to target_format,
    and returns new image bytes.
    """
    if not image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_size = img.size
        original_mode = img.mode

        # Resize if necessary
        if max(img.width, img.height) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logging.debug(
                f"Resized image from {original_size} to {img.size} (max_dim: {max_dimension})"
            )

        # Handle image mode and transparency for target format
        # Convert to RGB if it's RGBA or P (palette) for JPEG, or if it's an unusual mode for PNG
        if target_format == "JPEG":
            if img.mode == "RGBA" or img.mode == "P":
                img = img.convert("RGB")
        elif target_format == "PNG":
            # PNG can handle RGB, RGBA, L, P. If it's something else, convert to RGB/RGBA.
            if img.mode not in ("RGB", "RGBA", "L", "P"):
                img = img.convert(
                    "RGBA" if "A" in original_mode else "RGB"
                )  # Preserve alpha if original had it and PNG supports it

        # Save to buffer in the target format
        buffered = io.BytesIO()
        save_kwargs = {}
        if target_format == "JPEG":
            save_kwargs["quality"] = target_quality
            save_kwargs["optimize"] = True  # Try to optimize JPEG size

        img.save(buffered, format=target_format, **save_kwargs)
        processed_bytes = buffered.getvalue()
        logging.debug(
            f"Formatted image to {target_format}. Original size: {len(image_bytes)}, New size: {len(processed_bytes)}"
        )
        return processed_bytes

    except Exception as e:
        logging.error(f"Error processing image bytes for resizing/formatting: {e}")
        return image_bytes  # Return original bytes on error to avoid losing the image entirely


def get_embedding_vllm_image(
    image_bytes: bytes,
    vllm_endpoint: str,
    model_name: str,
    image_format: str = ".png",
) -> list[float] | None:
    """
    Get an IMAGE embedding from vLLM.

    The image model expects the OpenAI-style "messages" payload (image_url +
    short text) but is served on the normal /v1/embeddings route.
    """
    logging.debug(
        f"Requesting image embedding (size {len(image_bytes)} bytes) from {vllm_endpoint} using model {model_name}"
    )

    # ------------------------------------------------------------------ #
    # Build data-URI                                                      #
    # ------------------------------------------------------------------ #
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logging.error(f"Base64-encoding failed: {e}")
        return None

    mime = MIME_TYPE_MAP.get(image_format.lower(), "image/png")
    data_url = f"data:{mime};base64,{b64}"

    # ------------------------------------------------------------------ #
    # Endpoint – make sure we actually hit  …/v1/embeddings               #
    # ------------------------------------------------------------------ #
    if vllm_endpoint.endswith("/v1/embeddings"):
        api_url = vllm_endpoint
    elif vllm_endpoint.endswith("/v1"):
        api_url = f"{vllm_endpoint}/embeddings"
    else:
        api_url = f"{vllm_endpoint.rstrip('/')}/v1/embeddings"

    # ------------------------------------------------------------------ #
    # Payload in "messages" format (same as working PoC)                  #
    # ------------------------------------------------------------------ #
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Generate an embedding for this image."},
                ],
            }
        ],
        "encoding_format": "float",
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        # Error handling ------------------------------------------------
        if data.get("object") == "error" or "error" in data:
            msg = data.get("message") or data.get("error", {}).get("message")
            logging.error(f"vLLM image-embedding API error: {msg}")
            return None

        # Success path --------------------------------------------------
        if (
            "data" in data
            and isinstance(data["data"], list)
            and data["data"]
            and "embedding" in data["data"][0]
        ):
            embedding_result = data["data"][0]["embedding"]
            logging.info(
                f"Successfully generated image embedding from vLLM model {model_name}. Vector size: {len(embedding_result)}"
            )
            return embedding_result

        # Fallbacks (rare) ---------------------------------------------
        if "embedding" in data:  # some models
            embedding_result = data["embedding"]
            logging.info(
                f"Successfully generated image embedding (fallback path) from vLLM model {model_name}. Vector size: {len(embedding_result)}"
            )
            return embedding_result
        if isinstance(data, list):  # raw list of floats
            logging.info(
                f"Successfully generated image embedding (raw list fallback) from vLLM model {model_name}. Vector size: {len(data)}"
            )
            return data

        logging.error(f"Unexpected embedding response shape: {data}")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error from vLLM image-embedding API: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in image-embedding helper: {e}")

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
    text_embedding_vector: list[float],
    image_embedding_vector: list[float] | None,
    image_bytes: bytes | None,
):
    """Upserts the document page data to PostgreSQL into the 'documents' table."""
    global DocumentChunk
    if DocumentChunk is None:
        logging.error(
            "DocumentChunk model (for 'documents' table) not available for upsert."
        )
        return

    if not text_embedding_vector:
        logging.warning(
            f"Skipping upsert for {doc_path.name} page {page_num} due to missing text embedding."
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
            existing_chunk.pageContent = transcription
            existing_chunk.embedding = text_embedding_vector
            existing_chunk.doc_metadata = page_doc_metadata
            if hasattr(existing_chunk, "image_embedding"):
                existing_chunk.image_embedding = image_embedding_vector
            if hasattr(existing_chunk, "image_data"):
                existing_chunk.image_data = image_bytes
        else:
            logging.debug(
                f"Inserting new document chunk {point_id} for {doc_path.name} page {page_num}"
            )
            chunk_data = {
                "id": point_id,
                "pageContent": transcription,
                "embedding": text_embedding_vector,
                "doc_metadata": page_doc_metadata,
            }
            if image_embedding_vector is not None and hasattr(
                DocumentChunk, "image_embedding"
            ):
                chunk_data["image_embedding"] = image_embedding_vector
            if image_bytes is not None and hasattr(DocumentChunk, "image_data"):
                chunk_data["image_data"] = image_bytes

            new_chunk = DocumentChunk(**chunk_data)
            db.add(new_chunk)

        db.commit()
        logging.debug(
            f"Successfully upserted document chunk {point_id} for {doc_path.name} page {page_num}"
        )
        # Add a more specific info log
        log_message_parts = [
            f"Stored page {page_num} of '{doc_path.name}' to PostgreSQL:"
        ]
        if text_embedding_vector:
            log_message_parts.append("Text embedding (yes)")
        if image_embedding_vector:
            log_message_parts.append("Image embedding (yes)")
        else:
            log_message_parts.append("Image embedding (no)")
        if image_bytes:
            log_message_parts.append("Raw image data (yes)")
        else:
            log_message_parts.append("Raw image data (no)")
        logging.info(" ".join(log_message_parts))

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
    text_vector_size: int,
    image_vector_size: int | None,
):
    """Process a single page of a document."""
    logging.info(
        f"Processing page {page_num} of document {doc_path.name} ({image_path.name})"
    )

    db = None
    try:
        db = db_session_factory()

        transcription = transcribe_image_easyocr(image_path, args.ocr_language)
        if not transcription:
            logging.warning(f"Failed to transcribe {image_path.name}. Skipping page.")
            return False

        text_embedding = get_embedding_vllm_text(
            transcription,
            args.vllm_text_embedding_endpoint,
            args.vllm_text_embedding_model,
        )
        if not text_embedding:
            logging.warning(
                f"Failed to get text embedding for transcription of {image_path.name}. Skipping page."
            )
            return False

        if len(text_embedding) != text_vector_size:
            logging.warning(
                f"Text embedding size mismatch for {image_path.name} (Expected {text_vector_size}, Got {len(text_embedding)}). Skipping page."
            )
            return False

        processed_image_bytes = None
        original_image_bytes = None
        try:
            with open(image_path, "rb") as f:
                original_image_bytes = f.read()

            if original_image_bytes:
                logging.debug(
                    f"Read {len(original_image_bytes)} bytes from image {image_path.name}. Processing for storage..."
                )
                # Resize and reformat the image before embedding or storing
                processed_image_bytes = resize_and_format_image_bytes(
                    original_image_bytes,
                    MAX_IMAGE_DIMENSION_ETL,  # Use the new constant
                    target_format=TARGET_IMAGE_FORMAT_FOR_STORAGE,
                    target_quality=TARGET_IMAGE_QUALITY,
                )
                if not processed_image_bytes:  # Fallback if processing failed
                    logging.warning(
                        f"Image processing failed for {image_path.name}, using original image bytes if available."
                    )
                    processed_image_bytes = original_image_bytes
            else:
                logging.warning(
                    f"Could not read image file {image_path.name}. Image-related data will be skipped."
                )

        except Exception as e:
            logging.warning(
                f"Could not read or process image file {image_path.name}: {e}. Image-related data will be skipped."
            )

        image_embedding = None
        # Use processed_image_bytes for embedding if available
        if processed_image_bytes and image_vector_size is not None:
            image_format_for_embedding = Path(
                image_path
            ).suffix  # Original suffix for MIME type hints to vLLM
            # Some embedding models might be sensitive to format, even if they take bytes.
            # The actual bytes sent are now from processed_image_bytes (e.g., JPEG)
            # but we tell vLLM the original format for its "data:mime;base64" hint.
            # This might need adjustment if vLLM strictly requires the MIME type of the *actual* bytes being sent.
            # For now, let's assume the original suffix is a good enough hint.
            image_embedding = get_embedding_vllm_image(
                processed_image_bytes,  # Use the potentially resized/reformatted bytes
                args.vllm_image_embedding_endpoint,
                args.vllm_image_embedding_model,
                image_format_for_embedding,
            )
            if not image_embedding:
                logging.warning(
                    f"Failed to get image embedding for {image_path.name}. Storing page without image embedding."
                )
            elif len(image_embedding) != image_vector_size:
                logging.warning(
                    f"Image embedding size mismatch for {image_path.name} (Expected {image_vector_size}, Got {len(image_embedding)}). Skipping image embedding."
                )
                image_embedding = None

        upsert_to_postgres(
            db,
            doc_path,
            page_num,
            transcription,
            text_embedding,
            image_embedding,
            # Store the processed (resized/reformatted) image bytes if store_raw_images is true
            processed_image_bytes if args.store_raw_images else None,
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

    # 2. Determine text embedding vector size from vLLM model
    logging.info(
        "Determining text embedding vector size from vLLM text embedding model..."
    )
    sample_text_embedding = get_embedding_vllm_text(
        "test", args.vllm_text_embedding_endpoint, args.vllm_text_embedding_model
    )
    if not sample_text_embedding:
        logging.error(
            "Could not get sample text embedding from vLLM to determine vector size. Exiting."
        )
        return
    text_vector_size = len(sample_text_embedding)
    logging.info(f"Determined text embedding vector size: {text_vector_size}")

    # 3. Determine image embedding vector size using ./test_image.png
    image_vector_size = None
    test_image_path = Path("./test_image.jpg")  # Define the path to the test image

    if args.vllm_image_embedding_model and args.vllm_image_embedding_endpoint:
        logging.info(
            f"Determining image embedding vector size using {test_image_path}..."
        )
        if not test_image_path.is_file():
            logging.warning(
                f"Test image {test_image_path} not found. "
                "Image embedding vector size cannot be determined. Image embeddings will be skipped."
            )
        else:
            try:
                with open(test_image_path, "rb") as f:
                    sample_image_bytes = f.read()

                sample_image_embedding = get_embedding_vllm_image(
                    sample_image_bytes,
                    args.vllm_image_embedding_endpoint,
                    args.vllm_image_embedding_model,
                    test_image_path.suffix,  # Pass the actual suffix (e.g., ".png")
                )
                if sample_image_embedding:
                    image_vector_size = len(sample_image_embedding)
                    logging.info(
                        f"Determined image embedding vector size: {image_vector_size}"
                    )
                else:
                    logging.warning(
                        f"Could not get sample image embedding from vLLM using {test_image_path}. "
                        "Image embeddings will not be generated or stored."
                    )
            except Exception as e:
                logging.error(
                    f"Error processing test image {test_image_path} for determining image embedding size: {e}. "
                    "Image embeddings will be skipped."
                )
    else:
        logging.info(
            "Image embedding model/endpoint not fully specified. Image embeddings will be skipped."
        )

    # 4. Define the DocumentChunk model with the determined vector sizes
    define_document_chunk_model(text_vector_size, image_vector_size)

    # 5. Setup PostgreSQL connection
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

    # 6. Process pages in parallel (X at a time)
    logging.info(
        f"Processing {len(documents)} documents with maximum {args.max_workers} pages in parallel"
    )

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

        # 7. Process pages in parallel (X at a time)
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
                    text_vector_size,
                    image_vector_size,
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
        description="Document Processing Pipeline: PDF/DOCX -> Images -> easyOCR -> Markdown -> vLLM Text & Image Embeddings -> PostgreSQL Upsert"
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
        "--store-raw-images",
        action="store_true",
        help="If set, store raw page image bytes in the database. Otherwise, only image embeddings (if configured).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 1,  # Default to number of CPUs
        help="Maximum number of worker threads for parallel page processing",
    )

    # vLLM Text Embedding Configuration
    parser.add_argument(
        "--vllm-text-embedding-endpoint",
        type=str,
        default="http://localhost:8001/v1",  # Default for running script locally
        help="URL of the vLLM text embedding server (OpenAI compatible API, e.g., http://localhost:8001/v1).",
    )
    parser.add_argument(
        "--vllm-text-embedding-model",
        type=str,
        required=True,
        help="Name of the text embedding model served by vLLM (e.g., text-embedding-model).",
    )

    # vLLM Image Embedding Configuration
    parser.add_argument(
        "--vllm-image-embedding-endpoint",
        type=str,
        default="http://localhost:8002/v1",  # Default for running script locally
        help="URL of the vLLM image embedding server (OpenAI compatible API, e.g., http://localhost:8002/v1).",
    )
    parser.add_argument(
        "--vllm-image-embedding-model",
        type=str,
        required=True,
        help="Name of the image embedding model served by vLLM (e.g., image-embedding-model). Should be a model with vision capabilities.",
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
