"""FastAPI ingest routes."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from agents import get_agent
from models.schemas import IngestRequest, IngestResult, SourceType
from scrapers.base import ScrapedDocument
from utils.logger import logger
from utils.parsers import parse_file
from workflows.ingest_workflow import ingest_document

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.post("", response_model=IngestResult)
async def ingest(req: IngestRequest) -> IngestResult:
    """Ingest content from a URL or file path."""
    target = (str(req.url) if req.url else None) or req.file_path
    if not target:
        raise HTTPException(status_code=400, detail="url or file_path is required")
    try:
        return await get_agent().ingest(req.source_type, target, **req.extra)
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload", response_model=IngestResult)
async def ingest_upload(
    file: Annotated[UploadFile, File(...)],
    source_type: Annotated[SourceType, Form(...)] = SourceType.UPLOAD,
) -> IngestResult:
    """Upload a file (PDF/DOCX/MD/TXT) and ingest it."""
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        if source_type == SourceType.PDF or dest.suffix.lower() == ".pdf":
            return await get_agent().ingest(SourceType.PDF, str(dest))
        # generic file: parse + ingest as document
        text = parse_file(dest)
        doc = ScrapedDocument(
            title=dest.stem,
            content=text,
            source=str(dest),
            source_type=SourceType.UPLOAD.value,
            metadata={"filename": file.filename, "size": dest.stat().st_size},
        )
        return await ingest_document(doc)
    except Exception as exc:
        logger.exception("Upload ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))
