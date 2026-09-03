"""Multimodal source extraction for TrustRAG.

Text-like files are parsed locally. Images, audio and video are converted into a
faithful textual evidence representation with Gemini before entering the normal
TrustRAG chunk/retrieve/GroundCheck pipeline.
"""

from __future__ import annotations

import mimetypes
import os
import tempfile
import time
from pathlib import Path

from fastapi import HTTPException, UploadFile


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
}

IMAGE_PREFIX = "image/"
AUDIO_PREFIX = "audio/"
VIDEO_PREFIX = "video/"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB for this local demo
INLINE_GEMINI_LIMIT = 20 * 1024 * 1024  # 20 MB


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    raise HTTPException(
        status_code=400,
        detail="Could not decode this text file.",
    )


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(data))

        text = "\n\n".join(
            (page.extract_text() or "").strip()
            for page in reader.pages
        ).strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No selectable text was found in this PDF. "
                    "Try Gemini extraction with an image-based copy."
                ),
            )

        return text

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {exc}",
        ) from exc


def _gemini_extract(
    data: bytes,
    filename: str,
    mime_type: str,
    api_key: str,
    model: str,
) -> str:
    if not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "A Gemini API key is required to extract "
                "image, audio or video content."
            ),
        )

    try:
        from google import genai

    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ),
        ) from exc

    suffix = (
        Path(filename).suffix
        or mimetypes.guess_extension(mime_type)
        or ".bin"
    )

    temp_path = None
    uploaded = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp.write(data)
            temp_path = temp.name

        client = genai.Client(
            api_key=api_key
        )

        uploaded = client.files.upload(
            file=temp_path,
            config={
                "mime_type": mime_type,
                "display_name": filename,
            },
        )

        # Video/audio may briefly remain in PROCESSING state.
        for _ in range(60):
            state = str(
                getattr(
                    getattr(
                        uploaded,
                        "state",
                        None,
                    ),
                    "name",
                    getattr(
                        uploaded,
                        "state",
                        "",
                    ),
                )
            ).upper()

            if not state or state.endswith("ACTIVE"):
                break

            if state.endswith("FAILED"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Gemini could not process "
                        "this media file."
                    ),
                )

            time.sleep(1)

            uploaded = client.files.get(
                name=uploaded.name
            )

        extraction_prompt = (
            "Convert this uploaded source into a faithful textual evidence "
            "record for a retrieval-augmented generation system. "
            "Preserve all readable text, spoken words, labels, numbers, "
            "names, dates, and clearly observable facts. "
            "For audio, provide a transcript and identify speakers only "
            "when clearly distinguishable. "
            "For video, include spoken content plus important visible text "
            "and scene facts in chronological order. "
            "For images, transcribe visible text and describe only clearly "
            "observable content. "
            "Do not add outside knowledge, assumptions, advice, or inferred "
            "facts. Return plain text only."
        )

        response = client.models.generate_content(
            model=model,
            contents=[
                extraction_prompt,
                uploaded,
            ],
        )

        text = (
            getattr(
                response,
                "text",
                None,
            )
            or ""
        ).strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable content was returned "
                    "for this media file."
                ),
            )

        return text

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Media extraction failed: {exc}",
        ) from exc

    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        if uploaded is not None:
            try:
                client.files.delete(
                    name=uploaded.name
                )
            except Exception:
                pass


def _gemini_extract_inline(
    data: bytes,
    filename: str,
    mime_type: str,
    api_key: str,
    model: str,
) -> str:
    if not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "A Gemini API key is required to extract "
                "image, audio or video content."
            ),
        )

    try:
        from google import genai
        from google.genai import types

    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ),
        ) from exc

    extraction_prompt = (
        "Convert this uploaded source into a faithful textual evidence "
        "record for a retrieval-augmented generation system. "
        "Preserve all readable text, spoken words, labels, numbers, names, "
        "dates, and clearly observable facts. "
        "For audio, provide a transcript and identify speakers only when "
        "clearly distinguishable. "
        "For video, include spoken content plus important visible text and "
        "scene facts in chronological order. "
        "For images, transcribe visible text and describe only clearly "
        "observable content. "
        "Do not add outside knowledge, assumptions, advice, or inferred "
        "facts. Return plain text only."
    )

    try:
        client = genai.Client(
            api_key=api_key
        )

        media_part = types.Part.from_bytes(
            data=data,
            mime_type=mime_type,
        )

        response = client.models.generate_content(
            model=model,
            contents=[
                extraction_prompt,
                media_part,
            ],
        )

        text = (
            getattr(
                response,
                "text",
                None,
            )
            or ""
        ).strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable content was returned "
                    "for this media file."
                ),
            )

        return text

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Media extraction failed: {exc}",
        ) from exc


async def extract_upload(
    file: UploadFile,
    provider: str,
    api_key: str,
    model: str,
) -> dict:
    filename = file.filename or "upload"

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "File is too large. Maximum upload size "
                "is 100 MB for this build."
            ),
        )

    ext = Path(
        filename
    ).suffix.lower()

    mime_type = (
        file.content_type
        or mimetypes.guess_type(
            filename
        )[0]
        or "application/octet-stream"
    )

    if (
        ext in TEXT_EXTENSIONS
        or mime_type.startswith("text/")
    ):
        text = _decode_text(
            data
        )

        extraction = "local-text"

    elif (
        ext == ".pdf"
        or mime_type == "application/pdf"
    ):
        try:
            text = _extract_pdf(
                data
            )

            extraction = "local-pdf"

        except HTTPException:
            if provider != "gemini":
                raise

            text = _gemini_extract(
                data,
                filename,
                "application/pdf",
                api_key,
                model,
            )

            extraction = "gemini-pdf"

    elif mime_type.startswith(
        (
            IMAGE_PREFIX,
            AUDIO_PREFIX,
            VIDEO_PREFIX,
        )
    ):
        if provider != "gemini":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Image, audio and video extraction "
                    "currently requires Google Gemini "
                    "as the selected provider."
                ),
            )

        # Small multimodal files are sent directly to Gemini.
        # This avoids depending on the Gemini Files API.
        if len(data) <= INLINE_GEMINI_LIMIT:
            text = _gemini_extract_inline(
                data,
                filename,
                mime_type,
                api_key,
                model,
            )

            extraction = "gemini-inline"

        else:
            # Keep the original Files API behavior for larger files.
            text = _gemini_extract(
                data,
                filename,
                mime_type,
                api_key,
                model,
            )

            extraction = "gemini-multimodal"

    else:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. Use text, PDF, "
                "image, audio or video files."
            ),
        )

    return {
        "filename": filename,
        "mime_type": mime_type,
        "size": len(data),
        "document_text": text,
        "characters": len(text),
        "extraction": extraction,
    }