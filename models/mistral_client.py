# mistral_client.py
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from mistralai import Mistral


OCRModelName = Literal["mistral-ocr-latest", "mistral-ocr-2512"]


@dataclass(frozen=True)
class OCRConfig:
    model: OCRModelName = "mistral-ocr-2512"
    table_format: Literal["html", "markdown", None] = None
    extract_header: bool = False
    extract_footer: bool = False
    include_image_base64: bool = True


class MistralOCR:
    """
    Small wrapper around Mistral OCR:
    - Handling local paths (uploading -> file_id -> OCR)
    - Handling URLs (choosing image_url vs document_url)
    - Providing batch helpers (jsonl creation + batch job creation)

    Docs reference:
    - OCR processor usage and model name
    - Batch OCR cookbook (jsonl format + /v1/ocr endpoint for batch jobs)
    """

    def __init__(self, api_key: str | None = None, config: OCRConfig | None = None) -> None:
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing MISTRAL_API_KEY (pass api_key or set env var).")
        self._client = Mistral(api_key=api_key)
        self._config = config or OCRConfig()

    # ---------- Public: single document OCR ----------

    def ocr(self, src: str | Path, *, pages: list[int] | None = None) -> Any:
        """
        Running OCR on:
        - Local file path: uploading then using file_id
        - Remote URL: using document_url or image_url depending on extension / mime guess

        pages: only meaningful for multi-page docs (e.g. PDFs). API uses 0-based indices.
        """
        if isinstance(src, Path) or self._looks_like_local_path(src):
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(str(path))
            file_id = self._upload_for_ocr(path)
            return self._ocr_with_file_id(file_id=file_id, pages=pages)

        # Treating as URL
        url = str(src)
        return self._ocr_with_url(url=url, pages=pages)

    # ---------- Public: batching helpers ----------

    def write_batch_jsonl(
        self,
        inputs: Iterable[str | Path],
        output_path: str | Path,
        *,
        mode: Literal["upload_then_file_id", "data_url"] = "upload_then_file_id",
        include_image_base64: bool | None = None,
    ) -> Path:
        """
        Creating a JSONL batch input file suitable for Mistral Batch Inference.

        Each line is:
          {"custom_id": "...", "body": { ... OCR request body ... }}

        mode:
          - "upload_then_file_id": uploading each local file and referencing {"document": {"file_id": "..."}}
            - Works for PDFs and images.
            - Avoids large data URLs.
          - "data_url": encoding local files as data: URLs and sending as *url* inputs
            - Mirrors the Batch OCR cookbook for images: data:image/...;base64,...
            - For PDFs, using data:application/pdf;base64,... (commonly supported pattern).

        Note: for remote URLs, no uploading is needed; they are referenced directly.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        use_include_image_base64 = (
            self._config.include_image_base64 if include_image_base64 is None else include_image_base64
        )

        with out.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(inputs):
                body = self._make_ocr_body_for_batch_item(
                    item=item,
                    mode=mode,
                    include_image_base64=use_include_image_base64,
                )
                entry = {"custom_id": str(idx), "body": body}
                f.write(self._json_dumps(entry) + "\n")

        return out

    def create_batch_job(
        self,
        batch_jsonl_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        timeout_hours: int = 24,
    ) -> Any:
        """
        Uploading the batch JSONL then creating a batch job for /v1/ocr.
        """
        path = Path(batch_jsonl_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        batch_file = self._client.files.upload(
            file={"file_name": path.name, "content": path.open("rb")},
            purpose="batch",
        )

        return self._client.batch.jobs.create(
            input_files=[batch_file.id],
            model=self._config.model,
            endpoint="/v1/ocr",
            metadata=metadata or {},
            timeout_hours=timeout_hours,
        )

    # ---------- Internal helpers ----------

    @staticmethod
    def _looks_like_local_path(s: str) -> bool:
        # Checking for typical URL prefixes
        return not (s.startswith("http://") or s.startswith("https://") or s.startswith("data:"))

    def _upload_for_ocr(self, path: Path) -> str:
        # Uploading file with explicit purpose for OCR
        uploaded = self._client.files.upload(
            file={"file_name": path.name, "content": path.open("rb")},
            purpose="ocr",
        )
        return uploaded.id

    def _ocr_with_file_id(self, file_id: str, *, pages: list[int] | None) -> Any:
        req: dict[str, Any] = {
            "model": self._config.model,
            "document": {"file_id": file_id},
            "include_image_base64": self._config.include_image_base64,
        }

        # Adding optional OCR parameters
        if self._config.table_format is not None:
            req["table_format"] = self._config.table_format
        if self._config.extract_header:
            req["extract_header"] = True
        if self._config.extract_footer:
            req["extract_footer"] = True
        if pages is not None:
            req["pages"] = pages

        return self._client.ocr.process(**req)

    def _ocr_with_url(self, url: str, *, pages: list[int] | None) -> Any:
        doc = self._document_chunk_for_url(url)
        req: dict[str, Any] = {
            "model": self._config.model,
            "document": doc,
            "include_image_base64": self._config.include_image_base64,
        }

        if self._config.table_format is not None:
            req["table_format"] = self._config.table_format
        if self._config.extract_header:
            req["extract_header"] = True
        if self._config.extract_footer:
            req["extract_footer"] = True
        if pages is not None:
            req["pages"] = pages

        return self._client.ocr.process(**req)

    def _document_chunk_for_url(self, url: str) -> dict[str, Any]:
        # Choosing the correct "type" based on likely media type
        lowered = url.lower()
        if lowered.endswith(".pdf"):
            return {"type": "document_url", "document_url": url}
        if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".tif", ".tiff")):
            return {"type": "image_url", "image_url": url}

        # Falling back to guessing via mimetypes
        mime, _ = mimetypes.guess_type(url)
        if mime == "application/pdf":
            return {"type": "document_url", "document_url": url}
        if mime and mime.startswith("image/"):
            return {"type": "image_url", "image_url": url}

        # Defaulting to document_url (covers many doc types per docs)
        return {"type": "document_url", "document_url": url}

    def _make_ocr_body_for_batch_item(
        self,
        item: str | Path,
        *,
        mode: Literal["upload_then_file_id", "data_url"],
        include_image_base64: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "include_image_base64": include_image_base64,
        }

        if self._config.table_format is not None:
            body["table_format"] = self._config.table_format
        if self._config.extract_header:
            body["extract_header"] = True
        if self._config.extract_footer:
            body["extract_footer"] = True

        # Handling remote URLs
        if isinstance(item, str) and (item.startswith("http://") or item.startswith("https://") or item.startswith("data:")):
            body["document"] = self._document_chunk_for_url(item)
            return body

        # Handling local paths
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(str(path))

        if mode == "upload_then_file_id":
            file_id = self._upload_for_ocr(path)
            body["document"] = {"file_id": file_id}
            return body

        # mode == "data_url"
        data_url = self._path_to_data_url(path)
        if path.suffix.lower() == ".pdf":
            body["document"] = {"type": "document_url", "document_url": data_url}
        else:
            body["document"] = {"type": "image_url", "image_url": data_url}
        return body

    @staticmethod
    def _path_to_data_url(path: Path) -> str:
        # Encoding file content into a data: URL
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return f"data:application/pdf;base64,{b64}"

        mime, _ = mimetypes.guess_type(path.name)
        mime = mime or "application/octet-stream"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        # Importing locally to keep dependencies minimal
        import json

        return json.dumps(obj, ensure_ascii=False)


__all__ = ["MistralOCR", "OCRConfig"]
