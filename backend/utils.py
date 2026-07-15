import asyncio
import os

from langchain_community.document_loaders import PyPDFLoader
from backend.logger import app_logger


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file synchronously.

    Called via asyncio.to_thread from the async upload route so it does
    not block the event loop.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        app_logger.info(f"Extracted text from {file_path}, length: {len(text)}")
        return text
    except Exception as e:
        app_logger.error(f"Error extracting text from PDF: {e}")
        raise


async def save_uploaded_file(uploaded_file, save_path: str) -> str:
    """Read an UploadFile asynchronously, then write it to disk in a
    thread so neither operation blocks the event loop.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    contents = await uploaded_file.read()

    def _write(path: str, data: bytes) -> None:
        with open(path, "wb") as fh:
            fh.write(data)

    await asyncio.to_thread(_write, save_path, contents)
    app_logger.info(f"Saved file to {save_path}")
    return save_path
