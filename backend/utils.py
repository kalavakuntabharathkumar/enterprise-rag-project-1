from langchain_community.document_loaders import PyPDFLoader
from backend.logger import app_logger
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        app_logger.info(f"Extracted text from {file_path}, length: {len(text)}")
        return text
    except Exception as e:
        app_logger.error(f"Error extracting text from PDF: {e}")
        raise

def save_uploaded_file(uploaded_file, save_path: str) -> str:
    """Save uploaded file to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    app_logger.info(f"Saved file to {save_path}")
    return save_path