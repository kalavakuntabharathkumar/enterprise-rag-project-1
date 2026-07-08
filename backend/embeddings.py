from langchain_openai import OpenAIEmbeddings
from backend.config import Config
from backend.logger import app_logger

def get_embeddings():
    """Get OpenAI embeddings instance."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model=Config.EMBEDDING_MODEL)

def generate_embeddings(texts: list):
    """Generate embeddings for a list of texts."""
    embeddings = get_embeddings()
    try:
        vectors = embeddings.embed_documents(texts)
        app_logger.info(f"Generated embeddings for {len(texts)} texts")
        return vectors
    except Exception as e:
        app_logger.error(f"Error generating embeddings: {e}")
        raise