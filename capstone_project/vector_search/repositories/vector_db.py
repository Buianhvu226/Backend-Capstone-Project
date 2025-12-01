from chromadb.utils import embedding_functions
import chromadb
from .._constants import PRIMARY_GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME, CHROMA_PERSIST_PATH

def initialize_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=PRIMARY_GOOGLE_API_KEY, model_name=EMBEDDING_MODEL_NAME)
    try:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=google_ef, metadata={"hnsw:space": "cosine"})
        collection.count()
        return collection
    except Exception:
        raise

