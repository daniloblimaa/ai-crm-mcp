import logging
import numpy as np
from functools import lru_cache
from typing import Union

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        return model
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


def embed_text(text: str) -> np.ndarray:
    model = _get_model()
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return embeddings.astype(np.float32)
