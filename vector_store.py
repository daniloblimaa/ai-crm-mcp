import logging
import numpy as np
import faiss
from pathlib import Path
from threading import Lock
from embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)

INDEX_PATH = Path("faiss_index/index.faiss")
IDS_PATH = Path("faiss_index/ids.npy")


class VectorStore:

    def __init__(self) -> None:
        self._lock = Lock()
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        # parallel list: position i in FAISS → user_ids[i]
        self._user_ids: list[int] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if INDEX_PATH.exists() and IDS_PATH.exists():
            try:
                self._index = faiss.read_index(str(INDEX_PATH))
                self._user_ids = np.load(str(IDS_PATH)).tolist()
                logger.info(
                    "Loaded FAISS index with %d vectors", self._index.ntotal
                )
            except Exception as exc:
                logger.warning("Could not load existing index: %s – starting fresh", exc)
                self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._user_ids = []
        else:
            logger.info("No existing FAISS index found – starting fresh")

    def _save(self) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(INDEX_PATH))
        np.save(str(IDS_PATH), np.array(self._user_ids, dtype=np.int64))
        logger.debug("FAISS index persisted (%d vectors)", self._index.ntotal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, user_id: int, embedding: np.ndarray) -> None:
        vector = embedding.reshape(1, -1).astype(np.float32)
        with self._lock:
            self._index.add(vector)
            self._user_ids.append(user_id)
            self._save()
        logger.debug("Added vector for user_id=%d (total=%d)", user_id, self._index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        vector = query_embedding.reshape(1, -1).astype(np.float32)

        with self._lock:
            scores, indices = self._index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for "not found"
                continue
            user_id = self._user_ids[idx]
            results.append((user_id, float(score)))

        return results

    def size(self) -> int:
        return self._index.ntotal


# Module-level singleton
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
