
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import database
database.DB_PATH = __import__("pathlib").Path(":memory:")


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    database.init_db()
    yield


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path, monkeypatch):
    import vector_store as vs
    import server
    import faiss
    from threading import Lock
    from embeddings import EMBEDDING_DIM

    monkeypatch.setattr(vs, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(vs, "IDS_PATH", tmp_path / "ids.npy")

    fresh = vs.VectorStore.__new__(vs.VectorStore)
    fresh._lock = Lock()
    fresh._index = faiss.IndexFlatIP(EMBEDDING_DIM)
    fresh._user_ids = []

    vs._store = fresh
    server.store = fresh

    yield

    vs._store = None
    server.store = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

from server import create_user, search_users, get_user, list_users


def _make_user(name="Alice", email="alice@example.com", description="Python developer"):
    return create_user(name=name, email=email, description=description)


# ─── create_user ─────────────────────────────────────────────────────────────

class TestCreateUser:
    def test_returns_id(self):
        result = _make_user()
        assert "id" in result
        assert isinstance(result["id"], int)
        assert result["id"] >= 1

    def test_increments_id(self):
        id1 = _make_user()["id"]
        id2 = create_user("Bob", "bob@example.com", "Data scientist")["id"]
        assert id2 == id1 + 1

    def test_duplicate_email_returns_error(self):
        _make_user()
        result = _make_user()           # same email
        assert "error" in result
        assert "already registered" in result["error"]

    def test_invalid_email_returns_error(self):
        result = create_user("X", "not-an-email", "desc")
        assert "error" in result

    def test_short_name_returns_error(self):
        result = create_user("A", "a@example.com", "desc")
        assert "error" in result

    def test_email_normalised_to_lowercase(self):
        result = create_user("Eve", "Eve@Example.COM", "ML engineer")
        assert "id" in result
        user = get_user(result["id"])
        assert user["email"] == "eve@example.com"


# ─── get_user ─────────────────────────────────────────────────────────────────

class TestGetUser:
    def test_fetch_existing(self):
        uid = _make_user()["id"]
        user = get_user(uid)
        assert user["id"] == uid
        assert user["name"] == "Alice"
        assert user["email"] == "alice@example.com"

    def test_fetch_nonexistent(self):
        result = get_user(9999)
        assert "error" in result
        assert "not found" in result["error"]

    def test_invalid_id(self):
        result = get_user(0)
        assert "error" in result


# ─── search_users ────────────────────────────────────────────────────────────

class TestSearchUsers:
    def test_empty_store_returns_empty(self):
        result = search_users("python developer")
        assert result == []

    def test_returns_results(self):
        _make_user("Alice", "alice@example.com", "Senior Python backend developer")
        create_user("Bob", "bob@example.com", "Data scientist specialising in NLP")
        results = search_users("backend engineer Python", top_k=2)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all("score" in r for r in results)

    def test_results_have_required_fields(self):
        _make_user()
        results = search_users("python", top_k=1)
        assert len(results) == 1
        keys = {"id", "name", "email", "description", "score"}
        assert keys.issubset(results[0].keys())

    def test_top_k_limits_results(self):
        for i in range(5):
            create_user(f"User{i}", f"user{i}@example.com", f"Engineer number {i}")
        results = search_users("engineer", top_k=3)
        assert len(results) <= 3

    def test_semantic_ranking(self):
        create_user("Alice", "alice@example.com", "machine learning engineer deep learning neural networks AI")
        create_user("Bob",   "bob@example.com",   "plumber pipe installation water heating boiler")
        results = search_users("deep learning artificial intelligence", top_k=2)
        assert len(results) == 2
        alice = next(r for r in results if r["name"] == "Alice")
        bob   = next(r for r in results if r["name"] == "Bob")
        assert alice["score"] > bob["score"], (
            f"Expected Alice (score={alice['score']:.4f}) > Bob (score={bob['score']:.4f})"
        )

    def test_invalid_top_k(self):
        result = search_users("query", top_k=0)
        assert "error" in result


# ─── list_users ──────────────────────────────────────────────────────────────

class TestListUsers:
    def test_empty(self):
        assert list_users() == []

    def test_returns_all(self):
        _make_user("Alice", "alice@example.com", "desc 1")
        create_user("Bob", "bob@example.com", "desc 2")
        users = list_users()
        assert len(users) == 2
        names = {u["name"] for u in users}
        assert names == {"Alice", "Bob"}