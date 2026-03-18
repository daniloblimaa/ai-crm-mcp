import logging
import sys
from typing import Any

from fastmcp import FastMCP

import database
import embeddings
import vector_store as vs
from models import SearchQuery, UserCreate

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------
database.init_db()
store = vs.get_vector_store()

# ------------------------------------------------------------------
# MCP Server
# ------------------------------------------------------------------
mcp = FastMCP(
    name="CRM MCP Server",
    instructions=(
        "You are a CRM assistant. Use the available tools to create users, "
        "search for similar users semantically, or fetch users by ID."
    ),
)


# ------------------------------------------------------------------
# Tool 1 — create_user
# ------------------------------------------------------------------
@mcp.tool
def create_user(name: str, email: str, description: str) -> dict[str, Any]:
    try:
        payload = UserCreate(name=name, email=email, description=description)
    except Exception as exc:
        logger.warning("Validation error in create_user: %s", exc)
        return {"error": f"Validation error: {exc}"}

    if database.email_exists(payload.email):
        return {"error": f"Email '{payload.email}' is already registered"}

    try:
        user_id = database.insert_user(
            name=payload.name,
            email=payload.email,
            description=payload.description,
        )
        embedding = embeddings.embed_text(payload.description)
        store.add(user_id, embedding)
        logger.info("create_user OK id=%d", user_id)
        return {"id": user_id}
    except Exception as exc:
        logger.exception("Unexpected error in create_user")
        return {"error": f"Internal error: {exc}"}


# ------------------------------------------------------------------
# Tool 2 — search_users
# ------------------------------------------------------------------
@mcp.tool
def search_users(query: str, top_k: int = 5) -> list[dict[str, Any]] | dict[str, Any]:
    try:
        params = SearchQuery(query=query, top_k=top_k)
    except Exception as exc:
        return {"error": f"Validation error: {exc}"}

    try:
        query_embedding = embeddings.embed_text(params.query)
        hits = store.search(query_embedding, params.top_k)

        if not hits:
            return []

        ids = [uid for uid, _ in hits]
        scores = {uid: score for uid, score in hits}
        users_map = database.get_users_by_ids(ids)

        results = []
        for uid in ids:  # preserve ranking order
            if uid not in users_map:
                continue
            user = users_map[uid]
            results.append(
                {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "description": user["description"],
                    "score": round(scores[uid], 6),
                }
            )

        logger.info("search_users query=%r top_k=%d → %d results", query, top_k, len(results))
        return results

    except Exception as exc:
        logger.exception("Unexpected error in search_users")
        return {"error": f"Internal error: {exc}"}


# ------------------------------------------------------------------
# Tool 3 — get_user
# ------------------------------------------------------------------
@mcp.tool
def get_user(user_id: int) -> dict[str, Any]:
    if user_id < 1:
        return {"error": "user_id must be a positive integer"}

    try:
        user = database.get_user_by_id(user_id)
        if user is None:
            return {"error": f"User with id={user_id} not found"}
        logger.info("get_user id=%d OK", user_id)
        return user
    except Exception as exc:
        logger.exception("Unexpected error in get_user")
        return {"error": f"Internal error: {exc}"}


# ------------------------------------------------------------------
# Bonus Tool — list_users
# ------------------------------------------------------------------
@mcp.tool
def list_users() -> list[dict[str, Any]]:
    try:
        return database.list_all_users()
    except Exception as exc:
        logger.exception("Unexpected error in list_users")
        return [{"error": f"Internal error: {exc}"}]


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting CRM MCP Server…")
    mcp.run()
