from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re


class UserCreate(BaseModel):
    name: str
    email: str
    description: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email: {v}")
        return v.lower().strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Description must be at least 5 characters")
        return v


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    description: str


class UserSearchResult(BaseModel):
    id: int
    name: str
    email: str
    description: str
    score: float


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1")
        if v > 100:
            raise ValueError("top_k cannot exceed 100")
        return v