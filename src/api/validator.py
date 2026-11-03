"""Topic Validator API (/api/validator): stress-test a research idea against the literature."""
import asyncio
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.library import safe_name
from src.research import validator

router = APIRouter(prefix="/api/validator", tags=["validator"])
run = asyncio.to_thread
ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ValidateRequest(BaseModel):
    idea: str = Field(min_length=10, max_length=2000)
    project_type: str = "empirical_quantitative"
    topic: str = ""

    @field_validator("idea")
    @classmethod
    def _clean(cls, v: str) -> str:
        return " ".join(v.split())

    @field_validator("project_type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v not in validator.PROJECT_TYPES:
            raise ValueError(f"project_type must be one of: {', '.join(validator.PROJECT_TYPES)}")
        return v


def _public(v: dict) -> dict:
    return {**v, "running": validator.is_running(v["id"]) or v["status"] == "running"}


@router.get("/project-types")
async def project_types():
    return {"project_types": [{"key": k, "label": label} for k, label in validator.PROJECT_TYPES.items()]}


@router.get("")
async def list_validations():
    return {"validations": await run(validator.list_validations)}


@router.post("", status_code=202)
async def create(body: ValidateRequest):
    topic = safe_name(body.topic, "topic") if body.topic else ""
    v = await run(validator.create_validation, body.idea, body.project_type, topic)
    validator.start_validation(v["id"])
    return _public(v)


@router.get("/{validation_id}")
async def get(validation_id: str):
    if not ID_RE.match(validation_id):
        raise HTTPException(404, "Not found")
    v = await run(validator.get_validation, validation_id)
    if not v:
        raise HTTPException(404, "Validation not found")
    return _public(v)


@router.delete("/{validation_id}")
async def delete(validation_id: str):
    if not ID_RE.match(validation_id):
        raise HTTPException(404, "Not found")
    await run(validator.delete_validation, validation_id)
    return {"status": "deleted"}
