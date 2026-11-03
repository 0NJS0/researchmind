"""Safe file-system names and paths under PAPERS_DIR (shared by the API and the autopilot)."""
import re
from pathlib import Path

from src.config import PAPERS_DIR

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


def papers_root() -> Path:
    return Path(PAPERS_DIR).resolve()


def is_safe_name(value: str) -> bool:
    return bool(NAME_RE.match(value or "")) and ".." not in value


def to_name(raw: str, max_len: int = 120) -> str:
    """Turn a label, title or file stem into a safe single path component."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (raw or "").strip()).strip("._-")
    return re.sub(r"_+", "_", name)[:max_len].strip("._-")


def inside_root(*parts: str) -> Path | None:
    """Resolve a path under PAPERS_DIR; None if it would escape it."""
    root = papers_root()
    path = root.joinpath(*parts).resolve()
    return path if path == root or root in path.parents else None


def analysis_path(topic: str, paper_id: str) -> Path:
    return papers_root() / topic / paper_id / "_reports" / f"{paper_id}_analysis.md"


def living_report_path(topic: str) -> Path:
    return papers_root() / topic / "_reports" / "living_report.md"
