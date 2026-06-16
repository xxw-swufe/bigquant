"""Small helpers for loading local API keys without committing them."""

from __future__ import annotations

import os
from pathlib import Path


def load_deepseek_api_key(project_root: str | Path | None = None) -> str | None:
    """Load DeepSeek API key from env or a local .env file.

    Priority:
    1. DEEPSEEK_API_KEY environment variable
    2. .env file under project_root/current working directory
    """
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key.strip()

    roots = []
    if project_root is not None:
        roots.append(Path(project_root))
    roots.extend([Path.cwd(), Path.cwd().parent])

    for root in roots:
        env_path = root / ".env"
        if not env_path.exists():
            continue
        key = _read_env_value(env_path, "DEEPSEEK_API_KEY")
        if key:
            os.environ["DEEPSEEK_API_KEY"] = key
            return key
    return None


def _read_env_value(env_path: Path, name: str) -> str | None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        return value.strip().strip('"').strip("'")
    return None
