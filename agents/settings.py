from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from env, .env, or config.local.json."""

    value = os.environ.get(name)
    if value:
        return value

    env_value = _read_dotenv().get(name)
    if env_value:
        return env_value

    json_value = _read_json_settings().get(name)
    if isinstance(json_value, str) and json_value:
        return json_value

    return default


def get_deepseek_api_key() -> str | None:
    return get_setting("DEEPSEEK_API_KEY")


def get_deepseek_model(default: str = "deepseek-chat") -> str:
    return get_setting("DEEPSEEK_MODEL", default) or default


def has_deepseek_api_key() -> bool:
    return bool(get_deepseek_api_key())


def _read_dotenv() -> dict[str, str]:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_json_settings() -> dict[str, object]:
    for filename in ["config.local.json", "config.json"]:
        path = PROJECT_ROOT / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}
