import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config missing: {path}")
    return json.loads(path.read_text())


def get_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key} must be a dict")
    return value


def resolve(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def resolve_path(value: Any, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path
