def _get_path(path: str, *args) -> Path:
    return Path(path.format(*args))