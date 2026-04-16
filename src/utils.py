"""Shared utilities: config loader, logging, path helpers."""
import yaml
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(path=None):
    if path is None:
        path = PROJECT_ROOT / "config" / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
