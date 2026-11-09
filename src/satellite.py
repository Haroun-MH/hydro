"""Squelette réservé à la future collecte Copernicus."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from src.config import load_env_file
except ModuleNotFoundError:
    from config import load_env_file


def project_root() -> Path:
    """Retourne la racine du projet."""
    return Path(__file__).resolve().parents[1]


def read_copernicus_credentials() -> tuple[str | None, str | None]:
    """Lit les identifiants Copernicus depuis .env, sans lancer de collecte."""
    load_env_file(project_root() / ".env")
    return os.getenv("CDSE_CLIENT_ID"), os.getenv("CDSE_CLIENT_SECRET")


def main() -> None:
    """Point d’entrée réservé à une future implémentation."""
    read_copernicus_credentials()
    raise NotImplementedError(
        "TODO: implémenter la recherche, le téléchargement et la préparation Sentinel-1/2."
    )


if __name__ == "__main__":
    main()
