"""Contrôles qualité minimaux pour les sorties du pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def validate_timeseries(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Vérifie le schéma, la fréquence et les valeurs manquantes."""
    required = {
        "station_code",
        "datetime_utc",
        "H_m",
        "Q_m3_s",
        "H_m_missing",
        "Q_m3_s_missing",
    }
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")
    timestamps = pd.to_datetime(data["datetime_utc"], utc=True)
    if timestamps.duplicated().any():
        raise ValueError("Des timestamps dupliqués sont présents.")
    if len(timestamps) > 1:
        expected = pd.Timedelta(config["resample_frequency"])
        if not (timestamps.diff().dropna() == expected).all():
            raise ValueError("La fréquence de la série n'est pas régulière.")
    if (data[["H_m", "Q_m3_s"]].dropna() < 0).any().any():
        raise ValueError("Une valeur hydrométrique négative est présente.")
    return {
        "rows": int(len(data)),
        "missing_rate": {
            column: float(data[column].mean())
            for column in ("H_m_missing", "Q_m3_s_missing")
        },
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
    }


def validate_images(image_dir: Path, size: tuple[int, int]) -> dict[str, Any]:
    """Vérifie les dimensions et le nombre d'images préparées."""
    files = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    for path in files:
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Dimensions invalides pour {path}: {image.size}")
    return {"images": len(files), "size": list(size)}


def write_quality_report(report: dict[str, Any], output: Path) -> None:
    """Écrit les statistiques de qualité en JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
