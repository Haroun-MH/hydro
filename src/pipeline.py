"""Orchestrateur unique du pipeline DataOps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.config import load_env_file
from src.timeseries import fetch_recent_observations, make_session, prepare_timeseries
from src.validation import validate_timeseries, write_quality_report


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    with (project_root() / "configs" / "params.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_timeseries(config: dict[str, Any]) -> dict[str, Any]:
    session = make_session()
    raw = fetch_recent_observations(
        session,
        config["station_code"],
        config["days"],
        page_size=config["page_size"],
        max_pages=config["max_pages"],
    )
    clean = prepare_timeseries(
        raw,
        config["station_code"],
        frequency=config["resample_frequency"],
        interpolation_limit=config["interpolation_limit"],
    )
    output = project_root() / config["paths"]["timeseries_clean"]
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return validate_timeseries(clean, config)


def main() -> None:
    root = project_root()
    config = load_config()
    load_env_file(root / ".env")
    config["station_code"] = __import__("os").getenv(
        "HYDRO_STATION_CODE", config["station_code"]
    )
    report = {"timeseries": run_timeseries(config)}
    write_quality_report(report, root / config["paths"]["quality_report"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
