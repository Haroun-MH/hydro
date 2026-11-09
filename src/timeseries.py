"""Collecte Hub'Eau et préparation des séries hydrométriques."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from src.config import load_env_file
except ModuleNotFoundError:
    from config import load_env_file

BASE_URL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
PAGE_SIZE = 20_000
MAX_PAGES = 10
RESAMPLE_FREQUENCY = "10min"
INTERPOLATION_LIMIT = 3


def project_root() -> Path:
    """Retourne la racine du projet, quel que soit le répertoire courant."""
    return Path(__file__).resolve().parents[1]


def make_session() -> requests.Session:
    """Construit une session HTTP avec quelques reprises sur erreurs transitoires."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_recent_observations(
    session: requests.Session,
    station_code: str,
    days: int,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> pd.DataFrame:
    """Télécharge les observations récentes avec pagination."""
    start = datetime.now(timezone.utc) - timedelta(days=days)
    url = f"{BASE_URL}/observations_tr"
    params: dict[str, str | int] | None = {
        "code_entite": station_code,
        "date_debut_obs": start.isoformat().replace("+00:00", "Z"),
        "size": page_size,
    }
    rows: list[dict] = []

    for _ in range(max_pages):
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data", []))
        url = payload.get("next")
        params = None
        if not url:
            break
    else:
        raise RuntimeError("La limite de pagination est atteinte; réduisez DAYS.")

    if not rows:
        raise ValueError("Aucune observation Hub'Eau pour cette période.")
    return pd.DataFrame(rows)


def prepare_timeseries(
    raw: pd.DataFrame,
    station_code: str,
    frequency: str = RESAMPLE_FREQUENCY,
    interpolation_limit: int = INTERPOLATION_LIMIT,
) -> pd.DataFrame:
    """Nettoie, convertit et régularise les colonnes H et Q."""
    required = {"code_station", "date_obs", "grandeur_hydro", "resultat_obs"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Colonnes Hub'Eau absentes: {sorted(missing)}")

    observations = raw.loc[
        raw["grandeur_hydro"].isin(["H", "Q"]),
        ["code_station", "date_obs", "grandeur_hydro", "resultat_obs"],
    ].copy()
    observations["date_obs"] = pd.to_datetime(
        observations["date_obs"], utc=True, errors="coerce"
    )
    observations["resultat_obs"] = pd.to_numeric(
        observations["resultat_obs"], errors="coerce"
    )
    observations = observations.dropna(subset=["date_obs", "resultat_obs"])
    observations = observations.drop_duplicates(
        ["code_station", "date_obs", "grandeur_hydro"], keep="last"
    )

    pivot = observations.pivot(
        index="date_obs", columns="grandeur_hydro", values="resultat_obs"
    ).sort_index()
    if pivot.empty:
        raise ValueError("Aucune valeur H ou Q exploitable.")

    regular = pivot.rename(columns={"H": "H_m", "Q": "Q_m3_s"})
    regular = regular.resample(frequency).asfreq()
    for column in ("H_m", "Q_m3_s"):
        if column not in regular:
            regular[column] = pd.NA
    regular = regular[["H_m", "Q_m3_s"]].astype("float64")
    regular["H_m"] = regular["H_m"] / 1000
    regular["Q_m3_s"] = regular["Q_m3_s"] / 1000
    missing_columns = regular.isna().add_suffix("_missing").astype(bool)
    regular = regular.interpolate(
        method="time",
        limit=interpolation_limit,
        limit_area="inside",
    )
    result = pd.concat([regular, missing_columns], axis=1).reset_index()
    result = result.rename(columns={"date_obs": "datetime_utc"})
    result.insert(0, "station_code", station_code)
    return result


def main() -> None:
    load_env_file(project_root() / ".env")
    station_code = os.getenv("HYDRO_STATION_CODE")
    if not station_code:
        raise RuntimeError("HYDRO_STATION_CODE doit être défini dans .env ou l'environnement.")
    days = int(os.getenv("HYDRO_DAYS", "30"))
    session = make_session()
    raw = fetch_recent_observations(session, station_code, days)
    clean = prepare_timeseries(raw, station_code)
    output = project_root() / "data" / "processed" / "timeseries_clean.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"{len(raw)} observations téléchargées; fichier écrit: {output}")


if __name__ == "__main__":
    main()
