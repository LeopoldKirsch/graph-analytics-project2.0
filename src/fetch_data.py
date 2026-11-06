"""Download the external data and save tidy, versioned extracts in data/.

1. Eurostat routes (the edges of the graph)
   Dataset avia_par_fr, "Air passenger transport routes between partner
   airports and main airports in France": annual passengers carried
   (unit PAS, tra_meas PAS_CRD), one row per (reporting French airport,
   partner airport). Every route of the reporting airports is published,
   with no minimum number of airlines.
   Bale-Mulhouse (LFSB) is a French airport reported by Switzerland: its rows
   are taken from avia_par_ch (reporting code CH_LFSB).
   -> data/raw/eurostat/eurostat_routes.csv
      year, reporting_country, reporting, partner_country, partner, passengers
   Dropped: pairs with an aggregate code that is not an airport (e.g. LF90,
   "other airports") and empty or confidential cells (Eurostat flag "c").

2. Airport coordinates (maps and route distances)
   OurAirports (public domain) for every airport in the Eurostat extract,
   plus airports closed since (e.g. Berlin-Tegel) placed by hand.
   -> data/external/airport_coords.csv

Licences: Eurostat data CC BY 4.0 (source: Eurostat); OurAirports public
domain; DGAC data (data/raw/apt, data/raw/pap) Licence Ouverte 2.0.

Usage:
    python src/fetch_data.py            # both extracts, 2010-2024
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUTES_FILE = ROOT / "data" / "raw" / "eurostat" / "eurostat_routes.csv"
COORDS_FILE = ROOT / "data" / "external" / "airport_coords.csv"
YEARS = range(2010, 2025)
API = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
       "?format=JSON&lang=EN&freq=A&unit=PAS&tra_meas=PAS_CRD")
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
PAIR = re.compile(r"([A-Z]{2})_([A-Z]{4})_([A-Z]{2})_([A-Z]{4})")

# Airports closed or renamed since the data year, missing from OurAirports' ICAO list.
CLOSED_AIRPORTS = [
    # icao, lat, lon, iata, name, country
    ("EDDT", 52.5597, 13.2877, "TXL", "Berlin-Tegel (closed 2020)", "DE"),
    ("EDDI", 52.4730, 13.4039, "THF", "Berlin-Tempelhof (closed 2008)", "DE"),
    ("LTBA", 40.9769, 28.8146, "ISL", "Istanbul Ataturk (passenger flights until 2019)", "TR"),
    ("HLLT", 32.6635, 13.1590, "TIP", "Tripoli International (closed 2014)", "LY"),
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def _tidy(d: dict, keep) -> tuple[list[dict], dict]:
    pairs = {i: p for p, i in d["dimension"]["airp_pr"]["category"]["index"].items()}
    times = {i: t for t, i in d["dimension"]["time"]["category"]["index"].items()}
    nt = d["size"][-1]
    rows, dropped = [], {"aggregate_codes": 0, "confidential_cells": 0}
    dropped["confidential_cells"] = sum(1 for k, s in d.get("status", {}).items()
                                        if "c" in s.lower() and keep(pairs[int(k) // nt]))
    for k, v in d["value"].items():
        code = pairs[int(k) // nt]
        if not keep(code):
            continue
        m = PAIR.fullmatch(code)
        if not m:
            dropped["aggregate_codes"] += 1
            continue
        if v:
            rows.append({"year": int(times[int(k) % nt]), "reporting_country": m[1],
                         "reporting": m[2], "partner_country": m[3], "partner": m[4],
                         "passengers": int(v)})
    return rows, dropped


def fetch_eurostat(years=YEARS, path: Path = ROUTES_FILE) -> pd.DataFrame:
    """French airports (avia_par_fr) + Bale-Mulhouse (avia_par_ch, CH_LFSB)."""
    time = "".join(f"&time={y}" for y in years)
    fr = json.loads(_get(API.format(dataset="avia_par_fr") + time))
    ch = json.loads(_get(API.format(dataset="avia_par_ch") + time))
    rows_fr, drop_fr = _tidy(fr, lambda c: True)
    rows_ch, drop_ch = _tidy(ch, lambda c: c.startswith("CH_LFSB_"))
    df = (pd.DataFrame(rows_fr + rows_ch)
          .sort_values(["year", "reporting", "partner"], ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path.relative_to(ROOT)}; dropped: "
          f"France {drop_fr}, Bale-Mulhouse {drop_ch}")
    return df


def fetch_coords(routes: pd.DataFrame | None = None, path: Path = COORDS_FILE) -> pd.DataFrame:
    """Coordinates of every airport in the Eurostat extract."""
    routes = pd.read_csv(ROUTES_FILE) if routes is None else routes
    codes = set(routes["reporting"]) | set(routes["partner"])
    oa = pd.read_csv(io.BytesIO(_get(OURAIRPORTS)), low_memory=False,
                     usecols=["ident", "type", "name", "latitude_deg", "longitude_deg",
                              "iso_country", "icao_code", "iata_code"])
    oa["icao"] = oa["icao_code"].fillna(oa["ident"])
    oa = oa[oa["icao"].isin(codes)].drop_duplicates("icao")
    out = pd.DataFrame({"icao": oa["icao"], "lat": oa["latitude_deg"], "lon": oa["longitude_deg"],
                        "iata": oa["iata_code"], "name": oa["name"], "country": oa["iso_country"],
                        "type": oa["type"], "source": "OurAirports"})
    manual = pd.DataFrame(CLOSED_AIRPORTS, columns=["icao", "lat", "lon", "iata", "name", "country"])
    manual = manual[manual["icao"].isin(codes) & ~manual["icao"].isin(out["icao"])]
    out = pd.concat([out, manual.assign(type="closed", source="placed by hand")])
    out = out.sort_values("icao", ignore_index=True)
    missing = sorted(codes - set(out["icao"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} airports to {path.relative_to(ROOT)}; without coordinates: {missing}")
    return out


if __name__ == "__main__":
    fetch_coords(fetch_eurostat())
