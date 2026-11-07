"""Build the French air traffic network from Eurostat route data.

Edges: Eurostat avia_par_fr (+ avia_par_ch for Bale-Mulhouse), annual
passengers carried between each reporting French airport and its partner
airports (data/raw/eurostat/eurostat_routes.csv, see fetch_data.py).
Node attributes of French airports: DGAC airport series ASP_APT
(data/raw/apt), Licence Ouverte 2.0.

Graph model (one graph per year):
- Nodes: airports (ICAO codes), French (metropolitan and overseas) and foreign.
- Edges: one undirected edge per airport pair with passengers in the year.
  Bale-Mulhouse has a French and a Swiss sector, declared separately (French
  and Swiss data): the two declarations of a route are complementary and are
  added. Then, a route between two reporting airports (e.g. a domestic route)
  is reported by both of them: the two figures are averaged (they agree within
  a fraction of a percent). Otherwise the single report is used.
- Edge attributes: passengers, cost (= 1/passengers, for shortest paths),
  distance_km (great-circle, from data/external/airport_coords.csv),
  reports (1 or 2).
- Node attributes: name, french (bool), zone ("metro" / "overseas" /
  "foreign"), icao_region, and for French airports the annual traffic of the
  airport from DGAC (apt_passengers, apt_movements...).

The graph is undirected (Eurostat route figures combine both directions) and
simple. There are no foreign-foreign routes: only routes touching a French
airport are in the data.

Coverage: only airports that report to Eurostat (about 40 French airports,
over 95 % of French traffic) have all their routes; routes between two small
non-reporting airports are missing, as are routes inside French Polynesia and
New Caledonia (outside the EU statistical system). Annual data end in 2024.

Usage:
    python src/build_graph.py            # summary of the 2024 graph
    python src/build_graph.py 2019 2020  # several years

    import sys; sys.path.append("../src")
    from build_graph import build_graph
    G = build_graph(2024)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ROUTES_FILE = DATA_DIR / "eurostat" / "eurostat_routes.csv"
COORDS_FILE = ROOT / "data" / "external" / "airport_coords.csv"
YEARS = range(2010, 2025)
MAIN_YEAR = 2024

# APT rows that are national totals, not airports.
APT_AGGREGATES = {"ZZMT", "ZZOM"}

# First letter of an ICAO code -> broad world region (ICAO location indicator
# prefixes). Coarse on purpose: it is only used to help read communities.
ICAO_REGIONS = {
    "A": "South Pacific (west)",
    "B": "Iceland / Greenland",
    "C": "Canada",
    "D": "West & North Africa",
    "E": "Northern Europe",
    "F": "Southern & Indian Ocean Africa",
    "G": "West Africa & Morocco",
    "H": "East Africa & Egypt",
    "K": "United States",
    "L": "Southern Europe & Mediterranean",
    "M": "Central America & Mexico",
    "N": "South Pacific",
    "O": "Middle East",
    "P": "North Pacific",
    "R": "East Asia",
    "S": "South America",
    "T": "Caribbean",
    "U": "Russia & Central Asia",
    "V": "South & Southeast Asia",
    "W": "Maritime Southeast Asia",
    "Y": "Australia",
    "Z": "China",
}


def _read_dgac(path: Path) -> pd.DataFrame:
    # DGAC files: semicolon-separated, decimal comma, UTF-8 with BOM.
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")


def load_apt(year: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly French-airport rows for one year (DGAC), aggregates removed."""
    apt = _read_dgac(data_dir / "apt" / f"ASP_APT_{year}.csv")
    return apt[~apt["APT"].isin(APT_AGGREGATES)]


def node_table(year: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Annual traffic of French airports from the DGAC APT series, indexed by ICAO."""
    apt = load_apt(year, data_dir)
    return apt.groupby("APT").agg(
        apt_name=("APT_NOM", "first"),
        apt_zone=("APT_ZON", "first"),
        apt_pax_dep=("APT_PAX_dep", "sum"),
        apt_pax_arr=("APT_PAX_arr", "sum"),
        apt_pax_transit=("APT_PAX_tr", "sum"),
        apt_movements=("APT_NMVT_mxt", "sum"),
    ).assign(
        apt_passengers=lambda d: d["apt_pax_dep"] + d["apt_pax_arr"] + d["apt_pax_transit"]
    )


def load_routes(year: int, path: Path = ROUTES_FILE) -> pd.DataFrame:
    """Eurostat rows (reporting airport, partner airport) for one year."""
    r = pd.read_csv(path)
    return r[r["year"] == year].reset_index(drop=True)


def load_coords(path: Path = COORDS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False).set_index("icao")


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def edge_table(year: int) -> pd.DataFrame:
    """One row per undirected airport pair (u < v).

    1. Sum the declarations of the same airport across reporting countries
       (Bale-Mulhouse's French and Swiss sectors are complementary).
    2. Average the reports of the two ends of a route when both report.
    """
    r = load_routes(year)
    r = (r.groupby(["reporting", "partner"], as_index=False)["passengers"].sum())
    r = r.assign(u=r[["reporting", "partner"]].min(axis=1),
                 v=r[["reporting", "partner"]].max(axis=1))
    e = (r.groupby(["u", "v"])
         .agg(passengers=("passengers", "mean"), reports=("passengers", "count"))
         .reset_index())
    e["passengers"] = e["passengers"].round().astype(int)
    e = e[e["passengers"] > 0].reset_index(drop=True)
    coords = load_coords()
    e["distance_km"] = [
        round(great_circle_km(coords.at[u, "lat"], coords.at[u, "lon"],
                              coords.at[v, "lat"], coords.at[v, "lon"]))
        for u, v in zip(e["u"], e["v"])
    ]
    return e


def _french_codes(year: int, apt: pd.DataFrame) -> set[str]:
    r = load_routes(year)
    reported = set(r.loc[r["reporting_country"] == "FR", "reporting"]) | {"LFSB"}
    partners = set(r.loc[r["partner_country"] == "FR", "partner"])
    return set(apt.index) | reported | partners


def build_graph(year: int = MAIN_YEAR, data_dir: Path = DATA_DIR) -> nx.Graph:
    """Undirected, simple, weighted graph of the year's routes.

    Edge weights: use ``weight="passengers"`` for strength-type measures
    (weighted degree, Louvain) and ``weight="cost"`` (= 1/passengers) for
    shortest-path measures, because NetworkX treats weights as distances.
    """
    edges = edge_table(year)
    apt = node_table(year, data_dir)
    coords = load_coords()
    french = _french_codes(year, apt)

    G = nx.Graph(year=year, source="Eurostat avia_par_fr / avia_par_ch, DGAC ASP_APT",
                 weight="passengers")
    for row in edges.itertuples(index=False):
        G.add_edge(row.u, row.v, passengers=int(row.passengers),
                   cost=1.0 / row.passengers, distance_km=int(row.distance_km),
                   reports=int(row.reports))

    for code, attrs in G.nodes(data=True):
        is_french = code in french
        attrs["french"] = is_french
        attrs["icao_region"] = "France" if is_french else ICAO_REGIONS.get(code[0], "Other")
        if is_french and code in apt.index:
            a = apt.loc[code]
            attrs["name"] = a["apt_name"]
            attrs["zone"] = "metro" if a["apt_zone"] == "MT" else "overseas"
            for col in ("apt_passengers", "apt_pax_dep", "apt_pax_arr",
                        "apt_pax_transit", "apt_movements"):
                attrs[col] = int(a[col])
        else:
            attrs["name"] = coords.at[code, "name"] if code in coords.index else code
            attrs["zone"] = ("foreign" if not is_french
                             else "metro" if code.startswith("LF") else "overseas")
    return G


def build_graphs(years=(2019, 2020, 2024), data_dir: Path = DATA_DIR) -> dict[int, nx.Graph]:
    """Graphs for several years, keyed by year (for the comparison over time)."""
    return {y: build_graph(y, data_dir) for y in years}


def summary(G: nx.Graph) -> dict:
    """Basic facts the report must state: size, type, connectivity."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    french = [n for n, f in G.nodes(data="french") if f]
    return {
        "year": G.graph.get("year"),
        "nodes": G.number_of_nodes(),
        "french_nodes": len(french),
        "foreign_nodes": G.number_of_nodes() - len(french),
        "edges": G.number_of_edges(),
        "directed": G.is_directed(),
        "multigraph": G.is_multigraph(),
        "connected": nx.is_connected(G),
        "components": len(comps),
        "component_sizes": [len(c) for c in comps],
        "total_passengers": sum(d for _, _, d in G.edges(data="passengers")),
    }


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [MAIN_YEAR]
    for y in years:
        print(summary(build_graph(y)))
