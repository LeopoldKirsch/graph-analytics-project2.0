# French Air Traffic Network — Graph Analytics Project

**How centralised is France's air connectivity on Paris-CDG, and how fragile does that make the network?**

Graph analytics course project, Albert School (2026).
Authors: Léopold Kirsch, Léa Chapoton

## Overview

We build a graph of the air routes of French airports from official route statistics (Eurostat), then analyse it with NetworkX:

- **Centrality**: which airports hold the network together, and how far behind CDG are Orly, Nice and Lyon?
- **Resilience**: what happens to connectivity when CDG, or CDG and Orly, are removed?
- **Communities**: do airports form natural groups beyond "Paris vs the rest"?
- **Comparison over time**: how did the structure change between 2019 (pre-COVID), 2020 (COVID) and 2024?

## Data

| Source | Content | Use |
| --- | --- | --- |
| Eurostat [`avia_par_fr`](https://ec.europa.eu/eurostat/databrowser/view/avia_par_fr/default/table), [`avia_par_ch`](https://ec.europa.eu/eurostat/databrowser/view/avia_par_ch/default/table) | Annual passengers carried on each route of the main French airports, 2010-2024 (Bâle-Mulhouse's French and Swiss sectors are declared separately and added) | **Edges** (`data/raw/eurostat/eurostat_routes.csv`) |
| DGAC [`ASP_APT`](https://www.data.gouv.fr/datasets/trafic-aerien-commercial-mensuel-francais-par-paire-daeroports-par-sens-depuis-1990) | Monthly traffic per French airport, 2010-2024 | Node attributes: name, zone, real airport traffic (`data/raw/apt/`) |
| DGAC [`ASP_PAP`](https://www.data.gouv.fr/datasets/trafic-aerien-commercial-mensuel-francais-par-paire-daeroports-par-sens-depuis-1990) | Monthly traffic per published airport pair, 2019, 2020, 2024 | Only for the comparison of sources (`data/raw/pap/`) |

**Why Eurostat and not the DGAC airport-pair series.** DGAC only publishes a route-month served by at least 3 airlines with none above 85 % of the traffic. Single-airline routes (Air France's regional routes to CDG, most low-cost routes) are hidden: the published routes carry only 43 % of the traffic of French airports in 2024, against 89 % for Eurostat, and the missing routes are precisely those without competition, which biases hub and resilience measures. `src/source_comparison.py` documents the comparison.

**Limits of the Eurostat data.** Only airports that report to Eurostat (about 40 French airports, the vast majority of the traffic) have all their routes; routes between two small non-reporting airports are missing. Routes inside French Polynesia and New Caledonia are not covered (outside the EU statistical system). Annual data end in 2024. A few cells are confidential (31 values over 2010-2024).

External data:

| File | Source | Licence |
| --- | --- | --- |
| `data/raw/eurostat/eurostat_routes.csv` | Eurostat, `avia_par_fr` and `avia_par_ch` (rows of LFSB), extracted with `python src/fetch_data.py` | CC BY 4.0, source: Eurostat |
| `data/raw/apt/`, `data/raw/pap/` | DGAC, data.gouv.fr | Licence Ouverte 2.0 |
| `data/external/airport_coords.csv` | Airport positions from [OurAirports](https://ourairports.com/data/); closed airports placed by hand (column `source`) | Public domain |
| `data/external/ne_50m_countries.geojson` | Country outlines, [Natural Earth](https://www.naturalearthdata.com) 1:50m | Public domain |

## Graph model

- **Nodes**: airports (ICAO codes), French (metropolitan and overseas) and foreign
- **Edges**: one undirected edge per airport pair with passengers in the year; a route reported by both of its airports (e.g. a domestic route) takes the mean of the two reports
- **Edge weight**: annual passengers; `cost` = 1/passengers for shortest paths; great-circle `distance_km`
- **Node attributes**: name, French/foreign, zone (metro / overseas / foreign), ICAO world region, and the airport's real annual traffic from DGAC for French airports
- **Graph type**: undirected, simple, weighted. There are no foreign-foreign routes: only routes touching a French airport are in the data

Weight convention: use `weight="passengers"` for strength-type measures (weighted degree, Louvain) and `weight="cost"` for shortest-path measures (weighted betweenness, closeness), since NetworkX treats weights as distances. Main year: **2024** (latest Eurostat year).

## Repository structure

```
data/raw/eurostat/  Eurostat route extract, 2010-2024
data/raw/apt/       DGAC airport series, 2010-2024 (node attributes)
data/raw/pap/       DGAC airport-pair series, 2019 / 2020 / 2024 (source comparison only)
data/external/      airport coordinates and country outlines
src/fetch_data.py   downloads the Eurostat and OurAirports extracts
src/build_graph.py  builds one graph per year (python src/build_graph.py 2019 2020 2024)
src/                analysis modules, one per task
notebooks/          main analysis notebook
results/            figures and tables used in the report
requirements.txt    Python dependencies
```

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb      # the whole analysis, in defence order (~5 min)
```

Each module can also be run on its own and writes its tables to `results/`:

```bash
python src/build_graph.py 2019 2020 2024   # graph summaries
python src/core_metrics.py          # size, connectivity, diameter/radius, articulation points
python src/centrality.py            # degree, strength, betweenness, harmonic, PageRank
python src/views.py                 # international vs domestic network
python src/null_model.py            # degree-preserving null model (500 random graphs, ~2.5 min)
python src/resilience.py            # 2019, 2020, 2024 (~1 min)
python src/communities.py           # 2019, 2020, 2024
python src/comparison.py            # 2019 / 2020 / 2024
python src/source_comparison.py     # Eurostat vs DGAC published routes (~2 min)
python src/figures.py               # report figures (~1 min)
python src/network_map.py           # maps
python src/time_series.py           # exploratory: 2010-2024
```

The data extracts are versioned; `python src/fetch_data.py` downloads them again from Eurostat and OurAirports (Eurostat revises its tables, so a new download can differ slightly).

Requires Python 3.11+. The results in `results/` were produced in a clean virtual environment from `requirements.txt` with Python 3.12, NetworkX 3.7, pandas 3.1, SciPy and Matplotlib 3.11. Randomised steps use fixed seeds (Louvain `seed=42`, random removals seeds 0-99, null model seed 42) and graphs are built in a fixed order, so a rerun gives the same tables. Louvain's implementation can change between NetworkX versions: with another version, stability scores and small communities can differ slightly. Figures can differ cosmetically (fonts) from one machine to another.

## Main results (2024)

- **Centralised on CDG**: 179 of 836 routes, 43 % of passengers, betweenness 0.65 (Orly 0.14, Bâle-Mulhouse 0.12, Beauvais 0.11); with Orly, the centre of the network.
- **International, long-haul centralisation**: CDG carries 45 % of international passengers but 91 % of long-haul passengers, and 76 foreign destinations are reachable only through CDG.
- **Two Paris hubs domestically**: the domestic network is decentralised (betweenness centralisation 0.24 vs 0.65); CDG has the most domestic routes, Orly the most domestic passengers.
- **More than a degree effect**: in degree-preserving random graphs CDG's betweenness is 0.59 (real 0.65, z = 7.4); the network is more hub-and-spoke than chance.
- **Robust yet fragile**: removing CDG leaves 71 % of the airports in the main component and 57 % of the passengers on surviving routes (CDG + Orly: 65 % and 38 %), against 99.6 % and 99 % for one random removal.
- **Communities follow business models**: CDG's international group, Orly and the regional airports, a low-cost group (Bâle-Mulhouse, Beauvais), Nice-Bordeaux, the Indian Ocean (modularity 0.28).
- **Slowly less concentrated**: from 2010 to 2024 the Paris share of passengers fell from 70 % to 62 %; COVID lowered CDG's betweenness (0.70 in 2019, 0.63 in 2020, 0.65 in 2024). Traffic is back to its 2019 level (154.7 M vs 155.6 M).
- **Why Eurostat**: the DGAC published routes capture 43 % of the airports' traffic (Eurostat 89 %); with them, CDG's centrality looks like a pure degree effect and COVID seems to have reinforced CDG, two conclusions that do not hold on the complete data.

## Team workflow

**One branch per task, one pull request per branch, reviewed by the other teammate.**

1. Start from an up-to-date `main`:
   ```bash
   git switch main && git pull
   git switch <task-branch>        # or: git switch -c <type>/<name> for a new task
   git merge main                  # if the branch already exists, bring it up to date
   ```
2. Work in small commits with clear messages ("Add weighted betweenness", not "update").
3. Push and open a pull request into `main` on GitHub.
4. **Peer review**: the other teammate reviews before anything is merged. The reviewer:
   - runs the code from a clean checkout and checks it gives the same numbers;
   - checks the method matches what the report says (weights, parameters, `seed=42`, which component);
   - reads the interpretation and asks questions now, since the oral will ask them too.
5. The reviewer approves, the author merges and deletes the branch.

Rules:
- Nobody commits directly to `main`; `main` always runs end to end.
- Branch names: `feature/` for required analysis, `explore/` for the exploratory options, `fix/` and `docs/` otherwise.
- Load data only through `src/build_graph.py`; do not copy loading code into other files.
- Each task puts its code in its own module `src/<task>.py`. The notebook only calls these functions, which avoids merge conflicts in `.ipynb` files. Only the owner of `feature/analysis-notebook` edits `notebooks/analysis.ipynb`.
- Figures and tables for the report go in `results/`, named after the task.

### Tasks and branches

| Branch | Owner | Deliverable |
| --- | --- | --- |
| `feature/data` | Léopold | `src/fetch_data.py`, `src/build_graph.py`: Eurostat extract and graph builder |
| `feature/core-metrics` | Léopold | `src/core_metrics.py`: size and type, connectivity, diameter and radius, articulation points |
| `feature/centrality` | Léopold | `src/centrality.py`: degree, strength, betweenness, harmonic, PageRank; CDG vs Orly / Nice / Lyon |
| `feature/network-views` | Léopold | `src/views.py`: international vs domestic network |
| `feature/null-model` | Léopold | `src/null_model.py`: degree-preserving null model |
| `feature/resilience` | Léa | `src/resilience.py`: CDG, Orly, CDG + Orly removal vs random; static and adaptive attacks |
| `feature/communities` | Léa | `src/communities.py`: Louvain weighted vs unweighted (`seed=42`), modularity, stability |
| `feature/year-comparison` | Léa | `src/comparison.py`: 2019 / 2020 / 2024 |
| `feature/source-comparison` | Léopold | `src/source_comparison.py`: Eurostat vs DGAC published routes |
| `feature/network-map` | Léopold | `src/network_map.py`: world map and map of French airports |
| `feature/figures` | Léa | `src/figures.py`: report figures |
| `explore/time-series` | Léa | `src/time_series.py`: 2010-2024 |
| `feature/analysis-notebook` | Léopold | `notebooks/analysis.ipynb` calling every module, in report order |
| `docs/final-readme` | both | Final README and disclosures |

## Disclosures

- **Code reuse**: No external code was reused. This repository rebuilds the team's own earlier project (DGAC airport-pair data) on Eurostat data; its modules were adapted here.
- **AI tools**: Claude (Anthropic), through Claude Code, was used as a coding assistant: debugging, reviewing analysis code, checking that results reproduce on a clean install, and drafting documentation and parts of the notebook commentary. Commits where it contributed carry a `Co-Authored-By: Claude` line. All numbers in the report come from the code in this repository.
