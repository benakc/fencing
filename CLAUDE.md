# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fencing analytics project that scrapes bout-level data from fencingtracker.com and performs statistical analysis on competitive fencing results. The population of interest is fencers who competed at the 2024 USA Fencing Summer Nationals (Divisions I, IA, II, III).

## Running Commands

```bash
# Run the scraper (uses cached HTML by default, 2s delay between live requests)
python3 scraper.py

# Run with limited scope for testing
python3 scraper.py --limit-events 2 --limit-fencers 10

# Run the full analysis (generates summary text, CSVs, and PNG plots)
python3 analysis.py
```

Python environment: Anaconda (`/Users/bcohen/anaconda3/bin/python3`). Key dependencies: requests, beautifulsoup4, pandas, numpy, matplotlib, seaborn, statsmodels.

## Architecture

### Data Pipeline

1. **`scraper.py`** — Two-phase scraper for fencingtracker.com:
   - Phase 1: Scrapes Summer Nationals event pages to build a fencer population (fencer IDs, names, genders, divisions)
   - Phase 2: Fetches each fencer's `/history` page and extracts all 2024 bouts
   - Deduplicates bouts using canonical keys (sorted fencer IDs + event + score + bout type)
   - Infers opponent gender from win_probability presence (present = same gender, absent = cross-gender)
   - Output: `fencingtracker_bouts_2024.csv`

2. **`analysis.py`** — Statistical analysis sliced by bout type (Total/DE/Pool) × gender matchup (Total/M/F/Mixed):
   - Rating and ELO cross-tabulations with win percentages
   - Logistic regression: P(win) ~ elo_diff + weapon
   - Upset analysis (lower-rated fencer beating higher-rated)
   - Cross-gender analysis with oriented F-vs-M perspective
   - Generates ~30 PNG visualizations and summary text files
   - Uses vectorized pandas operations with a "mirroring" pattern (counts each bout from both fencer perspectives)

### Cache System

The `cache/` directory stores raw HTML responses keyed by MD5 hash of the URL. The scraper checks cache before making HTTP requests, so re-runs are fast and don't hit the server.

### Analysis Outputs

- `Analysis Part 1/` — Results from the first round of analysis (includes earlier CSV data, plots, and gender analysis)
- `Analysis Part 2/` — Planned for Division I NAC/Nationals seed-vs-placement analysis (see `analysis_plan_v3.txt`)
- `point_predict/` — Planned point-by-point prediction model using fencingworldwide.com data

### Key Domain Concepts

- **Ratings**: USFA letter ratings A > B > C > D > E > U (unrated). The number suffix is the year earned (e.g., A25).
- **ELO**: Separate from ratings; a numerical strength score tracked by fencingtracker.com.
- **Bout types**: Pool (round-robin, to 5 touches) and DE (direct elimination, to 15 touches).
- **Weapons**: epee, foil, saber — each has different scoring rules and competitive dynamics.

### Important Conventions

- Do not delete or overwrite existing scrapers, analysis files, or results when creating new analyses. Create new files in the appropriate `Analysis Part` folder.
- The scraper is rate-limited (2s delay) and identifies itself as `FencingResearchBot/1.0`.
- All analysis slicing uses the `slice_df()` helper which filters by bout_type and gender_matchup.
