#!/usr/bin/env python3
"""
Scraper for Analysis Part 2: Seed vs. Placement at Div I NAC/Nationals Events.

Extracts seed and placement data from cached fencer history pages.
No new HTTP requests — reads exclusively from the existing cache.

Population: Fencers from the 6 Div I Summer Nationals events.
Target events in history: 2024 NAC and Summer Nationals, DV1 division only.
"""

import csv
import os
import re
import hashlib

from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://fencingtracker.com"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "seed_placement_div1.csv")

# Div I Summer Nationals events only
DIV1_EVENTS = {
    10999: {"gender": "F", "weapon": "epee"},
    11002: {"gender": "M", "weapon": "epee"},
    11004: {"gender": "M", "weapon": "saber"},
    11012: {"gender": "F", "weapon": "foil"},
    11048: {"gender": "M", "weapon": "foil"},
    11049: {"gender": "F", "weapon": "saber"},
}

CSV_COLUMNS = [
    "fencer_id", "fencer_name", "gender", "weapon",
    "tournament_name", "event_id", "event_name", "date",
    "seed", "place", "field_size", "rating",
]


# ── Cache / Fetch (read-only) ───────────────────────────────────────────────

def _cache_path(url):
    """Return a filesystem path for a cached URL."""
    h = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, h + ".html")


def fetch_page(url):
    """Read a page from cache only. Returns None if not cached."""
    cached = _cache_path(url)
    if os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as f:
            return f.read()
    return None


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_event_fencers(html):
    """Extract fencer IDs, names, and profile URLs from an event page."""
    soup = BeautifulSoup(html, "html.parser")
    fencers = []
    seen_ids = set()

    for link in soup.find_all("a", href=re.compile(r'/p/\d+/')):
        href = link.get("href", "")
        id_match = re.search(r'/p/(\d+)/', href)
        if not id_match:
            continue
        fencer_id = id_match.group(1)
        if fencer_id in seen_ids:
            continue
        seen_ids.add(fencer_id)

        name = link.get_text(strip=True)
        if not name:
            continue

        profile_url = href.rstrip("/")
        if not profile_url.endswith("/history"):
            profile_url += "/history"

        fencers.append({
            "name": name,
            "fencer_id": fencer_id,
            "profile_url": profile_url,
        })
    return fencers


def _detect_weapon(event_name):
    """Detect weapon from event name."""
    name_lower = event_name.lower()
    if "epee" in name_lower or "épée" in name_lower:
        return "epee"
    elif "foil" in name_lower:
        return "foil"
    elif "saber" in name_lower or "sabre" in name_lower:
        return "saber"
    return ""


def _parse_bout_table_opponents(table):
    """Extract opponent seed/place/rating/id/name from a bout table.

    Bout table columns (0-indexed):
      0=Bout, 1=V/D, 2=Score, 3=Opponent, 4=flag, 5=Seed,
      6=Rank, 7=Rating, 8=Place, 9=Club, ...

    Returns list of dicts with: opponent_id, opponent_name, seed, place, rating.
    """
    opponents = []
    tbody = table.find("tbody")
    if not tbody:
        return opponents

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 9:
            continue

        # Opponent ID and name
        opponent_link = cells[3].find("a")
        if not opponent_link:
            continue
        opponent_name = opponent_link.get_text(strip=True)
        opponent_href = opponent_link.get("href", "")
        opp_id_match = re.search(r'/p/(\d+)/', opponent_href)
        if not opp_id_match:
            continue
        opponent_id = opp_id_match.group(1)

        # Opponent seed (col 5) and place (col 8)
        opp_seed = cells[5].get_text(strip=True) if len(cells) > 5 else ""
        opp_place = cells[8].get_text(strip=True) if len(cells) > 8 else ""
        opp_rating = cells[7].get_text(strip=True) if len(cells) > 7 else ""

        if opp_seed and opp_seed.isdigit() and opp_place and opp_place.isdigit():
            opponents.append({
                "opponent_id": opponent_id,
                "opponent_name": opponent_name,
                "seed": opp_seed,
                "place": opp_place,
                "rating": opp_rating,
            })

    return opponents


def parse_fencer_history_for_seeds(html):
    """Extract event-level seed/placement data from a fencer's history page.

    Returns list of dicts with: tournament_name, event_id, event_name, date,
    seed, place, field_size, rating, division_code, opponents (list).
    """
    soup = BeautifulSoup(html, "html.parser")

    events = []
    card_body = soup.find("div", class_="card-body")
    if not card_body:
        return events

    current_event = None
    for el in card_body.children:
        if not hasattr(el, 'name') or el.name is None:
            continue

        if el.name == "h4":
            tournament_name = el.get_text(strip=True)
            current_event = {
                "tournament_name": tournament_name,
                "event_name": "", "event_id": "",
                "date": "", "rating": "", "place": "",
                "field_size": "", "seed": "",
                "division_code": "",
                "opponents": [],
            }
            events.append(current_event)

        elif el.name == "h5" and current_event is not None:
            text = el.get_text(strip=True)
            link = el.find("a")
            if link:
                current_event["event_name"] = link.get_text(strip=True)
                event_href = link.get("href", "")
                eid_match = re.search(r'/event/(\d+)/', event_href)
                current_event["event_id"] = eid_match.group(1) if eid_match else ""
            date_match = re.search(r',\s*(\w+ \d+,\s*\d{4})', text)
            if date_match:
                current_event["date"] = date_match.group(1).strip()
            # Extract division code from parenthetical, e.g. "(Mixed, DV1)" or "(Women's, DV1)"
            div_match = re.search(r'\(.*?,\s*(\w+)\)', text)
            if div_match:
                current_event["division_code"] = div_match.group(1)

        elif el.name == "h6" and current_event is not None:
            text = el.get_text(strip=True)
            place_match = re.search(r'Place (\d+) of (\d+)', text)
            if place_match:
                current_event["place"] = place_match.group(1)
                current_event["field_size"] = place_match.group(2)
            seed_match = re.search(r'Seed (\d+)', text)
            if seed_match:
                current_event["seed"] = seed_match.group(1)
            rating_match = re.search(r'Rating (\S+)', text)
            if rating_match:
                current_event["rating"] = rating_match.group(1).rstrip(",")

        elif el.name == "div" and current_event is not None:
            table = el.find("table")
            if table:
                current_event["opponents"] = _parse_bout_table_opponents(table)

    return events


# ── Main Assembly ────────────────────────────────────────────────────────────

def build_div1_population():
    """Build fencer population from Div I Summer Nationals events only."""
    fencer_pop = {}

    for eid, event_info in DIV1_EVENTS.items():
        gender = event_info["gender"]
        weapon = event_info["weapon"]

        print(f"Scraping event {eid} ({weapon} {gender})...")
        event_url = f"{BASE_URL}/event/{eid}"
        html = fetch_page(event_url)
        if not html:
            print(f"  WARNING: No cached page for event {eid}")
            continue

        fencers = parse_event_fencers(html)
        print(f"  Found {len(fencers)} fencers")

        for f in fencers:
            fid = f["fencer_id"]
            if fid not in fencer_pop:
                fencer_pop[fid] = {
                    "name": f["name"],
                    "profile_url": f["profile_url"],
                    "gender": gender,
                    "weapon": weapon,
                }
            # If fencer appears in multiple Div I events (different weapons),
            # keep the first one encountered (weapon from their Nationals event)

    print(f"\nPopulation: {len(fencer_pop)} unique Div I fencers")
    return fencer_pop


def extract_seed_placement(fencer_pop):
    """Extract seed/placement data for 2024 DV1 NAC/Nationals events.

    For each cached fencer, extracts:
    1. The fencer's own seed/placement from the event header
    2. All opponents' seed/placement from the bout table

    Deduplicates at the (fencer_id, event_id) grain so each person appears
    once per event regardless of how many times they're observed.
    """
    rows = []
    seen = set()  # (person_id, event_id) dedup

    def _add_row(person_id, person_name, gender, weapon, ev, seed, place, rating):
        """Add a row if not already seen for this person × event."""
        key = (person_id, ev["event_id"])
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "fencer_id": person_id,
            "fencer_name": person_name,
            "gender": gender,
            "weapon": weapon,
            "tournament_name": ev.get("tournament_name", ""),
            "event_id": ev["event_id"],
            "event_name": ev.get("event_name", ""),
            "date": ev.get("date", ""),
            "seed": int(seed),
            "place": int(place),
            "field_size": int(ev["field_size"]) if ev.get("field_size") else "",
            "rating": rating,
        })

    total = len(fencer_pop)
    for i, (fid, info) in enumerate(fencer_pop.items(), 1):
        if i % 50 == 0 or i == 1:
            print(f"  Processing fencer {i}/{total} ({len(rows)} rows so far, "
                  f"{len(seen)} unique fencer-events)...")

        profile_url = BASE_URL + info["profile_url"]
        html = fetch_page(profile_url)
        if not html:
            continue

        events = parse_fencer_history_for_seeds(html)

        for ev in events:
            # Filter: 2024 only
            if "2024" not in ev.get("date", ""):
                continue

            # Filter: NAC or Summer Nationals
            tname = ev.get("tournament_name", "")
            if "NAC" not in tname and "Summer Nationals" not in tname:
                continue

            # Filter: DV1 division only
            if ev.get("division_code", "") != "DV1":
                continue

            weapon = _detect_weapon(ev.get("event_name", ""))
            if not weapon:
                weapon = info.get("weapon", "")

            # Gender: DV1 events are gender-specific, so all participants
            # (fencer and opponents) share the same gender
            gender = info["gender"]

            # Add the fencer's own seed/placement
            if ev.get("seed") and ev.get("place"):
                _add_row(fid, info["name"], gender, weapon, ev,
                         ev["seed"], ev["place"], ev.get("rating", ""))

            # Add each opponent's seed/placement from the bout table
            for opp in ev.get("opponents", []):
                _add_row(opp["opponent_id"], opp["opponent_name"],
                         gender, weapon, ev,
                         opp["seed"], opp["place"], opp.get("rating", ""))

    return rows


def write_csv(rows, path):
    """Write seed/placement data to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {len(rows)} rows to {path}")


def main():
    print("=" * 60)
    print("SCRAPER 2: Seed vs. Placement — Div I NAC/Nationals 2024")
    print("=" * 60)
    print()

    fencer_pop = build_div1_population()
    if not fencer_pop:
        print("No fencers found. Exiting.")
        return

    print(f"\nExtracting seed/placement data...")
    rows = extract_seed_placement(fencer_pop)

    if rows:
        write_csv(rows, OUTPUT_CSV)
    else:
        print("No seed/placement data found.")
        return

    # Summary
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Unique fencers: {df['fencer_id'].nunique()}")
    print(f"Unique events: {df['event_id'].nunique()}")
    print(f"Tournaments: {df['tournament_name'].unique().tolist()}")
    print(f"\nBy weapon & gender:")
    print(df.groupby(["weapon", "gender"]).size().to_string())
    print(f"\nBy tournament:")
    print(df.groupby("tournament_name").size().to_string())


if __name__ == "__main__":
    main()
