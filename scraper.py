#!/usr/bin/env python3
"""
FencingTracker Scraper v2 — collects bout-level data from fencingtracker.com

Population: All fencers who competed in Division I, IA, II, or III events
at the 2024 Summer Nationals (tournament 1041).

Data: All bouts in calendar year 2024 for each fencer in the population.
"""

import csv
import os
import re
import time
import hashlib

import requests
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://fencingtracker.com"
DELAY = 2  # seconds between requests
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fencingtracker_bouts_2024.csv")

HEADERS = {
    "User-Agent": "FencingResearchBot/1.0 (academic research; rate-limited)",
    "Accept": "text/html",
}

# 2024 Summer Nationals — Division I, IA, II, III events
# Extracted from https://fencingtracker.com/tournament/1041
SUMMER_NATIONALS_EVENTS = {
    # Division I
    10999: {"division": "I",  "gender": "F", "weapon": "epee",  "event_code": "DV1WE"},
    11002: {"division": "I",  "gender": "M", "weapon": "epee",  "event_code": "DV1ME"},
    11004: {"division": "I",  "gender": "M", "weapon": "saber", "event_code": "DV1MS"},
    11012: {"division": "I",  "gender": "F", "weapon": "foil",  "event_code": "DV1WF"},
    11048: {"division": "I",  "gender": "M", "weapon": "foil",  "event_code": "DV1MF"},
    11049: {"division": "I",  "gender": "F", "weapon": "saber", "event_code": "DV1WS"},
    # Division IA
    11001: {"division": "IA", "gender": "F", "weapon": "saber", "event_code": "D1AWS"},
    11026: {"division": "IA", "gender": "M", "weapon": "saber", "event_code": "D1AMS"},
    11027: {"division": "IA", "gender": "F", "weapon": "epee",  "event_code": "D1AWE"},
    11050: {"division": "IA", "gender": "M", "weapon": "epee",  "event_code": "D1AME"},
    11051: {"division": "IA", "gender": "F", "weapon": "foil",  "event_code": "D1AWF"},
    11053: {"division": "IA", "gender": "M", "weapon": "foil",  "event_code": "D1AMF"},
    # Division II
    11065: {"division": "II", "gender": "F", "weapon": "foil",  "event_code": "DV2WF"},
    11067: {"division": "II", "gender": "M", "weapon": "epee",  "event_code": "DV2ME"},
    11076: {"division": "II", "gender": "F", "weapon": "saber", "event_code": "DV2WS"},
    11077: {"division": "II", "gender": "F", "weapon": "epee",  "event_code": "DV2WE"},
    11079: {"division": "II", "gender": "M", "weapon": "saber", "event_code": "DV2MS"},
    11086: {"division": "II", "gender": "M", "weapon": "foil",  "event_code": "DV2MF"},
    # Division III
    11062: {"division": "III", "gender": "F", "weapon": "epee",  "event_code": "DV3WE"},
    11075: {"division": "III", "gender": "F", "weapon": "foil",  "event_code": "DV3WF"},
    11085: {"division": "III", "gender": "M", "weapon": "epee",  "event_code": "DV3ME"},
    11087: {"division": "III", "gender": "F", "weapon": "saber", "event_code": "DV3WS"},
    11088: {"division": "III", "gender": "M", "weapon": "foil",  "event_code": "DV3MF"},
    11091: {"division": "III", "gender": "M", "weapon": "saber", "event_code": "DV3MS"},
}

# CSV columns
CSV_COLUMNS = [
    "bout_id",
    "tournament_name", "event_id", "event_name", "weapon", "date",
    "fencer_name", "fencer_id", "fencer_gender", "fencer_rating",
    "fencer_birth_year", "fencer_club",
    "opponent_name", "opponent_id", "opponent_gender",
    "opponent_rating", "opponent_elo", "opponent_club",
    "fencer_elo",
    "result", "score", "bout_type", "bout_type_raw",
    "is_de", "win_probability",
    "fencer_divisions",
]


# ── HTTP / Caching ───────────────────────────────────────────────────────────

def _cache_path(url):
    """Return a filesystem path for caching a URL's content."""
    h = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, h + ".html")


def fetch_page(url):
    """GET a page with caching, rate-limiting, and retry."""
    cached = _cache_path(url)
    if os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as f:
            return f.read()

    for attempt in range(3):
        try:
            print(f"  Fetching: {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cached, "w", encoding="utf-8") as f:
                f.write(resp.text)
            time.sleep(DELAY)
            return resp.text
        except requests.RequestException as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(DELAY * (attempt + 1))
    print(f"  ERROR: Could not fetch {url}")
    return None


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_event_fencers(html):
    """Extract fencer IDs, names, and profile URLs from an event page.

    The event page (not /results) lists fencers in a table with links to /p/{id}/{name}.
    Returns list of dicts with: name, fencer_id, profile_url.
    """
    soup = BeautifulSoup(html, "html.parser")

    fencers = []
    seen_ids = set()

    # Find all fencer profile links on the page (pattern: /p/{id}/{name})
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

        # Ensure profile URL ends with /history for bout data
        profile_url = href.rstrip("/")
        if not profile_url.endswith("/history"):
            profile_url += "/history"

        fencers.append({
            "name": name,
            "fencer_id": fencer_id,
            "profile_url": profile_url,
        })

    return fencers


def parse_fencer_history(html):
    """Extract fencer profile info and bout history from their history page.

    Returns dict with: birth_year, club, events (list of event data with bouts).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Birth year
    birth_year = ""
    header = soup.find("div", class_="card-header")
    if header:
        h3 = header.find("h3")
        if h3:
            text = h3.get_text(strip=True)
            if re.match(r'^\d{4}$', text):
                birth_year = text

    # Club
    club = ""
    if header:
        club_link = header.find("a", href=re.compile(r'/club/'))
        if club_link:
            club = club_link.get_text(strip=True)

    # Parse events and their bouts
    events = []
    card_body = soup.find("div", class_="card-body")
    if not card_body:
        return {"birth_year": birth_year, "club": club, "events": []}

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
                "bouts": [],
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

        elif el.name == "h6" and current_event is not None:
            text = el.get_text(strip=True)
            place_match = re.search(r'Place (\d+) of (\d+)', text)
            if place_match:
                current_event["place"] = place_match.group(1)
            rating_match = re.search(r'Rating (\S+)', text)
            if rating_match:
                current_event["rating"] = rating_match.group(1).rstrip(",")

        elif el.name == "div" and current_event is not None:
            table = el.find("table")
            if table:
                current_event["bouts"] = _parse_bout_table(table)

    return {"birth_year": birth_year, "club": club, "events": events}


def _parse_bout_table(table):
    """Parse a bout table from a fencer's history page.

    Columns: Bout, V/D, Score, Opponent, flag, Seed, Rank, Rating, Place, Club,
             Opponent Strength, Strength, Change, Chance of Victory
    """
    bouts = []
    tbody = table.find("tbody")
    if not tbody:
        return bouts

    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 10:
            continue

        bout_type_raw = cells[0].get_text(strip=True)  # Pool, T256, T128, etc.
        result_span = cells[1].find("span")
        result = result_span.get_text(strip=True) if result_span else cells[1].get_text(strip=True)
        score = cells[2].get_text(strip=True)

        opponent_link = cells[3].find("a")
        opponent_name = opponent_link.get_text(strip=True) if opponent_link else cells[3].get_text(strip=True)
        opponent_href = opponent_link.get("href", "") if opponent_link else ""
        opp_id_match = re.search(r'/p/(\d+)/', opponent_href)
        opponent_id = opp_id_match.group(1) if opp_id_match else ""

        opponent_rating = cells[7].get_text(strip=True) if len(cells) > 7 else ""
        opponent_club = cells[9].get_text(strip=True) if len(cells) > 9 else ""

        # ELO ratings: index 10 = opponent strength, index 11 = fencer strength
        opponent_elo = ""
        fencer_elo = ""
        if len(cells) > 10:
            elo_text = cells[10].get_text(strip=True)
            if elo_text and re.match(r'^\d+$', elo_text):
                opponent_elo = elo_text
        if len(cells) > 11:
            elo_text = cells[11].get_text(strip=True)
            if elo_text and re.match(r'^\d+$', elo_text):
                fencer_elo = elo_text

        # Chance of Victory — index 13
        win_probability = ""
        if len(cells) >= 14:
            wp_text = cells[13].get_text(strip=True)
            wp_match = re.search(r'(\d+)%', wp_text)
            if wp_match:
                win_probability = wp_match.group(1)

        # Normalize bout type
        if bout_type_raw.startswith("T") or bout_type_raw in ("Finals", "Semi", "Quarter"):
            bout_type = "DE"
        elif bout_type_raw.lower() == "pool":
            bout_type = "Pool"
        else:
            bout_type = bout_type_raw

        bouts.append({
            "bout_type": bout_type,
            "bout_type_raw": bout_type_raw,
            "result": result,
            "score": score,
            "opponent_name": opponent_name,
            "opponent_id": opponent_id,
            "opponent_rating": opponent_rating,
            "opponent_club": opponent_club,
            "opponent_elo": opponent_elo,
            "fencer_elo": fencer_elo,
            "win_probability": win_probability,
        })

    return bouts


def _is_target_year(date_str):
    """Check if a date string contains the target year (2024)."""
    return "2024" in date_str


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


# ── Main Assembly ────────────────────────────────────────────────────────────

def build_fencer_population(event_ids=None):
    """Phase 1: Scrape Summer Nationals events to build fencer population.

    Returns:
        fencer_pop: dict mapping fencer_id -> {name, profile_url, gender, divisions}
    """
    if event_ids is None:
        event_ids = list(SUMMER_NATIONALS_EVENTS.keys())

    fencer_pop = {}  # fencer_id -> {name, profile_url, gender, divisions: set}

    for eid in event_ids:
        event_info = SUMMER_NATIONALS_EVENTS.get(eid, {})
        gender = event_info.get("gender", "")
        division = event_info.get("division", "")
        event_code = event_info.get("event_code", str(eid))

        print(f"\n{'='*60}")
        print(f"Phase 1: Scraping event {eid} ({event_code})")
        print(f"{'='*60}")

        event_url = f"{BASE_URL}/event/{eid}"
        results_html = fetch_page(event_url)
        if not results_html:
            print(f"  WARNING: Could not fetch results for event {eid}")
            continue

        fencers = parse_event_fencers(results_html)
        print(f"  Found {len(fencers)} fencers")

        for f in fencers:
            fid = f["fencer_id"]
            if fid in fencer_pop:
                # Fencer already seen — add this division
                fencer_pop[fid]["divisions"].add(division)
            else:
                fencer_pop[fid] = {
                    "name": f["name"],
                    "profile_url": f["profile_url"],
                    "gender": gender,
                    "divisions": {division},
                }

    print(f"\n{'='*60}")
    print(f"Phase 1 complete: {len(fencer_pop)} unique fencers across {len(event_ids)} events")
    print(f"{'='*60}")

    # Gender breakdown
    m_count = sum(1 for f in fencer_pop.values() if f["gender"] == "M")
    f_count = sum(1 for f in fencer_pop.values() if f["gender"] == "F")
    print(f"  Men: {m_count}, Women: {f_count}")

    return fencer_pop


def scrape_fencer_bouts(fencer_pop):
    """Phase 2: Scrape all 2025 bouts for each fencer in the population.

    Returns:
        all_bouts: list of bout dicts ready for CSV
        seen_bout_keys: set used for deduplication
    """
    all_bouts = []
    seen_bout_keys = set()  # (sorted_id_1, sorted_id_2, event_id, score_canonical, bout_type_raw)

    total = len(fencer_pop)
    import time as _time
    _phase2_start = _time.time()
    for i, (fid, fencer_info) in enumerate(fencer_pop.items(), 1):
        if i % 20 == 0 or i == 1:
            elapsed = _time.time() - _phase2_start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"\n[Progress] Fencer {i}/{total} | {len(all_bouts)} bouts | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining", flush=True)

        profile_url = BASE_URL + fencer_info["profile_url"]
        history_html = fetch_page(profile_url)
        if not history_html:
            print(f"  WARNING: Could not fetch history for {fencer_info['name']}")
            continue

        profile = parse_fencer_history(history_html)
        fencer_gender = fencer_info["gender"]
        fencer_divisions = ",".join(sorted(fencer_info["divisions"]))

        # Iterate over all events in fencer's history, filtering to 2025
        bout_count = 0
        for ev in profile.get("events", []):
            if not _is_target_year(ev.get("date", "")):
                continue

            event_id = ev.get("event_id", "")
            event_name = ev.get("event_name", "")
            weapon = _detect_weapon(event_name)
            fencer_rating = ev.get("rating", "")

            for bout in ev.get("bouts", []):
                opp_id = bout["opponent_id"]

                # ── Deduplication ──
                # Use canonical key: sorted fencer IDs + event_id + canonical score + raw bout type
                # This ensures each bout appears exactly once even when both
                # fencers are in our population.
                ids_sorted = tuple(sorted([fid, opp_id]))
                score_parts = bout["score"].split(":")
                if len(score_parts) == 2:
                    score_canonical = ":".join(sorted(score_parts, reverse=True))
                else:
                    score_canonical = bout["score"]
                bout_key = (ids_sorted[0], ids_sorted[1], event_id, score_canonical, bout["bout_type_raw"])

                if bout_key in seen_bout_keys:
                    continue
                seen_bout_keys.add(bout_key)

                # ── Gender inference for opponent ──
                # If win_probability is present, opponent is same gender as fencer.
                # If absent, opponent is different gender.
                wp = bout.get("win_probability", "")
                if wp:
                    opponent_gender = fencer_gender
                else:
                    # Cross-gender or unknown
                    # Check if opponent is in our population
                    if opp_id in fencer_pop:
                        opponent_gender = fencer_pop[opp_id]["gender"]
                    else:
                        # No win probability and not in population — could be cross-gender
                        # or simply unrated. Mark as opposite gender per the plan's logic.
                        opponent_gender = "F" if fencer_gender == "M" else "M" if fencer_gender == "F" else ""

                # Determine winner indicator
                winner_indicator = "W" if bout["result"] == "V" else "L"

                bout_id = f"{event_id}_{fid}_{opp_id}_{bout['bout_type_raw']}_{bout['score']}"

                all_bouts.append({
                    "bout_id": bout_id,
                    "tournament_name": ev.get("tournament_name", ""),
                    "event_id": event_id,
                    "event_name": event_name,
                    "weapon": weapon,
                    "date": ev.get("date", ""),
                    "fencer_name": fencer_info["name"],
                    "fencer_id": fid,
                    "fencer_gender": fencer_gender,
                    "fencer_rating": fencer_rating,
                    "fencer_birth_year": profile.get("birth_year", ""),
                    "fencer_club": profile.get("club", ""),
                    "opponent_name": bout["opponent_name"],
                    "opponent_id": opp_id,
                    "opponent_gender": opponent_gender,
                    "opponent_rating": bout.get("opponent_rating", ""),
                    "opponent_elo": bout.get("opponent_elo", ""),
                    "opponent_club": bout.get("opponent_club", ""),
                    "fencer_elo": bout.get("fencer_elo", ""),
                    "result": winner_indicator,
                    "score": bout["score"],
                    "bout_type": bout["bout_type"],
                    "bout_type_raw": bout["bout_type_raw"],
                    "is_de": bout["bout_type"] == "DE",
                    "win_probability": wp,
                    "fencer_divisions": fencer_divisions,
                })
                bout_count += 1

        pass  # progress printed every 20 fencers above

    return all_bouts


def write_csv(all_bouts, path):
    """Write bout data to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for bout in all_bouts:
            writer.writerow(bout)
    print(f"\nWrote {len(all_bouts)} bouts to {path}")


def main():
    """Main entry point: two-phase scrape and write CSV."""
    import argparse
    parser = argparse.ArgumentParser(description="Scrape fencingtracker.com bout data (2024 bouts, Summer Nationals population)")
    parser.add_argument("--events", type=int, nargs="*", default=None,
                        help="Summer Nationals event IDs to use for population (default: all 24)")
    parser.add_argument("--limit-events", type=int, default=None,
                        help="Max number of events to scrape for population")
    parser.add_argument("--limit-fencers", type=int, default=None,
                        help="Max number of fencers to scrape bouts for")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="Output CSV path")
    args = parser.parse_args()

    # Phase 1: Build population
    event_ids = args.events if args.events else list(SUMMER_NATIONALS_EVENTS.keys())
    if args.limit_events:
        event_ids = event_ids[:args.limit_events]

    fencer_pop = build_fencer_population(event_ids)

    if not fencer_pop:
        print("No fencers found. Exiting.")
        return

    # Optionally limit fencers for testing
    if args.limit_fencers:
        limited = dict(list(fencer_pop.items())[:args.limit_fencers])
        print(f"\nLimiting to {len(limited)} fencers for testing")
        fencer_pop = limited

    # Phase 2: Scrape all 2025 bouts
    all_bouts = scrape_fencer_bouts(fencer_pop)

    if all_bouts:
        write_csv(all_bouts, args.output)
    else:
        print("No bouts collected.")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Population events: {len(event_ids)}")
    print(f"Unique fencers: {len(fencer_pop)}")
    print(f"Total bouts: {len(all_bouts)}")
    pool_bouts = sum(1 for b in all_bouts if b["bout_type"] == "Pool")
    de_bouts = sum(1 for b in all_bouts if b["bout_type"] == "DE")
    print(f"  Pool bouts: {pool_bouts}")
    print(f"  DE bouts: {de_bouts}")
    m_bouts = sum(1 for b in all_bouts if b["fencer_gender"] == "M")
    f_bouts = sum(1 for b in all_bouts if b["fencer_gender"] == "F")
    print(f"  Bouts from male fencers: {m_bouts}")
    print(f"  Bouts from female fencers: {f_bouts}")
    wp_present = sum(1 for b in all_bouts if b["win_probability"])
    wp_absent = sum(1 for b in all_bouts if not b["win_probability"])
    print(f"  With win probability (same gender): {wp_present}")
    print(f"  Without win probability (cross/unknown): {wp_absent}")


if __name__ == "__main__":
    main()
