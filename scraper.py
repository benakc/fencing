#!/usr/bin/env python3
"""
FencingTracker Scraper — collects bout-level data from fencingtracker.com
Focuses on Senior Mixed events (epee, foil, saber).
"""

import csv
import os
import re
import time
import hashlib
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://fencingtracker.com"
DELAY = 2  # seconds between requests
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fencingtracker_bouts.csv")

HEADERS = {
    "User-Agent": "FencingResearchBot/1.0 (academic research; rate-limited)",
    "Accept": "text/html",
}

# Seed event IDs — Senior Mixed events across weapons
SEED_EVENTS = [
    # Epee
    34995,  # AFM Open Senior Mixed Epee (61 fencers, Oct 2025)
    31137,  # Fence Fest Senior Mixed Epee (25 fencers, Mar 2025)
    # Foil
    19911,  # GT Collegiate: Fall Hack & Slash Senior Mixed Foil (31 fencers, Nov 2023)
    14652,  # Hangover Open Walk N Roll Senior Mixed Foil (25 fencers, Jan 2023)
    26415,  # Back to School Open Senior Mixed Foil (24 fencers, Sep 2024)
    33431,  # Weekend Warrior I Senior Mixed Foil (22 fencers, Aug 2025)
    33948,  # AIC Foil Open Senior Mixed Foil (15 fencers, Sep 2025)
    34935,  # SAS Foil and Epee Open Senior Mixed Foil (15 fencers, Oct 2025)
    33320,  # Steel Clash Senior Mixed Foil (5 fencers)
    # Saber
    31694,  # 57th Annual Green Gator Senior Mixed Saber (30 fencers, Apr 2025)
    29857,  # Texas Fencing Academy Cup 3 Senior Mixed Saber (28 fencers, Feb 2025)
    32794,  # Scarsdale Fencing Cup #1 Mixed Saber (26 fencers, Jun 2025)
    33090,  # Shoreline Pre-Nat Open Senior Mixed Saber (19 fencers, Jun 2025)
    31984,  # Sebastian Ramirez Amaya Memorial Senior Mixed Saber (18 fencers, Apr 2025)
    32442,  # Attila Petschauer Senior Mixed Saber (7 fencers)
    30480,  # Sebastiani Spring Senior Mixed Saber (2 fencers)
]

# CSV columns
CSV_COLUMNS = [
    "bout_id", "tournament_name", "event_id", "event_name", "weapon", "date", "location",
    "fencer_1_name", "fencer_1_id", "fencer_1_rating", "fencer_1_elo_pool_before",
    "fencer_1_elo_pool_after", "fencer_1_elo_de_before", "fencer_1_elo_de_after",
    "fencer_1_birth_year", "fencer_1_club", "fencer_1_place",
    "fencer_2_name", "fencer_2_id", "fencer_2_rating", "fencer_2_elo_pool_before",
    "fencer_2_elo_pool_after", "fencer_2_elo_de_before", "fencer_2_elo_de_after",
    "fencer_2_birth_year", "fencer_2_club", "fencer_2_place",
    "winner", "score", "bout_type",
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

def parse_event_metadata(html):
    """Extract event name, weapon, date, location from the results page."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title_text = title_tag.text.strip() if title_tag else ""

    # Title format: "Results of Senior Mixed Épée - AFM Open ... - FencingTracker"
    parts = title_text.split(" - ")
    event_name = parts[0].replace("Results of ", "").strip() if parts else ""
    tournament_name = parts[1].strip() if len(parts) > 1 else ""

    # Weapon detection from event name
    weapon = ""
    name_lower = event_name.lower()
    if "epee" in name_lower or "épée" in name_lower:
        weapon = "epee"
    elif "foil" in name_lower:
        weapon = "foil"
    elif "saber" in name_lower or "sabre" in name_lower:
        weapon = "saber"

    # Date: look for h4 tag with date
    date_str = ""
    for h4 in soup.find_all("h4"):
        text = h4.get_text(strip=True)
        # Pattern: "Sunday, October 19, 2025 at 9:00 AM"
        if re.search(r'\b\d{4}\b', text) and ('AM' in text or 'PM' in text or re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text)):
            date_str = text
            break

    # Location: in a <p> tag like "Academy of Fencing Masters - Sunnyvale, CA, USA"
    location = ""
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if re.search(r',\s*[A-Z]{2},\s*USA', text) and len(text) < 200:
            location = text
            break

    return {
        "event_name": event_name,
        "tournament_name": tournament_name,
        "weapon": weapon,
        "date": date_str,
        "location": location,
    }


def parse_event_results(html):
    """Extract fencer info and bout results from the event results page.

    Returns list of dicts with: name, fencer_id, profile_url, place,
    and pool_bouts / de_bouts (lists of {result, score, opponent_name}).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="resultsTable")
    if not table:
        return []

    fencers = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        place = cells[0].get_text(strip=True)
        link = cells[1].find("a")
        if not link:
            continue

        href = link.get("href", "")
        # /p/{id}/{Name}/history
        id_match = re.search(r'/p/(\d+)/', href)
        fencer_id = id_match.group(1) if id_match else ""
        name = link.get_text(strip=True)

        # Pool bouts (cell 2) and DE bouts (cell 3) — span elements with data-tt
        pool_bouts = _parse_bout_spans(cells[2])
        de_bouts = _parse_bout_spans(cells[3])

        fencers.append({
            "name": name,
            "fencer_id": fencer_id,
            "profile_url": href,
            "place": place,
            "pool_bouts": pool_bouts,
            "de_bouts": de_bouts,
        })

    return fencers


def _parse_bout_spans(cell):
    """Parse bout spans from a results table cell.

    Each span has data-tt like '5:4 vs. GUIRAUDET Alistair - Very Easy'
    and inner text V or D.
    """
    bouts = []
    for span in cell.find_all("span", attrs={"data-tt": True}):
        tt = span["data-tt"]
        result = span.get_text(strip=True)  # V or D

        # Parse "5:4 vs. GUIRAUDET Alistair - Very Easy"
        m = re.match(r'(\d+:\d+)\s+vs\.\s+(.+?)\s*-\s*(.*)', tt)
        if m:
            score = m.group(1)
            opponent = m.group(2).strip()
            difficulty = m.group(3).strip()
        else:
            score, opponent, difficulty = tt, "", ""

        bouts.append({
            "result": result,
            "score": score,
            "opponent_name": opponent,
            "difficulty": difficulty,
        })
    return bouts


def parse_event_strength(html):
    """Extract ELO/strength data from the strength page.

    Returns dict keyed by fencer_id with pool_before, pool_after, de_before, de_after.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="resultsTable")
    if not table:
        return {}

    strengths = {}
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        link = cells[1].find("a")
        if not link:
            continue
        href = link.get("href", "")
        id_match = re.search(r'/p/(\d+)/', href)
        fencer_id = id_match.group(1) if id_match else ""

        # Columns: #, Name, Pool Before, Pool After, Pool Change, DE Before, DE After, DE Change
        def safe_int(cell):
            text = cell.get_text(strip=True)
            # Remove arrows and signs, keep digits
            digits = re.sub(r'[^\d]', '', text)
            return int(digits) if digits else None

        pool_before = safe_int(cells[2])
        pool_after = safe_int(cells[3])
        # cells[4] is change
        de_before = safe_int(cells[5]) if len(cells) > 5 else None
        de_after = safe_int(cells[6]) if len(cells) > 6 else None

        strengths[fencer_id] = {
            "pool_before": pool_before,
            "pool_after": pool_after,
            "de_before": de_before,
            "de_after": de_after,
        }

    return strengths


def parse_fencer_history(html):
    """Extract fencer profile info and bout history from their history page.

    Returns dict with: birth_year, club, rating, events (list of event bouts).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Birth year: in an h3 tag inside the header area
    birth_year = ""
    header = soup.find("div", class_="card-header")
    if header:
        h3 = header.find("h3")
        if h3:
            text = h3.get_text(strip=True)
            if re.match(r'^\d{4}$', text):
                birth_year = text

    # Club: first link in the profile header area to /club/
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

    # Events are grouped by h4 (tournament name), h5 (event + date), h6 (placement info)
    # then a table of bouts
    current_event = None
    for el in card_body.children:
        if not hasattr(el, 'name') or el.name is None:
            continue

        if el.name == "h4":
            tournament_name = el.get_text(strip=True)
            current_event = {"tournament_name": tournament_name, "event_name": "", "date": "", "rating": "", "place": "", "bouts": []}
            events.append(current_event)

        elif el.name == "h5" and current_event is not None:
            # "<a href="/event/34995/results">Senior Mixed Épée</a> (A2, SNR), October 19, 2025"
            text = el.get_text(strip=True)
            link = el.find("a")
            if link:
                current_event["event_name"] = link.get_text(strip=True)
                event_href = link.get("href", "")
                eid_match = re.search(r'/event/(\d+)/', event_href)
                current_event["event_id"] = eid_match.group(1) if eid_match else ""
            # Date is after the last comma
            date_match = re.search(r',\s*(\w+ \d+,\s*\d{4})', text)
            if date_match:
                current_event["date"] = date_match.group(1).strip()

        elif el.name == "h6" and current_event is not None:
            text = el.get_text(strip=True)
            # "Place 1 of 61, Seed 1, Not ranked, Rating A25, Academy Of Fencing Masters (AFM)"
            place_match = re.search(r'Place (\d+) of (\d+)', text)
            if place_match:
                current_event["place"] = place_match.group(1)
            rating_match = re.search(r'Rating (\S+)', text)
            if rating_match:
                current_event["rating"] = rating_match.group(1)
                # Strip trailing comma if present
                current_event["rating"] = current_event["rating"].rstrip(",")

        elif el.name == "div" and current_event is not None:
            # Look for bout table inside this div
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

        bout_type = cells[0].get_text(strip=True)  # Pool, T256, T128, etc.
        result_span = cells[1].find("span")
        result = result_span.get_text(strip=True) if result_span else cells[1].get_text(strip=True)
        score = cells[2].get_text(strip=True)

        opponent_link = cells[3].find("a")
        opponent_name = opponent_link.get_text(strip=True) if opponent_link else cells[3].get_text(strip=True)
        opponent_href = opponent_link.get("href", "") if opponent_link else ""
        opp_id_match = re.search(r'/p/(\d+)/', opponent_href)
        opponent_id = opp_id_match.group(1) if opp_id_match else ""

        # cells[4] is flag
        opponent_rating = cells[7].get_text(strip=True) if len(cells) > 7 else ""
        opponent_place = cells[8].get_text(strip=True) if len(cells) > 8 else ""
        opponent_club = cells[9].get_text(strip=True) if len(cells) > 9 else ""

        # Normalize bout type: Pool stays "Pool", T-numbers become "DE"
        if bout_type.startswith("T") or bout_type in ("Finals", "Semi", "Quarter"):
            bout_type_normalized = "DE"
        elif bout_type.lower() == "pool":
            bout_type_normalized = "Pool"
        else:
            bout_type_normalized = bout_type

        bouts.append({
            "bout_type": bout_type_normalized,
            "bout_type_raw": bout_type,
            "result": result,
            "score": score,
            "opponent_name": opponent_name,
            "opponent_id": opponent_id,
            "opponent_rating": opponent_rating,
            "opponent_place": opponent_place,
            "opponent_club": opponent_club,
        })

    return bouts


# ── Main Assembly ────────────────────────────────────────────────────────────

def scrape_event(event_id):
    """Scrape a single event and return bout-level rows."""
    print(f"\n{'='*60}")
    print(f"Scraping event {event_id}")
    print(f"{'='*60}")

    # 1. Fetch and parse event results page
    results_url = f"{BASE_URL}/event/{event_id}/results"
    results_html = fetch_page(results_url)
    if not results_html:
        return []

    metadata = parse_event_metadata(results_html)
    fencers = parse_event_results(results_html)
    print(f"  Found {len(fencers)} fencers in {metadata['event_name']}")

    # 2. Fetch and parse strength page
    strength_url = f"{BASE_URL}/event/{event_id}/results/strength"
    strength_html = fetch_page(strength_url)
    strengths = parse_event_strength(strength_html) if strength_html else {}

    # Build fencer lookup from results and strength
    fencer_lookup = {}
    for f in fencers:
        fid = f["fencer_id"]
        s = strengths.get(fid, {})
        fencer_lookup[fid] = {
            "name": f["name"],
            "place": f["place"],
            "pool_before": s.get("pool_before"),
            "pool_after": s.get("pool_after"),
            "de_before": s.get("de_before"),
            "de_after": s.get("de_after"),
        }

    # 3. Fetch fencer history pages to get ratings, birth years, clubs, and bout details
    fencer_profiles = {}
    for f in fencers:
        fid = f["fencer_id"]
        profile_url = BASE_URL + f["profile_url"]
        history_html = fetch_page(profile_url)
        if history_html:
            profile = parse_fencer_history(history_html)
            fencer_profiles[fid] = profile

    # 4. Assemble bout rows from fencer histories
    # We use fencer history pages as the source of truth for bout data,
    # matching bouts to this event by event_id.
    bouts = []
    seen_bout_keys = set()  # track (fencer1_id, fencer2_id, score, bout_type) to deduplicate

    for f in fencers:
        fid = f["fencer_id"]
        profile = fencer_profiles.get(fid)
        if not profile:
            continue

        # Find this event in the fencer's history
        event_bouts = []
        for ev in profile.get("events", []):
            if ev.get("event_id") == str(event_id):
                event_bouts = ev.get("bouts", [])
                break

        for bout in event_bouts:
            opp_id = bout["opponent_id"]

            # Create a canonical key to avoid double-counting bouts
            # (each bout appears in both fencers' histories)
            # Normalize score: "5:3" and "3:5" refer to the same bout
            ids_sorted = tuple(sorted([fid, opp_id]))
            score_parts = bout["score"].split(":")
            score_canonical = ":".join(sorted(score_parts, reverse=True)) if len(score_parts) == 2 else bout["score"]
            bout_key = (ids_sorted[0], ids_sorted[1], score_canonical, bout["bout_type"])
            if bout_key in seen_bout_keys:
                continue
            seen_bout_keys.add(bout_key)

            # Determine winner
            if bout["result"] == "V":
                winner = 1
                fencer_1_id, fencer_2_id = fid, opp_id
            else:
                winner = 2
                fencer_1_id, fencer_2_id = fid, opp_id

            f1_info = fencer_lookup.get(fencer_1_id, {})
            f2_info = fencer_lookup.get(fencer_2_id, {})
            f1_profile = fencer_profiles.get(fencer_1_id, {})
            f2_profile = fencer_profiles.get(fencer_2_id, {})

            # Get rating for fencer 1 from their event entry
            f1_rating = ""
            for ev in f1_profile.get("events", []):
                if ev.get("event_id") == str(event_id):
                    f1_rating = ev.get("rating", "")
                    break

            # For fencer 2, if they're in our profiles use that, otherwise use opponent_rating from bout
            f2_rating = ""
            if fencer_2_id in fencer_profiles:
                for ev in fencer_profiles[fencer_2_id].get("events", []):
                    if ev.get("event_id") == str(event_id):
                        f2_rating = ev.get("rating", "")
                        break
            if not f2_rating:
                f2_rating = bout.get("opponent_rating", "")

            bout_id = f"{event_id}_{fencer_1_id}_{fencer_2_id}_{bout['bout_type']}_{bout['score']}"

            bouts.append({
                "bout_id": bout_id,
                "tournament_name": metadata["tournament_name"],
                "event_id": str(event_id),
                "event_name": metadata["event_name"],
                "weapon": metadata["weapon"],
                "date": metadata["date"],
                "location": metadata["location"],
                "fencer_1_name": f1_info.get("name", bout.get("opponent_name", "") if bout["result"] == "D" else f["name"]),
                "fencer_1_id": fencer_1_id,
                "fencer_1_rating": f1_rating,
                "fencer_1_elo_pool_before": f1_info.get("pool_before", ""),
                "fencer_1_elo_pool_after": f1_info.get("pool_after", ""),
                "fencer_1_elo_de_before": f1_info.get("de_before", ""),
                "fencer_1_elo_de_after": f1_info.get("de_after", ""),
                "fencer_1_birth_year": f1_profile.get("birth_year", ""),
                "fencer_1_club": f1_profile.get("club", ""),
                "fencer_1_place": f1_info.get("place", ""),
                "fencer_2_name": f2_info.get("name", bout["opponent_name"]),
                "fencer_2_id": fencer_2_id,
                "fencer_2_rating": f2_rating,
                "fencer_2_elo_pool_before": f2_info.get("pool_before", ""),
                "fencer_2_elo_pool_after": f2_info.get("pool_after", ""),
                "fencer_2_elo_de_before": f2_info.get("de_before", ""),
                "fencer_2_elo_de_after": f2_info.get("de_after", ""),
                "fencer_2_birth_year": f2_profile.get("birth_year", ""),
                "fencer_2_club": f2_profile.get("club", ""),
                "fencer_2_place": f2_info.get("place", ""),
                "winner": winner,
                "score": bout["score"],
                "bout_type": bout["bout_type"],
            })

    print(f"  Assembled {len(bouts)} bouts from event {event_id}")
    return bouts


def write_csv(all_bouts, path):
    """Write bout data to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for bout in all_bouts:
            writer.writerow(bout)
    print(f"\nWrote {len(all_bouts)} bouts to {path}")


def main():
    """Main entry point: scrape seed events and write CSV."""
    import argparse
    parser = argparse.ArgumentParser(description="Scrape fencingtracker.com bout data")
    parser.add_argument("--events", type=int, nargs="*", default=None,
                        help="Event IDs to scrape (default: use seed list)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of events to scrape from seed list")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="Output CSV path")
    args = parser.parse_args()

    event_ids = args.events if args.events else SEED_EVENTS
    if args.limit:
        event_ids = event_ids[:args.limit]

    print(f"Scraping {len(event_ids)} events: {event_ids}")

    all_bouts = []
    for eid in event_ids:
        bouts = scrape_event(eid)
        all_bouts.extend(bouts)

    if all_bouts:
        write_csv(all_bouts, args.output)
    else:
        print("No bouts collected.")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Events scraped: {len(event_ids)}")
    print(f"Total bouts: {len(all_bouts)}")
    pool_bouts = sum(1 for b in all_bouts if b["bout_type"] == "Pool")
    de_bouts = sum(1 for b in all_bouts if b["bout_type"] == "DE")
    print(f"  Pool bouts: {pool_bouts}")
    print(f"  DE bouts: {de_bouts}")


if __name__ == "__main__":
    main()
