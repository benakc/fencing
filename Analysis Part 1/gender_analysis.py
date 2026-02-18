#!/usr/bin/env python3
"""
Gender-focused analysis using the limited cached fencer history pages.
"""

import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CSV_PATH = os.path.join(BASE_DIR, "fencingtracker_bouts.csv")

RATING_ORDER = ["A", "B", "C", "D", "E", "U"]
RATING_RANK = {r: i for i, r in enumerate(RATING_ORDER)}


def extract_rating_letter(rating_str):
    if not rating_str or str(rating_str).strip() == "" or str(rating_str) == "nan":
        return "U"
    first = str(rating_str).strip()[0].upper()
    return first if first in RATING_RANK else "U"


def build_fencer_cache_map():
    fencer_map = {}
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".html"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        header = soup.find("div", class_="card-header")
        if not header:
            continue
        links = soup.find_all("a", href=re.compile(r"/p/\d+/"))
        for link in links:
            m = re.search(r"/p/(\d+)/", link["href"])
            if m:
                fencer_map[m.group(1)] = path
                break
    return fencer_map


def infer_gender(fencer_ids, fencer_cache_map):
    genders = {}
    for fid in fencer_ids:
        cache_path = fencer_cache_map.get(fid)
        if not cache_path or not os.path.exists(cache_path):
            genders[fid] = "Unknown"
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        card_body = soup.find("div", class_="card-body")
        if not card_body:
            genders[fid] = "Unknown"
            continue
        mc, wc = 0, 0
        for h5 in card_body.find_all("h5"):
            link = h5.find("a")
            if not link:
                continue
            ename = link.get_text(strip=True).lower()
            if "women's" in ename:
                wc += 1
            elif "men's" in ename:
                mc += 1
        if mc > 0 and wc == 0:
            genders[fid] = "M"
        elif wc > 0 and mc == 0:
            genders[fid] = "F"
        elif mc > 0 and wc > 0:
            genders[fid] = "M" if mc >= wc else "F"
        else:
            genders[fid] = "Unknown"
    return genders


def main():
    print("=" * 60)
    print("GENDER-FOCUSED ANALYSIS")
    print("=" * 60)
    print()

    # Load and prepare
    df = pd.read_csv(CSV_PATH)
    df["r1"] = df["fencer_1_rating"].astype(str).apply(extract_rating_letter)
    df["r2"] = df["fencer_2_rating"].astype(str).apply(extract_rating_letter)
    df["elo_1"] = pd.to_numeric(df["fencer_1_elo_pool_before"], errors="coerce")
    df["elo_2"] = pd.to_numeric(df["fencer_2_elo_pool_before"], errors="coerce")
    df["fencer_1_wins"] = (df["winner"] == 1).astype(int)

    # Randomize fencer positions
    np.random.seed(42)
    df["swap"] = np.random.randint(0, 2, size=len(df))

    print("Inferring gender...")
    def clean_id(x):
        if pd.isna(x):
            return ""
        return str(int(float(x)))
    df["f1_id"] = df["fencer_1_id"].apply(clean_id)
    df["f2_id"] = df["fencer_2_id"].apply(clean_id)
    all_ids = set(df["f1_id"]) | set(df["f2_id"])
    all_ids.discard("")
    cache_map = build_fencer_cache_map()
    gender_map = infer_gender(all_ids, cache_map)

    df["g1"] = df["f1_id"].map(gender_map)
    df["g2"] = df["f2_id"].map(gender_map)

    # --- Report ---
    lines = []
    lines.append("=" * 60)
    lines.append("GENDER ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # 1. Inference summary
    gc = Counter(gender_map.values())
    total = len(gender_map)
    lines.append("1. GENDER INFERENCE SUMMARY")
    lines.append("-" * 40)
    for g in ["M", "F", "Unknown"]:
        n = gc.get(g, 0)
        lines.append(f"  {g:8s}: {n:4d} fencers ({100*n/total:.1f}%)")
    lines.append(f"  Total unique fencers: {total}")
    lines.append(f"  Cached history pages: {len(cache_map)}")
    lines.append(f"  Inference rate: {100*(1 - gc.get('Unknown',0)/total):.1f}%")
    lines.append(f"  (Fencers without cache: {total - len(cache_map)})")
    lines.append(f"  (Fencers cached but only Mixed events: "
                 f"{sum(1 for fid in cache_map if gender_map.get(fid)=='Unknown')})")
    lines.append("")

    # 2. Bout-level gender matchups
    lines.append("2. BOUT-LEVEL GENDER MATCHUPS")
    lines.append("-" * 40)

    # Categorize bouts
    both_known = df[(df["g1"] != "Unknown") & (df["g2"] != "Unknown")]
    at_least_one = df[(df["g1"] != "Unknown") | (df["g2"] != "Unknown")]
    neither = df[(df["g1"] == "Unknown") & (df["g2"] == "Unknown")]

    lines.append(f"  Both genders known:     {len(both_known):4d} bouts ({100*len(both_known)/len(df):.1f}%)")
    lines.append(f"  At least one known:     {len(at_least_one):4d} bouts ({100*len(at_least_one)/len(df):.1f}%)")
    lines.append(f"  Neither known:          {len(neither):4d} bouts ({100*len(neither)/len(df):.1f}%)")
    lines.append("")

    # For bouts where at least one is known, classify
    # Create symmetric gender matchup
    def classify_bout(row):
        g1, g2 = row["g1"], row["g2"]
        if g1 == "Unknown" or g2 == "Unknown":
            return "Involves Unknown"
        return f"{g1} vs {g2}"

    df["gender_matchup"] = df.apply(classify_bout, axis=1)

    # Both-known matchups
    if len(both_known) > 0:
        bk = both_known.copy()
        bk["gm"] = bk["g1"] + " vs " + bk["g2"]
        gm_counts = bk["gm"].value_counts()
        lines.append("  Matchups (both known):")
        for gm, n in gm_counts.items():
            lines.append(f"    {gm:12s}: {n:4d}")
        lines.append("")

        # Symmetric view: who wins in cross-gender bouts?
        lines.append("3. CROSS-GENDER WIN RATES (both genders known)")
        lines.append("-" * 40)

        # Symmetrize: from each fencer's perspective
        records = []
        for _, row in bk.iterrows():
            f1_wins = row["fencer_1_wins"]
            records.append({"my_gender": row["g1"], "opp_gender": row["g2"], "win": f1_wins})
            records.append({"my_gender": row["g2"], "opp_gender": row["g1"], "win": 1 - f1_wins})

        rdf = pd.DataFrame(records)
        for my_g in ["M", "F"]:
            for opp_g in ["M", "F"]:
                subset = rdf[(rdf["my_gender"] == my_g) & (rdf["opp_gender"] == opp_g)]
                if len(subset) > 0:
                    wr = subset["win"].mean() * 100
                    lines.append(f"  {my_g} vs {opp_g}: {wr:.1f}% win rate (N={len(subset)})")
        lines.append("")

        # 4. Cross-gender by weapon
        lines.append("4. CROSS-GENDER WIN RATES BY WEAPON")
        lines.append("-" * 40)
        for weapon in sorted(bk["weapon"].unique()):
            wdf = bk[bk["weapon"] == weapon]
            records_w = []
            for _, row in wdf.iterrows():
                f1_wins = row["fencer_1_wins"]
                records_w.append({"my_gender": row["g1"], "opp_gender": row["g2"], "win": f1_wins})
                records_w.append({"my_gender": row["g2"], "opp_gender": row["g1"], "win": 1 - f1_wins})
            wrdf = pd.DataFrame(records_w)
            lines.append(f"  {weapon.upper()}:")
            for my_g in ["M", "F"]:
                for opp_g in ["M", "F"]:
                    subset = wrdf[(wrdf["my_gender"] == my_g) & (wrdf["opp_gender"] == opp_g)]
                    if len(subset) > 0:
                        wr = subset["win"].mean() * 100
                        flag = " *" if len(subset) < 20 else ""
                        lines.append(f"    {my_g} vs {opp_g}: {wr:.1f}% (N={len(subset)}){flag}")
            lines.append("")

        # 5. Cross-gender by rating
        lines.append("5. M vs F WIN RATES BY RATING MATCHUP")
        lines.append("-" * 40)
        lines.append("   (Only showing matchups with N >= 3)")

        cross_gender = bk[((bk["g1"] == "M") & (bk["g2"] == "F")) |
                          ((bk["g1"] == "F") & (bk["g2"] == "M"))].copy()

        if len(cross_gender) > 0:
            # Normalize: put M as fencer_left, F as fencer_right
            cg_records = []
            for _, row in cross_gender.iterrows():
                if row["g1"] == "M":
                    m_rating = row["r1"]
                    f_rating = row["r2"]
                    m_wins = row["fencer_1_wins"]
                    m_elo = row["elo_1"]
                    f_elo = row["elo_2"]
                else:
                    m_rating = row["r2"]
                    f_rating = row["r1"]
                    m_wins = 1 - row["fencer_1_wins"]
                    m_elo = row["elo_2"]
                    f_elo = row["elo_1"]
                cg_records.append({
                    "m_rating": m_rating, "f_rating": f_rating,
                    "m_wins": m_wins, "m_elo": m_elo, "f_elo": f_elo,
                    "elo_diff": m_elo - f_elo if pd.notna(m_elo) and pd.notna(f_elo) else np.nan,
                    "weapon": row["weapon"], "bout_type": row["bout_type"],
                })
            cgdf = pd.DataFrame(cg_records)

            lines.append(f"  Total M vs F bouts: {len(cgdf)}")
            lines.append(f"  Overall M win rate: {cgdf['m_wins'].mean()*100:.1f}%")
            lines.append("")

            # By M rating vs F rating
            lines.append("  M rating vs F rating -> M win%:")
            for mr in RATING_ORDER:
                for fr in RATING_ORDER:
                    subset = cgdf[(cgdf["m_rating"] == mr) & (cgdf["f_rating"] == fr)]
                    if len(subset) >= 3:
                        wr = subset["m_wins"].mean() * 100
                        flag = " *" if len(subset) < 10 else ""
                        lines.append(f"    M({mr}) vs F({fr}): {wr:.0f}% M wins (N={len(subset)}){flag}")
            lines.append("")

            # ELO analysis for cross-gender
            cg_with_elo = cgdf.dropna(subset=["elo_diff"])
            if len(cg_with_elo) > 0:
                lines.append("  ELO context for M vs F bouts:")
                lines.append(f"    Mean M ELO: {cg_with_elo['m_elo'].mean():.0f}")
                lines.append(f"    Mean F ELO: {cg_with_elo['f_elo'].mean():.0f}")
                lines.append(f"    Mean ELO diff (M-F): {cg_with_elo['elo_diff'].mean():.0f}")
                lines.append(f"    Median ELO diff (M-F): {cg_with_elo['elo_diff'].median():.0f}")
                lines.append("")

                # Control for ELO: win rate when M and F have similar ELO
                close_elo = cg_with_elo[cg_with_elo["elo_diff"].abs() <= 200]
                if len(close_elo) >= 5:
                    lines.append(f"    When ELO within +/-200 (N={len(close_elo)}):")
                    lines.append(f"      M win rate: {close_elo['m_wins'].mean()*100:.1f}%")
                    lines.append(f"      Mean ELO diff: {close_elo['elo_diff'].mean():.0f}")

                close_elo_500 = cg_with_elo[cg_with_elo["elo_diff"].abs() <= 500]
                if len(close_elo_500) >= 5:
                    lines.append(f"    When ELO within +/-500 (N={len(close_elo_500)}):")
                    lines.append(f"      M win rate: {close_elo_500['m_wins'].mean()*100:.1f}%")
                    lines.append(f"      Mean ELO diff: {close_elo_500['elo_diff'].mean():.0f}")
                lines.append("")

            # By weapon
            lines.append("  M vs F by weapon:")
            for w in sorted(cgdf["weapon"].unique()):
                ws = cgdf[cgdf["weapon"] == w]
                lines.append(f"    {w:8s}: M wins {ws['m_wins'].mean()*100:.1f}% (N={len(ws)})")
            lines.append("")

            # By bout type
            lines.append("  M vs F by bout type:")
            for bt in sorted(cgdf["bout_type"].unique()):
                bs = cgdf[cgdf["bout_type"] == bt]
                lines.append(f"    {bt:8s}: M wins {bs['m_wins'].mean()*100:.1f}% (N={len(bs)})")
            lines.append("")

        # 6. Same-gender comparisons
        lines.append("6. SAME-GENDER COMPARISONS")
        lines.append("-" * 40)

        for g in ["M", "F"]:
            same = bk[(bk["g1"] == g) & (bk["g2"] == g)]
            if len(same) > 0:
                lines.append(f"  {g} vs {g} (N={len(same)}):")
                # Rating distribution
                all_r = list(same["r1"]) + list(same["r2"])
                rc = Counter(all_r)
                lines.append(f"    Rating dist: {', '.join(f'{r}={rc.get(r,0)}' for r in RATING_ORDER)}")
                # ELO stats
                all_elo = pd.concat([same["elo_1"], same["elo_2"]]).dropna()
                if len(all_elo) > 0:
                    lines.append(f"    ELO: mean={all_elo.mean():.0f}, median={all_elo.median():.0f}, "
                                 f"range={all_elo.min():.0f}-{all_elo.max():.0f}")
                lines.append("")

    # 7. Limitations
    lines.append("7. LIMITATIONS & CAVEATS")
    lines.append("-" * 40)
    lines.append("  - Gender inferred from event names (Men's/Women's) in fencer history pages")
    lines.append(f"  - Only {len(cache_map)}/{total} fencers have cached history pages")
    lines.append(f"  - Of cached fencers, {sum(1 for fid in cache_map if gender_map.get(fid)=='Unknown')} "
                 "competed only in Mixed events (gender unknown)")
    lines.append(f"  - Overall inference rate: {100*(1 - gc.get('Unknown',0)/total):.1f}%")
    lines.append("  - Fencers with cached pages are biased toward those who appeared as fencer_1")
    lines.append("    (the reporting fencer in the scraper), who tend to be more active competitors")
    lines.append("  - Small sample sizes in many cells — interpret with caution")
    lines.append("  - * marks cells with N < 10 or N < 20 as noted")
    lines.append("")

    # Write report
    report_path = os.path.join(BASE_DIR, "gender_analysis.txt")
    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
