#!/usr/bin/env python3
"""
FencingTracker Analysis — Summary statistics, cross-tabs, logistic regression,
and visualizations for Senior Mixed fencing bout data.
"""

import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup

# statsmodels import
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CSV_PATH = os.path.join(BASE_DIR, "fencingtracker_bouts.csv")

RATING_ORDER = ["A", "B", "C", "D", "E", "U"]
RATING_RANK = {r: i for i, r in enumerate(RATING_ORDER)}


# ── Step 1: Gender Inference ─────────────────────────────────────────────────

def _build_fencer_cache_map():
    """Scan cache files and map fencer_id -> cache file path for history pages."""
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
        # Find fencer ID from links in the page
        links = soup.find_all("a", href=re.compile(r"/p/\d+/"))
        for link in links:
            m = re.search(r"/p/(\d+)/", link["href"])
            if m:
                fencer_map[m.group(1)] = path
                break
    return fencer_map


def infer_gender(fencer_ids, fencer_cache_map):
    """Infer gender from fencer history pages.

    Looks at event names for "Men's" or "Women's" keywords.
    Returns dict: fencer_id -> "M" | "F" | "Unknown"
    """
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

        mens_count = 0
        womens_count = 0
        for h5 in card_body.find_all("h5"):
            link = h5.find("a")
            if not link:
                continue
            ename = link.get_text(strip=True).lower()
            if "women's" in ename:
                womens_count += 1
            elif "men's" in ename:
                mens_count += 1

        if mens_count > 0 and womens_count == 0:
            genders[fid] = "M"
        elif womens_count > 0 and mens_count == 0:
            genders[fid] = "F"
        elif mens_count > 0 and womens_count > 0:
            genders[fid] = "M" if mens_count >= womens_count else "F"
        else:
            genders[fid] = "Unknown"

    return genders


# ── Step 2: Data Preparation ─────────────────────────────────────────────────

def extract_rating_letter(rating_str):
    """Extract letter-only rating: 'A25' -> 'A', 'U' -> 'U', '' -> 'U'."""
    if not rating_str or rating_str.strip() == "":
        return "U"
    first = rating_str.strip()[0].upper()
    if first in RATING_RANK:
        return first
    return "U"


def parse_score(score_str):
    """Parse score string '5:3' -> (5, 3). Returns (None, None) on failure."""
    parts = score_str.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def make_rating_matchup(r1, r2):
    """Create sorted rating matchup string, e.g. 'A vs C'."""
    pair = sorted([r1, r2], key=lambda x: RATING_RANK.get(x, 99))
    return f"{pair[0]} vs {pair[1]}"


def load_and_prepare_data():
    """Load CSV and add computed columns. Returns DataFrame."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} bouts")

    # Extract rating letters
    df["r1"] = df["fencer_1_rating"].astype(str).apply(extract_rating_letter)
    df["r2"] = df["fencer_2_rating"].astype(str).apply(extract_rating_letter)

    # Parse scores
    scores = df["score"].apply(lambda s: pd.Series(parse_score(str(s)), index=["score_1", "score_2"]))
    df["score_1"] = scores["score_1"]
    df["score_2"] = scores["score_2"]
    df["margin"] = df["score_1"] - df["score_2"]

    # ELO diff (pool ELO before)
    df["elo_1"] = pd.to_numeric(df["fencer_1_elo_pool_before"], errors="coerce")
    df["elo_2"] = pd.to_numeric(df["fencer_2_elo_pool_before"], errors="coerce")
    df["elo_diff"] = df["elo_1"] - df["elo_2"]

    # Rating matchup
    df["rating_matchup"] = df.apply(lambda row: make_rating_matchup(row["r1"], row["r2"]), axis=1)

    # fencer_1_wins (raw — note: fencer_1 is biased toward being the winner at ~90%)
    df["fencer_1_wins"] = (df["winner"] == 1).astype(int)

    # Create a randomized version to remove fencer_1 selection bias.
    # Randomly swap fencer 1 and 2 for each bout.
    np.random.seed(42)
    df["swap"] = np.random.randint(0, 2, size=len(df))
    # After swap: "left" fencer and "right" fencer
    df["left_elo"] = np.where(df["swap"] == 1, df["elo_2"], df["elo_1"])
    df["right_elo"] = np.where(df["swap"] == 1, df["elo_1"], df["elo_2"])
    df["left_rating"] = np.where(df["swap"] == 1, df["r2"], df["r1"])
    df["right_rating"] = np.where(df["swap"] == 1, df["r1"], df["r2"])
    df["left_wins"] = np.where(df["swap"] == 1, 1 - df["fencer_1_wins"], df["fencer_1_wins"])
    df["elo_diff_rand"] = df["left_elo"] - df["right_elo"]

    # Gender inference
    print("Inferring gender from cached history pages...")
    def clean_id(x):
        if pd.isna(x):
            return ""
        return str(int(float(x)))
    df["fencer_1_id_str"] = df["fencer_1_id"].apply(clean_id)
    df["fencer_2_id_str"] = df["fencer_2_id"].apply(clean_id)
    all_fencer_ids = set(df["fencer_1_id_str"].unique()) | set(df["fencer_2_id_str"].unique())
    all_fencer_ids.discard("")
    fencer_cache_map = _build_fencer_cache_map()
    print(f"  Found {len(fencer_cache_map)} cached history pages for {len(all_fencer_ids)} unique fencers")
    gender_map = infer_gender(all_fencer_ids, fencer_cache_map)

    df["fencer_1_gender"] = df["fencer_1_id_str"].map(gender_map)
    df["fencer_2_gender"] = df["fencer_2_id_str"].map(gender_map)
    df["gender_matchup"] = df["fencer_1_gender"] + " vs " + df["fencer_2_gender"]

    # Rating gap (numeric distance between ratings)
    df["rating_gap"] = abs(df["r1"].map(RATING_RANK).fillna(5) - df["r2"].map(RATING_RANK).fillna(5)).astype(int)

    print(f"  Gender inference: M={sum(1 for v in gender_map.values() if v=='M')}, "
          f"F={sum(1 for v in gender_map.values() if v=='F')}, "
          f"Unknown={sum(1 for v in gender_map.values() if v=='Unknown')}")

    return df, gender_map


# ── Step 3: Summary Statistics ────────────────────────────────────────────────

def write_summary(df, gender_map, out_path):
    """Write analysis_summary.txt with all stats."""
    lines = []
    lines.append("=" * 70)
    lines.append("FENCINGTRACKER ANALYSIS — SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Basic counts
    lines.append("1. DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total bouts: {len(df)}")
    lines.append(f"Unique events: {df['event_id'].nunique()}")
    lines.append(f"Unique fencers: {len(set(df['fencer_1_id'].astype(str)) | set(df['fencer_2_id'].astype(str)))}")
    lines.append("")

    f1_win_rate = df["fencer_1_wins"].mean()
    lines.append(f"Fencer 1 win rate: {f1_win_rate:.1%} (selection bias — fencer 1 is the reporting fencer)")
    lines.append("NOTE: ELO and logistic analyses use randomized fencer assignment to correct this bias.")
    lines.append("")

    # By weapon
    lines.append("Bouts by weapon:")
    for w in sorted(df["weapon"].unique()):
        n = len(df[df["weapon"] == w])
        lines.append(f"  {w:8s}: {n:4d}")
    lines.append("")

    # By bout type
    lines.append("Bouts by type:")
    for bt in sorted(df["bout_type"].unique()):
        n = len(df[df["bout_type"] == bt])
        lines.append(f"  {bt:8s}: {n:4d}")
    lines.append("")

    # By weapon x type
    lines.append("Bouts by weapon x type:")
    ct = pd.crosstab(df["weapon"], df["bout_type"])
    lines.append(ct.to_string())
    lines.append("")

    # Rating distribution
    lines.append("2. RATING DISTRIBUTION")
    lines.append("-" * 40)
    all_ratings = list(df["r1"]) + list(df["r2"])
    rating_counts = Counter(all_ratings)
    for r in RATING_ORDER:
        lines.append(f"  {r}: {rating_counts.get(r, 0)}")
    lines.append("")

    # ELO distribution
    lines.append("3. ELO DISTRIBUTION")
    lines.append("-" * 40)
    all_elo = pd.concat([df["elo_1"], df["elo_2"]]).dropna()
    lines.append(f"  Count: {len(all_elo)}")
    lines.append(f"  Mean:  {all_elo.mean():.0f}")
    lines.append(f"  Std:   {all_elo.std():.0f}")
    lines.append(f"  Min:   {all_elo.min():.0f}")
    lines.append(f"  25%:   {all_elo.quantile(0.25):.0f}")
    lines.append(f"  50%:   {all_elo.quantile(0.50):.0f}")
    lines.append(f"  75%:   {all_elo.quantile(0.75):.0f}")
    lines.append(f"  Max:   {all_elo.max():.0f}")
    lines.append("")

    # Gender inference
    lines.append("4. GENDER INFERENCE")
    lines.append("-" * 40)
    g_counts = Counter(gender_map.values())
    total_fencers = len(gender_map)
    for g in ["M", "F", "Unknown"]:
        n = g_counts.get(g, 0)
        pct = 100 * n / total_fencers if total_fencers > 0 else 0
        lines.append(f"  {g:8s}: {n:4d} ({pct:.1f}%)")
    lines.append(f"  Inference rate: {100 * (1 - g_counts.get('Unknown', 0) / total_fencers):.1f}%")
    lines.append("")

    lines.append("Gender matchup counts (bout-level):")
    gm_counts = df["gender_matchup"].value_counts()
    for gm, n in gm_counts.items():
        lines.append(f"  {gm:16s}: {n:4d}")
    lines.append("")

    # Rating matchup sample sizes
    lines.append("5. RATING MATCHUP SAMPLE SIZES")
    lines.append("-" * 40)
    rm_counts = df["rating_matchup"].value_counts().sort_index()
    for rm, n in rm_counts.items():
        flag = " *** SMALL SAMPLE" if n < 20 else ""
        lines.append(f"  {rm:10s}: {n:4d}{flag}")
    lines.append("")

    # DE score margins
    de = df[df["bout_type"] == "DE"].copy()
    if len(de) > 0:
        lines.append("6. DE SCORE MARGINS")
        lines.append("-" * 40)
        # For DE, margin from winner's perspective
        de_margins = de.apply(
            lambda row: row["score_1"] - row["score_2"] if row["winner"] == 1 else row["score_2"] - row["score_1"],
            axis=1
        ).dropna()
        lines.append(f"  Count: {len(de_margins)}")
        lines.append(f"  Mean:  {de_margins.mean():.1f}")
        lines.append(f"  Std:   {de_margins.std():.1f}")
        lines.append(f"  Min:   {de_margins.min():.0f}")
        lines.append(f"  Median:{de_margins.median():.0f}")
        lines.append(f"  Max:   {de_margins.max():.0f}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote summary to {out_path}")


# ── Step 4: Rating Cross-Tabs ────────────────────────────────────────────────

def compute_rating_crosstab(df, label="all"):
    """Compute win% cross-tab where row=fencer rating, col=opponent rating.

    We create a symmetric view: for each bout, count from both perspectives.
    """
    records = []
    for _, row in df.iterrows():
        r1, r2 = row["r1"], row["r2"]
        f1_wins = row["fencer_1_wins"]
        # Perspective of fencer 1
        records.append({"my_rating": r1, "opp_rating": r2, "win": f1_wins})
        # Perspective of fencer 2
        records.append({"my_rating": r2, "opp_rating": r1, "win": 1 - f1_wins})

    rdf = pd.DataFrame(records)
    # Build the cross-tab
    rows = []
    for my_r in RATING_ORDER:
        row_data = {"rating": my_r}
        for opp_r in RATING_ORDER:
            subset = rdf[(rdf["my_rating"] == my_r) & (rdf["opp_rating"] == opp_r)]
            n = len(subset)
            if n > 0:
                win_pct = 100 * subset["win"].mean()
                flag = "*" if n < 20 else ""
                row_data[opp_r] = f"{win_pct:.0f}% ({n}){flag}"
            else:
                row_data[opp_r] = "-"
        rows.append(row_data)
    return pd.DataFrame(rows).set_index("rating")


def write_rating_crosstabs(df, out_path, summary_lines=None):
    """Write rating cross-tabs to CSV and append to summary."""
    # Combined
    ct_all = compute_rating_crosstab(df, "all")

    # Per weapon
    weapon_cts = {}
    for w in sorted(df["weapon"].unique()):
        wdf = df[df["weapon"] == w]
        if len(wdf) >= 20:
            weapon_cts[w] = compute_rating_crosstab(wdf, w)

    # Write to CSV
    with open(out_path, "w") as f:
        f.write("Rating Win% Cross-Tab (row=my rating, col=opponent rating)\n")
        f.write("Cell = win% (N), * = N < 20\n\n")
        f.write("ALL WEAPONS\n")
        f.write(ct_all.to_csv())
        f.write("\n")
        for w, ct in weapon_cts.items():
            f.write(f"\n{w.upper()}\n")
            f.write(ct.to_csv())
            f.write("\n")

    print(f"  Wrote rating cross-tabs to {out_path}")

    # Also append text version to summary
    if summary_lines is not None:
        summary_lines.append("\n7. RATING WIN% CROSS-TAB (all weapons)")
        summary_lines.append("-" * 40)
        summary_lines.append("Row = my rating, Col = opponent rating")
        summary_lines.append("Cell = win% (N), * = N < 20")
        summary_lines.append(ct_all.to_string())
        summary_lines.append("")
        for w, ct in weapon_cts.items():
            summary_lines.append(f"\n  {w.upper()}:")
            summary_lines.append(ct.to_string())
            summary_lines.append("")

    return ct_all, weapon_cts


# ── Step 5: ELO Analysis ─────────────────────────────────────────────────────

def compute_elo_crosstab(df, bin_width=200):
    """Bin ELO into ranges and compute win rate by bin matchup."""
    edf = df.dropna(subset=["elo_1", "elo_2"]).copy()
    if len(edf) == 0:
        return pd.DataFrame()

    elo_min = min(edf["elo_1"].min(), edf["elo_2"].min())
    elo_max = max(edf["elo_1"].max(), edf["elo_2"].max())
    bins = list(range(int(elo_min // bin_width * bin_width),
                      int(elo_max // bin_width * bin_width) + bin_width + 1, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]

    edf["elo_bin_1"] = pd.cut(edf["elo_1"], bins=bins, labels=labels, include_lowest=True)
    edf["elo_bin_2"] = pd.cut(edf["elo_2"], bins=bins, labels=labels, include_lowest=True)

    # Build symmetric records
    records = []
    for _, row in edf.iterrows():
        records.append({"my_bin": str(row["elo_bin_1"]), "opp_bin": str(row["elo_bin_2"]), "win": row["fencer_1_wins"]})
        records.append({"my_bin": str(row["elo_bin_2"]), "opp_bin": str(row["elo_bin_1"]), "win": 1 - row["fencer_1_wins"]})

    rdf = pd.DataFrame(records)
    # Pivot
    used_labels = sorted(rdf["my_bin"].unique())
    rows = []
    for my_b in used_labels:
        row_data = {"elo_bin": my_b}
        for opp_b in used_labels:
            subset = rdf[(rdf["my_bin"] == my_b) & (rdf["opp_bin"] == opp_b)]
            n = len(subset)
            if n > 0:
                win_pct = 100 * subset["win"].mean()
                flag = "*" if n < 20 else ""
                row_data[opp_b] = f"{win_pct:.0f}% ({n}){flag}"
            else:
                row_data[opp_b] = "-"
        rows.append(row_data)
    return pd.DataFrame(rows).set_index("elo_bin")


def compute_elo_win_rate_curve(df, bin_width=100):
    """Win rate as a function of ELO difference (using randomized assignment)."""
    edf = df.dropna(subset=["elo_diff_rand"]).copy()
    edf["elo_diff_bin"] = (edf["elo_diff_rand"] / bin_width).round() * bin_width
    grouped = edf.groupby("elo_diff_bin").agg(
        win_rate=("left_wins", "mean"),
        count=("left_wins", "count"),
    ).reset_index()
    return grouped


# ── Step 6: Logistic Regression ──────────────────────────────────────────────

def run_logistic_regression(df, summary_path):
    """P(left_wins) ~ elo_diff + weapon, using randomized fencer assignment."""
    rdf = df.dropna(subset=["elo_diff_rand"]).copy()
    rdf = rdf[rdf["elo_diff_rand"].abs() < 5000]  # remove extreme outliers

    # Encode weapon (reference: epee)
    rdf["weapon_foil"] = (rdf["weapon"] == "foil").astype(int)
    rdf["weapon_saber"] = (rdf["weapon"] == "saber").astype(int)

    X_cols = ["elo_diff_rand", "weapon_foil", "weapon_saber"]
    model_desc = ("P(left_wins) ~ elo_diff + weapon\n"
                  "(Fencer positions randomized to remove fencer_1 selection bias; "
                  f"raw fencer_1 win rate was {df['fencer_1_wins'].mean():.1%})")

    X = rdf[X_cols].astype(float)
    X = sm.add_constant(X)
    y = rdf["left_wins"].astype(float)

    try:
        model = Logit(y, X)
        result = model.fit(disp=0)
        summary_text = str(result.summary())
    except Exception as e:
        summary_text = f"Logistic regression failed: {e}"
        result = None

    # Predictions for common scenarios
    predictions = []
    if result is not None:
        scenarios = [
            ("200 ELO advantage, epee", [1, 200, 0, 0]),
            ("200 ELO advantage, foil", [1, 200, 1, 0]),
            ("200 ELO advantage, saber", [1, 200, 0, 1]),
            ("500 ELO advantage, epee", [1, 500, 0, 0]),
            ("1000 ELO advantage, epee", [1, 1000, 0, 0]),
            ("0 ELO diff, epee", [1, 0, 0, 0]),
            ("-200 ELO disadvantage, epee", [1, -200, 0, 0]),
            ("-500 ELO disadvantage, epee", [1, -500, 0, 0]),
        ]
        for desc, x_vals in scenarios:
            prob = result.predict(np.array([x_vals]))[0]
            predictions.append(f"  {desc}: P(win) = {prob:.3f}")

    # Append to summary
    with open(summary_path, "a") as f:
        f.write("\n\n8. LOGISTIC REGRESSION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {model_desc}\n\n")
        f.write(summary_text + "\n")
        if predictions:
            f.write("\nPredicted win probabilities:\n")
            f.write("\n".join(predictions) + "\n")

    print("  Logistic regression complete")
    return result


# ── Step 7: Visualizations ───────────────────────────────────────────────────

def plot_rating_heatmap(df, out_path):
    """Win% heatmap by rating pair, all weapons."""
    records = []
    for _, row in df.iterrows():
        records.append({"my_rating": row["r1"], "opp_rating": row["r2"], "win": row["fencer_1_wins"]})
        records.append({"my_rating": row["r2"], "opp_rating": row["r1"], "win": 1 - row["fencer_1_wins"]})
    rdf = pd.DataFrame(records)

    pivot = rdf.groupby(["my_rating", "opp_rating"])["win"].mean().unstack()
    pivot = pivot.reindex(index=RATING_ORDER, columns=RATING_ORDER) * 100

    counts = rdf.groupby(["my_rating", "opp_rating"])["win"].count().unstack()
    counts = counts.reindex(index=RATING_ORDER, columns=RATING_ORDER)

    # Annotation: show win% and N
    annot = pivot.copy().astype(str)
    for r in RATING_ORDER:
        for c in RATING_ORDER:
            v = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
            n = counts.loc[r, c] if r in counts.index and c in counts.columns else 0
            if pd.isna(v) or pd.isna(n) or n == 0:
                annot.loc[r, c] = "-"
            else:
                flag = "*" if n < 20 else ""
                annot.loc[r, c] = f"{v:.0f}%\n({int(n)}){flag}"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Win %"})
    ax.set_title("Win % by Rating Matchup (All Weapons)\n* = N < 20", fontsize=14)
    ax.set_xlabel("Opponent Rating")
    ax.set_ylabel("My Rating")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_rating_heatmap_by_weapon(df, out_path):
    """One heatmap subplot per weapon."""
    weapons = sorted(df["weapon"].unique())
    fig, axes = plt.subplots(1, len(weapons), figsize=(8 * len(weapons), 7))
    if len(weapons) == 1:
        axes = [axes]

    for ax, w in zip(axes, weapons):
        wdf = df[df["weapon"] == w]
        records = []
        for _, row in wdf.iterrows():
            records.append({"my_rating": row["r1"], "opp_rating": row["r2"], "win": row["fencer_1_wins"]})
            records.append({"my_rating": row["r2"], "opp_rating": row["r1"], "win": 1 - row["fencer_1_wins"]})
        rdf = pd.DataFrame(records)

        pivot = rdf.groupby(["my_rating", "opp_rating"])["win"].mean().unstack()
        pivot = pivot.reindex(index=RATING_ORDER, columns=RATING_ORDER) * 100

        counts = rdf.groupby(["my_rating", "opp_rating"])["win"].count().unstack()
        counts = counts.reindex(index=RATING_ORDER, columns=RATING_ORDER)

        annot = pivot.copy().astype(str)
        for r in RATING_ORDER:
            for c in RATING_ORDER:
                v = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
                n = counts.loc[r, c] if r in counts.index and c in counts.columns else 0
                if pd.isna(v) or pd.isna(n) or n == 0:
                    annot.loc[r, c] = "-"
                else:
                    flag = "*" if n < 20 else ""
                    annot.loc[r, c] = f"{v:.0f}%\n({int(n)}){flag}"

        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Win %"})
        ax.set_title(f"{w.upper()} (N={len(wdf)})", fontsize=13)
        ax.set_xlabel("Opponent Rating")
        ax.set_ylabel("My Rating")

    plt.suptitle("Win % by Rating Matchup — By Weapon\n* = N < 20", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_elo_win_curve(df, logit_result, out_path):
    """P(win) vs ELO difference with logistic curve overlay."""
    curve = compute_elo_win_rate_curve(df, bin_width=100)
    curve = curve[curve["count"] >= 5]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(curve["elo_diff_bin"], curve["win_rate"], s=curve["count"] * 2,
               alpha=0.7, color="steelblue", label="Observed (size = N)")

    # Logistic curve
    if logit_result is not None:
        x_range = np.linspace(-2000, 2000, 200)
        n_params = len(logit_result.params)
        # Build prediction matrix: constant + elo_diff + zeros for other vars (reference: epee)
        X_pred = np.zeros((len(x_range), n_params))
        X_pred[:, 0] = 1  # constant
        X_pred[:, 1] = x_range  # elo_diff
        y_pred = logit_result.predict(X_pred)
        ax.plot(x_range, y_pred, "r-", linewidth=2, label="Logistic fit (epee)")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ELO Difference (higher - lower)", fontsize=12)
    ax.set_ylabel("P(win)", fontsize=12)
    ax.set_title("Win Probability vs ELO Difference", fontsize=14)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_elo_win_curve_by_weapon(df, out_path):
    """ELO win curve split by weapon."""
    weapons = sorted(df["weapon"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"epee": "steelblue", "foil": "green", "saber": "darkorange"}

    for w in weapons:
        wdf = df[df["weapon"] == w]
        curve = compute_elo_win_rate_curve(wdf, bin_width=150)
        curve = curve[curve["count"] >= 5]
        ax.scatter(curve["elo_diff_bin"], curve["win_rate"], s=curve["count"],
                   alpha=0.6, color=colors.get(w, "gray"), label=f"{w} (N={len(wdf)})")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ELO Difference", fontsize=12)
    ax.set_ylabel("P(win)", fontsize=12)
    ax.set_title("Win Probability vs ELO Difference — By Weapon", fontsize=14)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_rating_distribution(df, out_path):
    """Bar chart of rating frequencies."""
    all_ratings = list(df["r1"]) + list(df["r2"])
    rating_counts = Counter(all_ratings)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = RATING_ORDER
    y = [rating_counts.get(r, 0) for r in x]
    bars = ax.bar(x, y, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Appearances (fencer-bouts)", fontsize=12)
    ax.set_title("Rating Distribution Across All Bouts", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_elo_distribution(df, out_path):
    """Histogram of ELO values."""
    all_elo = pd.concat([df["elo_1"], df["elo_2"]]).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_elo, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(all_elo.mean(), color="red", linestyle="--", label=f"Mean: {all_elo.mean():.0f}")
    ax.axvline(all_elo.median(), color="orange", linestyle="--", label=f"Median: {all_elo.median():.0f}")
    ax.set_xlabel("Pool ELO (before)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("ELO Distribution Across All Bouts", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_upset_analysis(df, out_path):
    """Upset rate by rating gap + pool vs DE comparison."""
    # Determine who is the "higher-rated" fencer and whether they won
    records = []
    for _, row in df.iterrows():
        r1_rank = RATING_RANK.get(row["r1"], 5)
        r2_rank = RATING_RANK.get(row["r2"], 5)
        if r1_rank == r2_rank:
            continue  # same rating, no "upset" possible
        if r1_rank < r2_rank:
            # fencer 1 is higher-rated
            higher_won = row["fencer_1_wins"]
        else:
            higher_won = 1 - row["fencer_1_wins"]
        gap = abs(r1_rank - r2_rank)
        records.append({"rating_gap": gap, "higher_won": higher_won,
                        "upset": 1 - higher_won, "bout_type": row["bout_type"]})

    if not records:
        return

    udf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: upset rate by rating gap
    ax = axes[0]
    gap_stats = udf.groupby("rating_gap").agg(
        upset_rate=("upset", "mean"),
        count=("upset", "count"),
    ).reset_index()
    bars = ax.bar(gap_stats["rating_gap"], gap_stats["upset_rate"] * 100,
                  color="steelblue", edgecolor="white")
    for bar, row_data in zip(bars, gap_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"N={row_data.count}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Rating Gap (letters apart)", fontsize=12)
    ax.set_ylabel("Upset Rate (%)", fontsize=12)
    ax.set_title("Upset Rate by Rating Gap", fontsize=13)
    ax.set_xticks(sorted(udf["rating_gap"].unique()))

    # Right: pool vs DE upset rate by rating gap
    ax = axes[1]
    pivot = udf.groupby(["rating_gap", "bout_type"])["upset"].mean().unstack() * 100
    pivot.plot(kind="bar", ax=ax, color=["steelblue", "darkorange"], edgecolor="white")
    ax.set_xlabel("Rating Gap (letters apart)", fontsize=12)
    ax.set_ylabel("Upset Rate (%)", fontsize=12)
    ax.set_title("Upset Rate: Pool vs DE", fontsize=13)
    ax.legend(title="Bout Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.suptitle("Upset Analysis (lower-rated fencer wins)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Step 8: Upset & Additional Analyses ──────────────────────────────────────

def write_upset_analysis(df, summary_path):
    """Append upset and prediction accuracy analysis to summary."""
    lines = []
    lines.append("\n9. UPSET ANALYSIS")
    lines.append("-" * 40)

    # Upset rate by rating gap
    records = []
    for _, row in df.iterrows():
        r1_rank = RATING_RANK.get(row["r1"], 5)
        r2_rank = RATING_RANK.get(row["r2"], 5)
        if r1_rank == r2_rank:
            continue
        if r1_rank < r2_rank:
            higher_won = row["fencer_1_wins"]
        else:
            higher_won = 1 - row["fencer_1_wins"]
        gap = abs(r1_rank - r2_rank)
        records.append({"rating_gap": gap, "higher_won": higher_won,
                        "upset": 1 - higher_won, "bout_type": row["bout_type"]})

    if records:
        udf = pd.DataFrame(records)
        lines.append(f"Bouts with different ratings: {len(udf)}")
        lines.append("")

        lines.append("Upset rate by rating gap:")
        for gap in sorted(udf["rating_gap"].unique()):
            subset = udf[udf["rating_gap"] == gap]
            rate = subset["upset"].mean() * 100
            lines.append(f"  {gap}-letter gap: {rate:.1f}% upset rate (N={len(subset)})")
        lines.append("")

        lines.append("Pool vs DE upset rates:")
        for bt in ["DE", "Pool"]:
            subset = udf[udf["bout_type"] == bt]
            if len(subset) > 0:
                rate = subset["upset"].mean() * 100
                lines.append(f"  {bt:8s}: {rate:.1f}% upset rate (N={len(subset)})")
        lines.append("")

        lines.append("Pool vs DE upset rates by gap:")
        for gap in sorted(udf["rating_gap"].unique()):
            gap_df = udf[udf["rating_gap"] == gap]
            parts = []
            for bt in ["Pool", "DE"]:
                subset = gap_df[gap_df["bout_type"] == bt]
                if len(subset) > 0:
                    rate = subset["upset"].mean() * 100
                    parts.append(f"{bt}={rate:.1f}% (N={len(subset)})")
            lines.append(f"  {gap}-letter gap: {', '.join(parts)}")
        lines.append("")

    # Rating vs ELO prediction accuracy
    lines.append("\n10. RATING vs ELO PREDICTION ACCURACY")
    lines.append("-" * 40)

    valid = df.dropna(subset=["elo_diff"]).copy()

    # Rating-based prediction: higher-rated fencer wins
    rating_correct = 0
    rating_total = 0
    for _, row in valid.iterrows():
        r1_rank = RATING_RANK.get(row["r1"], 5)
        r2_rank = RATING_RANK.get(row["r2"], 5)
        if r1_rank == r2_rank:
            continue
        rating_total += 1
        if r1_rank < r2_rank:
            predicted_1_wins = True
        else:
            predicted_1_wins = False
        if predicted_1_wins == bool(row["fencer_1_wins"]):
            rating_correct += 1

    # ELO-based prediction: higher-ELO fencer wins
    elo_correct = 0
    elo_total = 0
    for _, row in valid.iterrows():
        if row["elo_diff"] == 0:
            continue
        elo_total += 1
        predicted_1_wins = row["elo_diff"] > 0
        if predicted_1_wins == bool(row["fencer_1_wins"]):
            elo_correct += 1

    if rating_total > 0:
        lines.append(f"Rating-based accuracy: {100 * rating_correct / rating_total:.1f}% ({rating_correct}/{rating_total})")
    if elo_total > 0:
        lines.append(f"ELO-based accuracy:    {100 * elo_correct / elo_total:.1f}% ({elo_correct}/{elo_total})")
    lines.append("")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))
    print("  Wrote upset analysis to summary")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FENCINGTRACKER ANALYSIS")
    print("=" * 60)
    print()

    # Step 1-2: Load and prepare data (includes gender inference)
    df, gender_map = load_and_prepare_data()
    print()

    # Step 3: Summary statistics
    summary_path = os.path.join(BASE_DIR, "analysis_summary.txt")
    print("Writing summary statistics...")
    write_summary(df, gender_map, summary_path)
    print()

    # Step 4: Rating cross-tabs
    print("Computing rating cross-tabs...")
    ct_path = os.path.join(BASE_DIR, "rating_crosstab.csv")
    # Also collect lines for appending to summary
    extra_lines = []
    ct_all, weapon_cts = write_rating_crosstabs(df, ct_path, extra_lines)
    with open(summary_path, "a") as f:
        f.write("\n".join(extra_lines))
    print()

    # Step 5: ELO analysis
    print("Computing ELO cross-tabs...")
    elo_ct = compute_elo_crosstab(df)
    elo_ct_path = os.path.join(BASE_DIR, "elo_crosstab.csv")
    elo_ct.to_csv(elo_ct_path)
    print(f"  Wrote ELO cross-tab to {elo_ct_path}")
    print()

    # Step 6: Logistic regression
    print("Running logistic regression...")
    logit_result = run_logistic_regression(df, summary_path)
    print()

    # Step 7: Visualizations
    print("Generating visualizations...")
    plot_rating_heatmap(df, os.path.join(BASE_DIR, "rating_heatmap.png"))
    plot_rating_heatmap_by_weapon(df, os.path.join(BASE_DIR, "rating_heatmap_by_weapon.png"))
    plot_elo_win_curve(df, logit_result, os.path.join(BASE_DIR, "elo_win_curve.png"))
    plot_elo_win_curve_by_weapon(df, os.path.join(BASE_DIR, "elo_win_curve_by_weapon.png"))
    plot_rating_distribution(df, os.path.join(BASE_DIR, "rating_distribution.png"))
    plot_elo_distribution(df, os.path.join(BASE_DIR, "elo_distribution.png"))
    plot_upset_analysis(df, os.path.join(BASE_DIR, "upset_analysis.png"))
    print()

    # Step 8: Upset & additional analyses
    print("Writing upset & additional analyses...")
    write_upset_analysis(df, summary_path)
    print()

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Summary:     {summary_path}")
    print(f"  Rating CT:   {ct_path}")
    print(f"  ELO CT:      {elo_ct_path}")
    print(f"  Plots:       7 PNG files in {BASE_DIR}")


if __name__ == "__main__":
    main()
