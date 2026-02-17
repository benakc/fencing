#!/usr/bin/env python3
"""
FencingTracker Analysis v3 — Summary statistics, cross-tabs, logistic regression,
and visualizations for 2024 bout data (Summer Nationals population).

All analyses are sliced by:
  - Bout type: Total, DE, Pool
  - Gender matchup: Total, M (M vs M), F (F vs F), Mixed (cross-gender)
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fencingtracker_bouts_2024.csv")

RATING_ORDER = ["A", "B", "C", "D", "E", "U"]
RATING_RANK = {r: i for i, r in enumerate(RATING_ORDER)}

BOUT_TYPES = ["Total", "DE", "Pool"]
GENDER_SLICES = ["Total", "M", "F", "Mixed"]


# ── Slicing ──────────────────────────────────────────────────────────────────

def slice_df(df, bout_type, gender):
    """Filter df by bout_type and gender matchup."""
    sub = df
    if bout_type != "Total":
        sub = sub[sub["bout_type"] == bout_type]
    if gender == "M":
        sub = sub[(sub["fencer_gender"] == "M") & (sub["opponent_gender"] == "M")]
    elif gender == "F":
        sub = sub[(sub["fencer_gender"] == "F") & (sub["opponent_gender"] == "F")]
    elif gender == "Mixed":
        sub = sub[sub["fencer_gender"] != sub["opponent_gender"]]
    return sub


def slice_label(bout_type, gender):
    return f"{bout_type} / {gender}"


# ── Data Preparation ─────────────────────────────────────────────────────────

def extract_rating_letter(rating_str):
    """Extract letter-only rating: 'A25' -> 'A', 'U' -> 'U', '' -> 'U'."""
    if not rating_str or str(rating_str).strip() in ("", "nan"):
        return "U"
    first = str(rating_str).strip()[0].upper()
    if first in RATING_RANK:
        return first
    return "U"


def parse_score(score_str):
    """Parse score string '5:3' -> (5, 3). Returns (None, None) on failure."""
    parts = str(score_str).split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def load_and_prepare_data():
    """Load CSV and add computed columns. Returns DataFrame."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} bouts")

    # Extract rating letters
    df["r_fencer"] = df["fencer_rating"].astype(str).apply(extract_rating_letter)
    df["r_opponent"] = df["opponent_rating"].astype(str).apply(extract_rating_letter)

    # Parse scores
    scores = df["score"].apply(lambda s: pd.Series(parse_score(str(s)), index=["score_fencer", "score_opponent"]))
    df["score_fencer"] = scores["score_fencer"]
    df["score_opponent"] = scores["score_opponent"]

    # Win flag (from fencer's perspective)
    df["fencer_wins"] = (df["result"] == "W").astype(int)

    # ELO columns
    df["elo_fencer"] = pd.to_numeric(df["fencer_elo"], errors="coerce")
    df["elo_opponent"] = pd.to_numeric(df["opponent_elo"], errors="coerce")
    df["elo_diff"] = df["elo_fencer"] - df["elo_opponent"]

    # Rating ranks (for vectorized upset analysis)
    df["rf_rank"] = df["r_fencer"].map(RATING_RANK).fillna(5).astype(int)
    df["ro_rank"] = df["r_opponent"].map(RATING_RANK).fillna(5).astype(int)
    df["rating_gap"] = (df["rf_rank"] - df["ro_rank"]).abs()

    # Gender matchup
    df["gender_matchup"] = df["fencer_gender"].fillna("?") + " vs " + df["opponent_gender"].fillna("?")

    # Print overview
    all_ids = set(df["fencer_id"].astype(str).unique()) | set(df["opponent_id"].astype(str).unique())
    all_ids.discard(""); all_ids.discard("nan")
    print(f"  Unique fencer IDs: {len(all_ids)}")
    g = df.groupby("fencer_gender")["fencer_id"].nunique()
    print(f"  Gender: {dict(g)}")
    elo_present = df["elo_fencer"].notna().sum()
    print(f"  ELO coverage: {elo_present}/{len(df)} ({100*elo_present/len(df):.1f}%)")

    return df


# ── Vectorized Mirroring ─────────────────────────────────────────────────────

def mirror_ratings(df):
    """Create mirrored rating records (fencer + opponent perspectives). Vectorized."""
    fwd = df[["r_fencer", "r_opponent", "fencer_wins"]].copy()
    fwd.columns = ["my_rating", "opp_rating", "win"]
    rev = df[["r_opponent", "r_fencer", "fencer_wins"]].copy()
    rev.columns = ["my_rating", "opp_rating", "win"]
    rev["win"] = 1 - rev["win"]
    return pd.concat([fwd, rev], ignore_index=True)


def mirror_elo_bins(df, bins, labels):
    """Create mirrored ELO bin records. Vectorized."""
    edf = df.dropna(subset=["elo_fencer", "elo_opponent"]).copy()
    if len(edf) == 0:
        return pd.DataFrame()
    edf["elo_bin_f"] = pd.cut(edf["elo_fencer"], bins=bins, labels=labels, include_lowest=True).astype(str)
    edf["elo_bin_o"] = pd.cut(edf["elo_opponent"], bins=bins, labels=labels, include_lowest=True).astype(str)
    fwd = edf[["elo_bin_f", "elo_bin_o", "fencer_wins"]].copy()
    fwd.columns = ["my_bin", "opp_bin", "win"]
    rev = edf[["elo_bin_o", "elo_bin_f", "fencer_wins"]].copy()
    rev.columns = ["my_bin", "opp_bin", "win"]
    rev["win"] = 1 - rev["win"]
    return pd.concat([fwd, rev], ignore_index=True)


def compute_upset_df(df):
    """Build vectorized upset DataFrame. Returns subset where ratings differ."""
    diff_mask = df["rf_rank"] != df["ro_rank"]
    sub = df[diff_mask].copy()
    if len(sub) == 0:
        return sub
    sub["higher_won"] = (
        ((sub["rf_rank"] < sub["ro_rank"]) & (sub["fencer_wins"] == 1)) |
        ((sub["rf_rank"] > sub["ro_rank"]) & (sub["fencer_wins"] == 0))
    ).astype(int)
    sub["upset"] = 1 - sub["higher_won"]
    return sub


# ── Rating Cross-Tab ─────────────────────────────────────────────────────────

def compute_rating_crosstab(df):
    """Compute win% cross-tab: row=my rating, col=opponent rating. Vectorized."""
    rdf = mirror_ratings(df)
    grouped = rdf.groupby(["my_rating", "opp_rating"])["win"].agg(["mean", "count"]).reset_index()

    rows = []
    for my_r in RATING_ORDER:
        row_data = {"rating": my_r}
        for opp_r in RATING_ORDER:
            match = grouped[(grouped["my_rating"] == my_r) & (grouped["opp_rating"] == opp_r)]
            if len(match) > 0 and match.iloc[0]["count"] > 0:
                win_pct = 100 * match.iloc[0]["mean"]
                n = int(match.iloc[0]["count"])
                flag = "*" if n < 20 else ""
                row_data[opp_r] = f"{win_pct:.0f}% ({n}){flag}"
            else:
                row_data[opp_r] = "-"
        rows.append(row_data)
    return pd.DataFrame(rows).set_index("rating")


# ── ELO Analysis ─────────────────────────────────────────────────────────────

def compute_elo_crosstab(df, bin_width=200):
    """Bin ELO into ranges and compute win rate by bin matchup. Vectorized."""
    edf = df.dropna(subset=["elo_fencer", "elo_opponent"])
    if len(edf) == 0:
        return pd.DataFrame()

    all_elo = pd.concat([edf["elo_fencer"], edf["elo_opponent"]])
    elo_min, elo_max = all_elo.min(), all_elo.max()
    bins = list(range(int(elo_min // bin_width * bin_width),
                      int(elo_max // bin_width * bin_width) + bin_width + 1, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]

    rdf = mirror_elo_bins(df, bins, labels)
    if len(rdf) == 0:
        return pd.DataFrame()

    grouped = rdf.groupby(["my_bin", "opp_bin"])["win"].agg(["mean", "count"]).reset_index()
    used_labels = sorted(rdf["my_bin"].unique())

    rows = []
    for my_b in used_labels:
        row_data = {"elo_bin": my_b}
        for opp_b in used_labels:
            match = grouped[(grouped["my_bin"] == my_b) & (grouped["opp_bin"] == opp_b)]
            if len(match) > 0 and match.iloc[0]["count"] > 0:
                win_pct = 100 * match.iloc[0]["mean"]
                n = int(match.iloc[0]["count"])
                flag = "*" if n < 20 else ""
                row_data[opp_b] = f"{win_pct:.0f}% ({n}){flag}"
            else:
                row_data[opp_b] = "-"
        rows.append(row_data)
    return pd.DataFrame(rows).set_index("elo_bin")


def compute_elo_win_rate_curve(df, bin_width=100):
    """Win rate as a function of ELO difference."""
    edf = df.dropna(subset=["elo_diff"]).copy()
    edf["elo_diff_bin"] = (edf["elo_diff"] / bin_width).round() * bin_width
    grouped = edf.groupby("elo_diff_bin").agg(
        win_rate=("fencer_wins", "mean"),
        count=("fencer_wins", "count"),
    ).reset_index()
    return grouped


# ── Logistic Regression ──────────────────────────────────────────────────────

def run_logistic_regression(df):
    """P(fencer_wins) ~ elo_diff + weapon. Returns (result, summary_text, predictions)."""
    rdf = df.dropna(subset=["elo_diff"]).copy()
    rdf = rdf[rdf["elo_diff"].abs() < 5000]

    if len(rdf) < 50:
        return None, "Insufficient data for logistic regression", []

    rdf["weapon_foil"] = (rdf["weapon"] == "foil").astype(int)
    rdf["weapon_saber"] = (rdf["weapon"] == "saber").astype(int)

    X_cols = ["elo_diff", "weapon_foil", "weapon_saber"]
    X = rdf[X_cols].astype(float)
    X = sm.add_constant(X)
    y = rdf["fencer_wins"].astype(float)

    try:
        model = Logit(y, X)
        result = model.fit(disp=0)
        summary_text = str(result.summary())
    except Exception as e:
        return None, f"Logistic regression failed: {e}", []

    predictions = []
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

    return result, summary_text, predictions


# ── Upset & Prediction Accuracy (vectorized) ────────────────────────────────

def compute_upset_stats(df):
    """Return dict with upset analysis stats. Fully vectorized."""
    udf = compute_upset_df(df)
    if len(udf) == 0:
        return None

    stats = {"total_different_rating": len(udf)}

    # By rating gap
    gap_stats = udf.groupby("rating_gap")["upset"].agg(["mean", "count"]).reset_index()
    gap_stats.columns = ["gap", "upset_rate", "count"]
    stats["by_gap"] = gap_stats

    # By bout type
    bt_stats = udf.groupby("bout_type")["upset"].agg(["mean", "count"]).reset_index()
    bt_stats.columns = ["bout_type", "upset_rate", "count"]
    stats["by_bout_type"] = bt_stats

    # By gap x bout type
    gbt_stats = udf.groupby(["rating_gap", "bout_type"])["upset"].agg(["mean", "count"]).reset_index()
    gbt_stats.columns = ["gap", "bout_type", "upset_rate", "count"]
    stats["by_gap_bout_type"] = gbt_stats

    return stats


def compute_prediction_accuracy(df):
    """Compute rating-based and ELO-based prediction accuracy. Vectorized."""
    results = {}

    # Rating-based (exclude same-rating bouts)
    diff = df[df["rf_rank"] != df["ro_rank"]].copy()
    if len(diff) > 0:
        predicted_fencer_wins = diff["rf_rank"] < diff["ro_rank"]
        actual = diff["fencer_wins"].astype(bool)
        correct = (predicted_fencer_wins == actual).sum()
        results["rating"] = (int(correct), len(diff))

    # ELO-based (exclude zero-diff bouts)
    valid = df.dropna(subset=["elo_diff"]).copy()
    valid = valid[valid["elo_diff"] != 0]
    if len(valid) > 0:
        predicted_fencer_wins = valid["elo_diff"] > 0
        actual = valid["fencer_wins"].astype(bool)
        correct = (predicted_fencer_wins == actual).sum()
        results["elo"] = (int(correct), len(valid))

    return results


# ── Summary Writer ───────────────────────────────────────────────────────────

def write_overview(df, lines):
    """Write data overview section."""
    lines.append("1. DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total bouts: {len(df):,}")
    lines.append(f"Unique events: {df['event_id'].nunique():,}")
    all_ids = set(df["fencer_id"].astype(str)) | set(df["opponent_id"].astype(str))
    all_ids.discard(""); all_ids.discard("nan")
    lines.append(f"Unique fencers: {len(all_ids):,}")
    lines.append(f"Fencer win rate: {df['fencer_wins'].mean():.1%} (from reporting fencer's perspective)")
    lines.append("")

    lines.append("Bouts by weapon:")
    for w in sorted(df["weapon"].dropna().unique()):
        lines.append(f"  {w:8s}: {len(df[df['weapon'] == w]):,}")
    lines.append("")

    lines.append("Bouts by type:")
    for bt in sorted(df["bout_type"].dropna().unique()):
        lines.append(f"  {bt:8s}: {len(df[df['bout_type'] == bt]):,}")
    lines.append("")

    lines.append("Bouts by fencer gender:")
    for g in ["M", "F"]:
        lines.append(f"  {g:8s}: {len(df[df['fencer_gender'] == g]):,}")
    lines.append("")

    lines.append("Bouts by weapon x type:")
    ct = pd.crosstab(df["weapon"], df["bout_type"])
    lines.append(ct.to_string())
    lines.append("")

    # Slice count table
    lines.append("Bout counts by slice (Bout Type x Gender Matchup):")
    header = f"  {'':12s}" + "".join(f"{g:>12s}" for g in GENDER_SLICES)
    lines.append(header)
    for bt in BOUT_TYPES:
        row = f"  {bt:12s}"
        for gs in GENDER_SLICES:
            sub = slice_df(df, bt, gs)
            row += f"{len(sub):>12,}"
        lines.append(row)
    lines.append("")


def write_distributions(df, lines):
    """Write rating and ELO distribution sections."""
    # Rating distribution
    lines.append("2. RATING DISTRIBUTION")
    lines.append("-" * 40)
    all_ratings = list(df["r_fencer"]) + list(df["r_opponent"])
    rating_counts = Counter(all_ratings)
    for r in RATING_ORDER:
        lines.append(f"  {r}: {rating_counts.get(r, 0):,}")
    lines.append("")

    # ELO distribution
    lines.append("3. ELO DISTRIBUTION")
    lines.append("-" * 40)
    all_elo = pd.concat([df["elo_fencer"], df["elo_opponent"]]).dropna()
    if len(all_elo) > 0:
        lines.append(f"  Count: {len(all_elo):,}")
        lines.append(f"  Mean:  {all_elo.mean():.0f}")
        lines.append(f"  Std:   {all_elo.std():.0f}")
        lines.append(f"  Min:   {all_elo.min():.0f}")
        lines.append(f"  25%:   {all_elo.quantile(0.25):.0f}")
        lines.append(f"  50%:   {all_elo.quantile(0.50):.0f}")
        lines.append(f"  75%:   {all_elo.quantile(0.75):.0f}")
        lines.append(f"  Max:   {all_elo.max():.0f}")
    lines.append("")

    # Gender
    lines.append("4. GENDER DISTRIBUTION")
    lines.append("-" * 40)
    fencer_genders = df.drop_duplicates("fencer_id")[["fencer_id", "fencer_gender"]]
    g_counts = fencer_genders["fencer_gender"].value_counts()
    for g in ["M", "F"]:
        lines.append(f"  {g}: {g_counts.get(g, 0):,}")
    lines.append("")
    lines.append("Gender matchup counts (bout-level):")
    gm_counts = df["gender_matchup"].value_counts()
    for gm, n in gm_counts.head(10).items():
        lines.append(f"  {gm:16s}: {n:,}")
    lines.append("")

    # Rating matchup sample sizes
    lines.append("5. RATING MATCHUP SAMPLE SIZES")
    lines.append("-" * 40)
    matchup = df.apply(
        lambda row: " vs ".join(sorted([row["r_fencer"], row["r_opponent"]],
                                       key=lambda x: RATING_RANK.get(x, 99))), axis=1)
    rm_counts = matchup.value_counts().sort_index()
    for rm, n in rm_counts.items():
        flag = " *** SMALL SAMPLE" if n < 20 else ""
        lines.append(f"  {rm:10s}: {n:,}{flag}")
    lines.append("")

    # DE score margins
    de = df[df["bout_type"] == "DE"].copy()
    if len(de) > 0:
        lines.append("6. DE SCORE MARGINS (from winner's perspective)")
        lines.append("-" * 40)
        de_wins = de[de["fencer_wins"] == 1]
        de_margins = (de_wins["score_fencer"] - de_wins["score_opponent"]).dropna()
        if len(de_margins) > 0:
            lines.append(f"  Count: {len(de_margins):,}")
            lines.append(f"  Mean:  {de_margins.mean():.1f}")
            lines.append(f"  Std:   {de_margins.std():.1f}")
            lines.append(f"  Min:   {de_margins.min():.0f}")
            lines.append(f"  Median:{de_margins.median():.0f}")
            lines.append(f"  Max:   {de_margins.max():.0f}")
        lines.append("")


def write_slice_analysis(df, bout_type, gender, section_num, lines):
    """Write analysis for a single slice. Returns section_num for next section."""
    label = slice_label(bout_type, gender)
    sub = slice_df(df, bout_type, gender)
    n = len(sub)

    lines.append(f"\n{'='*70}")
    lines.append(f"SECTION {section_num}: {label.upper()} (N={n:,})")
    lines.append(f"{'='*70}")

    if n < 20:
        lines.append("  Insufficient data (< 20 bouts). Skipping.")
        return section_num + 1

    lines.append(f"  Win rate: {sub['fencer_wins'].mean():.1%}")
    lines.append("")

    # Rating cross-tab (all weapons)
    lines.append(f"{section_num}.1 RATING WIN% CROSS-TAB — {label}")
    lines.append("-" * 40)
    lines.append("Row = my rating, Col = opponent rating")
    lines.append("Cell = win% (N), * = N < 20")
    ct = compute_rating_crosstab(sub)
    lines.append(ct.to_string())
    lines.append("")

    # Per-weapon rating cross-tabs
    for w in sorted(sub["weapon"].dropna().unique()):
        wdf = sub[sub["weapon"] == w]
        if len(wdf) >= 20:
            lines.append(f"  {w.upper()} (N={len(wdf):,}):")
            wct = compute_rating_crosstab(wdf)
            lines.append(wct.to_string())
            lines.append("")

    # Logistic regression
    lines.append(f"{section_num}.2 LOGISTIC REGRESSION — {label}")
    lines.append("-" * 40)
    result, summary_text, predictions = run_logistic_regression(sub)
    lines.append("Model: P(fencer_wins) ~ elo_diff + weapon")
    lines.append("")
    lines.append(summary_text)
    if predictions:
        lines.append("")
        lines.append("Predicted win probabilities:")
        lines.extend(predictions)
    lines.append("")

    # Upset analysis
    lines.append(f"{section_num}.3 UPSET ANALYSIS — {label}")
    lines.append("-" * 40)
    ustats = compute_upset_stats(sub)
    if ustats:
        lines.append(f"Bouts with different ratings: {ustats['total_different_rating']:,}")
        lines.append("")
        lines.append("Upset rate by rating gap:")
        for _, row in ustats["by_gap"].iterrows():
            lines.append(f"  {int(row['gap'])}-letter gap: {row['upset_rate']:.1%} upset rate (N={int(row['count']):,})")
        lines.append("")

        if len(ustats["by_bout_type"]) > 1:
            lines.append("Pool vs DE upset rates:")
            for _, row in ustats["by_bout_type"].iterrows():
                lines.append(f"  {row['bout_type']:8s}: {row['upset_rate']:.1%} upset rate (N={int(row['count']):,})")
            lines.append("")

            lines.append("Pool vs DE upset rates by gap:")
            gbt = ustats["by_gap_bout_type"]
            for gap in sorted(gbt["gap"].unique()):
                gap_rows = gbt[gbt["gap"] == gap]
                parts = []
                for _, row in gap_rows.iterrows():
                    parts.append(f"{row['bout_type']}={row['upset_rate']:.1%} (N={int(row['count']):,})")
                lines.append(f"  {int(gap)}-letter gap: {', '.join(parts)}")
            lines.append("")
    else:
        lines.append("No bouts with different ratings found.")
        lines.append("")

    # Prediction accuracy
    lines.append(f"{section_num}.4 PREDICTION ACCURACY — {label}")
    lines.append("-" * 40)
    acc = compute_prediction_accuracy(sub)
    if "rating" in acc:
        correct, total = acc["rating"]
        lines.append(f"Rating-based accuracy: {100*correct/total:.1f}% ({correct:,}/{total:,})")
    if "elo" in acc:
        correct, total = acc["elo"]
        lines.append(f"ELO-based accuracy:    {100*correct/total:.1f}% ({correct:,}/{total:,})")
    lines.append("")

    return section_num + 1


# ── Visualizations ───────────────────────────────────────────────────────────

def _build_rating_pivot(df):
    """Build win% pivot table from mirrored ratings. Vectorized."""
    rdf = mirror_ratings(df)
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
    return pivot, annot


def plot_rating_heatmap(df, out_path):
    """Win% heatmap by rating pair, all weapons."""
    pivot, annot = _build_rating_pivot(df)
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
    weapons = sorted(df["weapon"].dropna().unique())
    fig, axes = plt.subplots(1, len(weapons), figsize=(8 * len(weapons), 7))
    if len(weapons) == 1:
        axes = [axes]

    for ax, w in zip(axes, weapons):
        wdf = df[df["weapon"] == w]
        pivot, annot = _build_rating_pivot(wdf)
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Win %"})
        ax.set_title(f"{w.upper()} (N={len(wdf):,})", fontsize=13)
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
    ax.scatter(curve["elo_diff_bin"], curve["win_rate"], s=np.clip(curve["count"] / 10, 5, 200),
               alpha=0.7, color="steelblue", label="Observed (size ~ N)")

    if logit_result is not None:
        x_range = np.linspace(-2000, 2000, 200)
        n_params = len(logit_result.params)
        X_pred = np.zeros((len(x_range), n_params))
        X_pred[:, 0] = 1
        X_pred[:, 1] = x_range
        y_pred = logit_result.predict(X_pred)
        ax.plot(x_range, y_pred, "r-", linewidth=2, label="Logistic fit (epee)")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ELO Difference (fencer - opponent)", fontsize=12)
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
    weapons = sorted(df["weapon"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"epee": "steelblue", "foil": "green", "saber": "darkorange"}

    for w in weapons:
        wdf = df[df["weapon"] == w]
        curve = compute_elo_win_rate_curve(wdf, bin_width=150)
        curve = curve[curve["count"] >= 5]
        ax.scatter(curve["elo_diff_bin"], curve["win_rate"],
                   s=np.clip(curve["count"] / 10, 5, 100),
                   alpha=0.6, color=colors.get(w, "gray"), label=f"{w} (N={len(wdf):,})")

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
    all_ratings = list(df["r_fencer"]) + list(df["r_opponent"])
    rating_counts = Counter(all_ratings)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = RATING_ORDER
    y = [rating_counts.get(r, 0) for r in x]
    bars = ax.bar(x, y, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(y) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Appearances (fencer-bouts)", fontsize=12)
    ax.set_title("Rating Distribution Across All Bouts", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_elo_distribution(df, out_path):
    """Histogram of ELO values."""
    all_elo = pd.concat([df["elo_fencer"], df["elo_opponent"]]).dropna()
    if len(all_elo) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_elo, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(all_elo.mean(), color="red", linestyle="--", label=f"Mean: {all_elo.mean():.0f}")
    ax.axvline(all_elo.median(), color="orange", linestyle="--", label=f"Median: {all_elo.median():.0f}")
    ax.set_xlabel("ELO", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("ELO Distribution Across All Bouts", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_upset_analysis(df, out_path):
    """Upset rate by rating gap + pool vs DE comparison. Vectorized."""
    udf = compute_upset_df(df)
    if len(udf) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    gap_stats = udf.groupby("rating_gap")["upset"].agg(["mean", "count"]).reset_index()
    bars = ax.bar(gap_stats["rating_gap"], gap_stats["mean"] * 100,
                  color="steelblue", edgecolor="white")
    for bar, (_, row) in zip(bars, gap_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"N={int(row['count']):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Rating Gap (letters apart)", fontsize=12)
    ax.set_ylabel("Upset Rate (%)", fontsize=12)
    ax.set_title("Upset Rate by Rating Gap", fontsize=13)
    ax.set_xticks(sorted(udf["rating_gap"].unique()))

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


# ── Cross-Gender Analysis ────────────────────────────────────────────────────

def _orient_mixed_bouts(df):
    """Orient mixed-gender bouts so F is always 'row' and M is always 'col'.

    Returns DataFrame with columns: f_rating, m_rating, f_elo_avg, m_elo_avg,
    f_wins (1 if female won), weapon, bout_type.
    """
    mixed = df[df["fencer_gender"] != df["opponent_gender"]].copy()
    if len(mixed) == 0:
        return pd.DataFrame()

    # When fencer is F, opponent is M
    fm = mixed[mixed["fencer_gender"] == "F"].copy()
    fm_rows = pd.DataFrame({
        "f_rating": fm["r_fencer"].values,
        "m_rating": fm["r_opponent"].values,
        "f_wins": fm["fencer_wins"].values,
        "weapon": fm["weapon"].values,
        "bout_type": fm["bout_type"].values,
        "f_id": fm["fencer_id"].values,
        "m_id": fm["opponent_id"].values,
    })

    # When fencer is M, opponent is F
    mf = mixed[mixed["fencer_gender"] == "M"].copy()
    mf_rows = pd.DataFrame({
        "f_rating": mf["r_opponent"].values,
        "m_rating": mf["r_fencer"].values,
        "f_wins": (1 - mf["fencer_wins"]).values,
        "weapon": mf["weapon"].values,
        "bout_type": mf["bout_type"].values,
        "f_id": mf["opponent_id"].values,
        "m_id": mf["fencer_id"].values,
    })

    return pd.concat([fm_rows, mf_rows], ignore_index=True)


def _compute_fencer_avg_elo(df):
    """Compute average ELO per fencer from same-gender bouts (where ELO exists)."""
    same = df[df["fencer_gender"] == df["opponent_gender"]].copy()
    same["elo_f"] = pd.to_numeric(same["fencer_elo"], errors="coerce")
    elo_map = same.dropna(subset=["elo_f"]).groupby("fencer_id")["elo_f"].mean()
    return elo_map


def compute_cross_gender_rating_crosstab(oriented_df):
    """F rating (row) vs M rating (col) → F win%. No mirroring needed."""
    if len(oriented_df) == 0:
        return pd.DataFrame()

    grouped = oriented_df.groupby(["f_rating", "m_rating"])["f_wins"].agg(["mean", "count"]).reset_index()

    rows = []
    for f_r in RATING_ORDER:
        row_data = {"F_rating": f_r}
        for m_r in RATING_ORDER:
            match = grouped[(grouped["f_rating"] == f_r) & (grouped["m_rating"] == m_r)]
            if len(match) > 0 and match.iloc[0]["count"] > 0:
                win_pct = 100 * match.iloc[0]["mean"]
                n = int(match.iloc[0]["count"])
                flag = "*" if n < 20 else ""
                row_data[m_r] = f"{win_pct:.0f}% ({n}){flag}"
            else:
                row_data[m_r] = "-"
        rows.append(row_data)
    return pd.DataFrame(rows).set_index("F_rating")


def compute_cross_gender_elo_vs_rating(oriented_df, elo_map, elo_bin_width=200):
    """F ELO bucket (row) vs M rating (col) → F win%.
    Also M ELO bucket (row) vs F rating (col) → M win%.
    Returns (f_elo_vs_m_rating, m_elo_vs_f_rating) DataFrames.
    """
    if len(oriented_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    odf = oriented_df.copy()
    odf["f_elo"] = odf["f_id"].map(elo_map)
    odf["m_elo"] = odf["m_id"].map(elo_map)

    results = []
    for perspective, elo_col, rating_col, win_col, label in [
        ("F ELO vs M Rating", "f_elo", "m_rating", "f_wins", "F_ELO"),
        ("M ELO vs F Rating", "m_elo", "f_rating", "f_wins", "M_ELO"),
    ]:
        sub = odf.dropna(subset=[elo_col]).copy()
        if len(sub) == 0:
            results.append(pd.DataFrame())
            continue

        all_elo = sub[elo_col]
        elo_min, elo_max = all_elo.min(), all_elo.max()
        bins = list(range(int(elo_min // elo_bin_width * elo_bin_width),
                          int(elo_max // elo_bin_width * elo_bin_width) + elo_bin_width + 1,
                          elo_bin_width))
        bin_labels = [f"{b}-{b + elo_bin_width - 1}" for b in bins[:-1]]
        sub["elo_bin"] = pd.cut(sub[elo_col], bins=bins, labels=bin_labels, include_lowest=True).astype(str)

        # For M ELO perspective, win = M wins = 1 - f_wins
        if "M ELO" in perspective:
            sub["win"] = 1 - sub[win_col]
        else:
            sub["win"] = sub[win_col]

        grouped = sub.groupby(["elo_bin", rating_col])["win"].agg(["mean", "count"]).reset_index()
        used_bins = sorted(sub["elo_bin"].unique())

        rows = []
        for eb in used_bins:
            row_data = {label: eb}
            for r in RATING_ORDER:
                match = grouped[(grouped["elo_bin"] == eb) & (grouped[rating_col] == r)]
                if len(match) > 0 and match.iloc[0]["count"] > 0:
                    win_pct = 100 * match.iloc[0]["mean"]
                    n = int(match.iloc[0]["count"])
                    flag = "*" if n < 20 else ""
                    row_data[r] = f"{win_pct:.0f}% ({n}){flag}"
                else:
                    row_data[r] = "-"
            rows.append(row_data)
        results.append(pd.DataFrame(rows).set_index(label))

    return results[0], results[1]


def write_cross_gender_analysis(df, lines):
    """Write full cross-gender analysis section to summary lines."""
    oriented = _orient_mixed_bouts(df)
    elo_map = _compute_fencer_avg_elo(df)

    lines.append("")
    lines.append("=" * 70)
    lines.append("CROSS-GENDER ANALYSIS: F WIN RATE vs M")
    lines.append("Row = Female fencer, Col = Male fencer")
    lines.append("=" * 70)

    for bt in ["Total", "DE", "Pool"]:
        if bt == "Total":
            sub = oriented
        else:
            sub = oriented[oriented["bout_type"] == bt]

        if len(sub) < 5:
            continue

        lines.append(f"\n{'─'*70}")
        lines.append(f"  {bt.upper()} (N={len(sub):,})")
        lines.append(f"{'─'*70}")
        f_win_rate = sub["f_wins"].mean()
        lines.append(f"  Overall F win rate vs M: {f_win_rate:.1%}")
        lines.append("")

        # Rating cross-tab (all weapons)
        lines.append(f"  F Rating (row) vs M Rating (col) → F Win% — {bt}, All Weapons")
        lines.append("  " + "-" * 40)
        lines.append("  Cell = F win% (N), * = N < 20")
        ct = compute_cross_gender_rating_crosstab(sub)
        if len(ct) > 0:
            lines.append(ct.to_string())
        lines.append("")

        # Per-weapon
        for w in sorted(sub["weapon"].dropna().unique()):
            wsub = sub[sub["weapon"] == w]
            if len(wsub) >= 10:
                lines.append(f"  {w.upper()} (N={len(wsub):,}):")
                wct = compute_cross_gender_rating_crosstab(wsub)
                if len(wct) > 0:
                    lines.append(wct.to_string())
                lines.append("")

        # ELO vs rating
        f_elo_m_rat, m_elo_f_rat = compute_cross_gender_elo_vs_rating(sub, elo_map)

        if len(f_elo_m_rat) > 0:
            lines.append(f"  F Avg ELO (row) vs M Rating (col) → F Win% — {bt}")
            lines.append("  " + "-" * 40)
            lines.append("  ELO = fencer's average from same-gender bouts")
            lines.append("  Cell = F win% (N), * = N < 20")
            lines.append(f_elo_m_rat.to_string())
            lines.append("")

        if len(m_elo_f_rat) > 0:
            lines.append(f"  M Avg ELO (row) vs F Rating (col) → M Win% — {bt}")
            lines.append("  " + "-" * 40)
            lines.append("  ELO = fencer's average from same-gender bouts")
            lines.append("  Cell = M win% (N), * = N < 20")
            lines.append(m_elo_f_rat.to_string())
            lines.append("")


# ── Cross-Gender Plots ───────────────────────────────────────────────────────

def _cg_rating_pivot_and_annot(oriented_df):
    """Build F-win% pivot and annotation matrices from oriented mixed bouts."""
    if len(oriented_df) == 0:
        return None, None, None

    grouped = oriented_df.groupby(["f_rating", "m_rating"])["f_wins"]
    win_pct = grouped.mean().unstack()
    counts = grouped.count().unstack()
    win_pct = win_pct.reindex(index=RATING_ORDER, columns=RATING_ORDER) * 100
    counts = counts.reindex(index=RATING_ORDER, columns=RATING_ORDER)

    annot = win_pct.copy().astype(str)
    for r in RATING_ORDER:
        for c in RATING_ORDER:
            v = win_pct.loc[r, c] if r in win_pct.index and c in win_pct.columns else np.nan
            n = counts.loc[r, c] if r in counts.index and c in counts.columns else 0
            if pd.isna(v) or pd.isna(n) or n == 0:
                annot.loc[r, c] = "-"
            else:
                flag = "*" if n < 20 else ""
                annot.loc[r, c] = f"{v:.0f}%\n({int(n)}){flag}"
    return win_pct, annot, counts


def plot_cross_gender_rating_heatmap(oriented_df, bout_type, out_path):
    """F rating vs M rating heatmap for a single bout type."""
    pivot, annot, _ = _cg_rating_pivot_and_annot(oriented_df)
    if pivot is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                linewidths=0.5, ax=ax, cbar_kws={"label": "F Win %"})
    ax.set_title(f"F Win % vs M by Rating — {bout_type} (N={len(oriented_df):,})\n* = N < 20", fontsize=14)
    ax.set_xlabel("Male Rating")
    ax.set_ylabel("Female Rating")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_cross_gender_rating_by_weapon(oriented_df, bout_type, out_path):
    """F rating vs M rating heatmap, one subplot per weapon."""
    weapons = sorted(oriented_df["weapon"].dropna().unique())
    if len(weapons) == 0:
        return

    fig, axes = plt.subplots(1, len(weapons), figsize=(8 * len(weapons), 7))
    if len(weapons) == 1:
        axes = [axes]

    for ax, w in zip(axes, weapons):
        wsub = oriented_df[oriented_df["weapon"] == w]
        pivot, annot, _ = _cg_rating_pivot_and_annot(wsub)
        if pivot is None:
            continue
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "F Win %"})
        ax.set_title(f"{w.upper()} (N={len(wsub):,})", fontsize=13)
        ax.set_xlabel("Male Rating")
        ax.set_ylabel("Female Rating")

    plt.suptitle(f"F Win % vs M by Rating & Weapon — {bout_type}\n* = N < 20", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def _cg_elo_vs_rating_pivot(oriented_df, elo_map, elo_col_id, rating_col,
                             is_m_perspective, elo_bin_width=200):
    """Build ELO-bucket vs rating pivot for cross-gender plots.
    Returns (pivot, annot, elo_labels_sorted) or (None, None, None)."""
    odf = oriented_df.copy()
    odf["_elo"] = odf[elo_col_id].map(elo_map)
    sub = odf.dropna(subset=["_elo"]).copy()
    if len(sub) < 10:
        return None, None, None

    elo_min, elo_max = sub["_elo"].min(), sub["_elo"].max()
    bins = list(range(int(elo_min // elo_bin_width * elo_bin_width),
                      int(elo_max // elo_bin_width * elo_bin_width) + elo_bin_width + 1,
                      elo_bin_width))
    labels = [f"{b}-{b + elo_bin_width - 1}" for b in bins[:-1]]
    sub["elo_bin"] = pd.cut(sub["_elo"], bins=bins, labels=labels, include_lowest=True).astype(str)

    if is_m_perspective:
        sub["win"] = 1 - sub["f_wins"]
    else:
        sub["win"] = sub["f_wins"]

    grouped_mean = sub.groupby(["elo_bin", rating_col])["win"].mean().unstack()
    grouped_count = sub.groupby(["elo_bin", rating_col])["win"].count().unstack()

    # Sort ELO bins numerically
    def _elo_sort_key(s):
        try:
            return int(s.split("-")[0])
        except (ValueError, IndexError):
            return 0
    elo_sorted = sorted(grouped_mean.index, key=_elo_sort_key)

    pivot = (grouped_mean.reindex(index=elo_sorted, columns=RATING_ORDER) * 100)
    counts = grouped_count.reindex(index=elo_sorted, columns=RATING_ORDER)

    annot = pivot.copy().astype(str)
    for r in elo_sorted:
        for c in RATING_ORDER:
            v = pivot.loc[r, c] if r in pivot.index and c in pivot.columns else np.nan
            n = counts.loc[r, c] if r in counts.index and c in counts.columns else 0
            if pd.isna(v) or pd.isna(n) or n == 0:
                annot.loc[r, c] = "-"
            else:
                flag = "*" if n < 20 else ""
                annot.loc[r, c] = f"{v:.0f}%\n({int(n)}){flag}"

    return pivot, annot, elo_sorted


def plot_cross_gender_elo_vs_rating(oriented_df, elo_map, bout_type, out_dir):
    """Generate F-ELO vs M-Rating and M-ELO vs F-Rating heatmaps."""
    bt_tag = bout_type.lower()

    # F ELO vs M Rating
    pivot, annot, _ = _cg_elo_vs_rating_pivot(
        oriented_df, elo_map, "f_id", "m_rating", is_m_perspective=False)
    if pivot is not None and len(pivot) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.55)))
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=50, vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "F Win %"})
        ax.set_title(f"F Avg ELO vs M Rating → F Win% — {bout_type}\nELO from same-gender bouts, * = N < 20",
                     fontsize=13)
        ax.set_xlabel("Male Rating")
        ax.set_ylabel("Female Avg ELO")
        plt.tight_layout()
        path = os.path.join(out_dir, f"cg_f_elo_vs_m_rating_{bt_tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    # M ELO vs F Rating
    pivot, annot, _ = _cg_elo_vs_rating_pivot(
        oriented_df, elo_map, "m_id", "f_rating", is_m_perspective=True)
    if pivot is not None and len(pivot) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.55)))
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn_r", center=50, vmin=0, vmax=100,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "M Win %"})
        ax.set_title(f"M Avg ELO vs F Rating → M Win% — {bout_type}\nELO from same-gender bouts, * = N < 20",
                     fontsize=13)
        ax.set_xlabel("Female Rating")
        ax.set_ylabel("Male Avg ELO")
        plt.tight_layout()
        path = os.path.join(out_dir, f"cg_m_elo_vs_f_rating_{bt_tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def generate_cross_gender_plots(df):
    """Generate all cross-gender plots for Total, DE, Pool."""
    oriented = _orient_mixed_bouts(df)
    elo_map = _compute_fencer_avg_elo(df)

    for bt in ["Total", "DE", "Pool"]:
        if bt == "Total":
            sub = oriented
        else:
            sub = oriented[oriented["bout_type"] == bt]

        if len(sub) < 10:
            continue

        bt_tag = bt.lower()

        # Rating heatmaps
        plot_cross_gender_rating_heatmap(
            sub, bt, os.path.join(BASE_DIR, f"cg_rating_{bt_tag}.png"))
        plot_cross_gender_rating_by_weapon(
            sub, bt, os.path.join(BASE_DIR, f"cg_rating_by_weapon_{bt_tag}.png"))

        # ELO vs rating heatmaps
        plot_cross_gender_elo_vs_rating(sub, elo_map, bt, BASE_DIR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FENCINGTRACKER ANALYSIS — 2024 DATA")
    print("=" * 60)
    print()

    df = load_and_prepare_data()
    print()

    summary_path = os.path.join(BASE_DIR, "analysis_summary.txt")

    # Build summary text
    lines = []
    lines.append("=" * 70)
    lines.append("FENCINGTRACKER ANALYSIS — 2024 BOUT DATA")
    lines.append("Population: 2024 Summer Nationals (Div I, IA, II, III)")
    lines.append("All analyses sliced by Bout Type (Total/DE/Pool) x Gender (Total/M/F/Mixed)")
    lines.append("=" * 70)
    lines.append("")

    # Overview (Total only)
    print("Writing overview statistics...")
    write_overview(df, lines)
    write_distributions(df, lines)
    print()

    # Sliced analysis
    section_num = 7
    total_slices = len(BOUT_TYPES) * len(GENDER_SLICES)
    slice_idx = 0
    for bt in BOUT_TYPES:
        for gs in GENDER_SLICES:
            slice_idx += 1
            label = slice_label(bt, gs)
            sub = slice_df(df, bt, gs)
            print(f"  [{slice_idx}/{total_slices}] {label} (N={len(sub):,})...")
            section_num = write_slice_analysis(df, bt, gs, section_num, lines)

    # Cross-gender analysis
    print("Computing cross-gender analysis...")
    write_cross_gender_analysis(df, lines)
    print()

    # Write summary file
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote summary to {summary_path}")
    print()

    # Rating cross-tab CSV (all slices)
    print("Writing rating cross-tab CSV...")
    ct_path = os.path.join(BASE_DIR, "rating_crosstab.csv")
    with open(ct_path, "w") as f:
        f.write("Rating Win% Cross-Tab (row=my rating, col=opponent rating)\n")
        f.write("Cell = win% (N), * = N < 20\n\n")
        for bt in BOUT_TYPES:
            for gs in GENDER_SLICES:
                label = slice_label(bt, gs)
                sub = slice_df(df, bt, gs)
                if len(sub) < 20:
                    continue
                f.write(f"\n{label.upper()}\n")
                ct = compute_rating_crosstab(sub)
                f.write(ct.to_csv())
                f.write("\n")
    print(f"  Wrote {ct_path}")
    print()

    # ELO cross-tab CSV (Total/Total only)
    print("Writing ELO cross-tab CSV...")
    elo_ct = compute_elo_crosstab(df)
    elo_ct_path = os.path.join(BASE_DIR, "elo_crosstab.csv")
    elo_ct.to_csv(elo_ct_path)
    print(f"  Wrote {elo_ct_path}")
    print()

    # Visualizations (Total/Total)
    print("Generating overall visualizations...")
    logit_result, _, _ = run_logistic_regression(df)
    plot_rating_heatmap(df, os.path.join(BASE_DIR, "rating_heatmap.png"))
    plot_rating_heatmap_by_weapon(df, os.path.join(BASE_DIR, "rating_heatmap_by_weapon.png"))
    plot_elo_win_curve(df, logit_result, os.path.join(BASE_DIR, "elo_win_curve.png"))
    plot_elo_win_curve_by_weapon(df, os.path.join(BASE_DIR, "elo_win_curve_by_weapon.png"))
    plot_rating_distribution(df, os.path.join(BASE_DIR, "rating_distribution.png"))
    plot_elo_distribution(df, os.path.join(BASE_DIR, "elo_distribution.png"))
    plot_upset_analysis(df, os.path.join(BASE_DIR, "upset_analysis.png"))
    print()

    # Per bout-type visualizations (DE, Pool)
    print("Generating per-bout-type visualizations...")
    for bt in ["DE", "Pool"]:
        bt_df = df[df["bout_type"] == bt]
        bt_tag = bt.lower()
        logit_bt, _, _ = run_logistic_regression(bt_df)
        plot_rating_heatmap(bt_df, os.path.join(BASE_DIR, f"rating_heatmap_{bt_tag}.png"))
        plot_rating_heatmap_by_weapon(bt_df, os.path.join(BASE_DIR, f"rating_heatmap_by_weapon_{bt_tag}.png"))
        plot_elo_win_curve(bt_df, logit_bt, os.path.join(BASE_DIR, f"elo_win_curve_{bt_tag}.png"))
        plot_elo_win_curve_by_weapon(bt_df, os.path.join(BASE_DIR, f"elo_win_curve_by_weapon_{bt_tag}.png"))
        plot_upset_analysis(bt_df, os.path.join(BASE_DIR, f"upset_analysis_{bt_tag}.png"))
    print()

    # Cross-gender plots
    print("Generating cross-gender visualizations...")
    generate_cross_gender_plots(df)
    print()

    # Count PNGs
    png_count = len([f for f in os.listdir(BASE_DIR) if f.endswith(".png")])
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Summary:     {summary_path}")
    print(f"  Rating CT:   {ct_path}")
    print(f"  ELO CT:      {elo_ct_path}")
    print(f"  Plots:       {png_count} PNG files in {BASE_DIR}")


if __name__ == "__main__":
    main()
