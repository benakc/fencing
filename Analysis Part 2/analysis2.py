#!/usr/bin/env python3
"""
Analysis Part 2: Seed vs. Placement at Div I NAC/Nationals Events.

Four analytical approaches:
1. Descriptive/Correlation — Spearman rank correlation, scatter plots
2. Ordinal Regression — placement ~ seed + weapon + gender + weapon:gender
3. Upset Distance — distribution of (place - seed) by weapon/gender
4. Top-N Survival — fraction of top-N seeds finishing in top-N

All segmented by weapon (epee/foil/saber) and gender (M/F), 6 slices + totals.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "seed_placement_div1.csv")

WEAPONS = ["epee", "foil", "saber"]
GENDERS = ["M", "F"]
WEAPON_COLORS = {"epee": "steelblue", "foil": "green", "saber": "darkorange"}
GENDER_MARKERS = {"M": "o", "F": "s"}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load seed/placement CSV."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["place"] = pd.to_numeric(df["place"], errors="coerce")
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce")
    df = df.dropna(subset=["seed", "place"])
    df["seed"] = df["seed"].astype(int)
    df["place"] = df["place"].astype(int)
    print(f"  Loaded {len(df)} rows")
    print(f"  Weapons: {sorted(df['weapon'].unique())}")
    print(f"  Genders: {sorted(df['gender'].unique())}")
    print(f"  Tournaments: {df['tournament_name'].nunique()}")
    return df


# ── Approach 1: Descriptive / Correlation ────────────────────────────────────

def compute_correlations(df):
    """Compute Spearman rank correlation (seed vs place) per weapon/gender slice."""
    results = []

    # Overall
    r, p = stats.spearmanr(df["seed"], df["place"])
    results.append({"weapon": "All", "gender": "All", "n": len(df),
                     "spearman_r": r, "p_value": p})

    # By weapon
    for w in WEAPONS:
        wdf = df[df["weapon"] == w]
        if len(wdf) >= 10:
            r, p = stats.spearmanr(wdf["seed"], wdf["place"])
            results.append({"weapon": w, "gender": "All", "n": len(wdf),
                             "spearman_r": r, "p_value": p})

    # By gender
    for g in GENDERS:
        gdf = df[df["gender"] == g]
        if len(gdf) >= 10:
            r, p = stats.spearmanr(gdf["seed"], gdf["place"])
            results.append({"weapon": "All", "gender": g, "n": len(gdf),
                             "spearman_r": r, "p_value": p})

    # By weapon x gender
    for w in WEAPONS:
        for g in GENDERS:
            sub = df[(df["weapon"] == w) & (df["gender"] == g)]
            if len(sub) >= 10:
                r, p = stats.spearmanr(sub["seed"], sub["place"])
                results.append({"weapon": w, "gender": g, "n": len(sub),
                                 "spearman_r": r, "p_value": p})

    return pd.DataFrame(results)


def plot_seed_vs_placement(df, out_path):
    """Scatter plot: seed (x) vs placement (y) with identity line, by weapon & gender."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, w in zip(axes, WEAPONS):
        wdf = df[df["weapon"] == w]
        for g in GENDERS:
            sub = wdf[wdf["gender"] == g]
            if len(sub) == 0:
                continue
            ax.scatter(sub["seed"], sub["place"],
                       alpha=0.3, s=15, label=f"{g} (n={len(sub)})",
                       marker=GENDER_MARKERS[g],
                       color="steelblue" if g == "M" else "coral")

        # Identity line
        max_val = max(wdf["seed"].max(), wdf["place"].max()) if len(wdf) > 0 else 100
        ax.plot([1, max_val], [1, max_val], "k--", alpha=0.4, linewidth=1, label="seed = place")

        # Correlation
        if len(wdf) >= 10:
            r, p = stats.spearmanr(wdf["seed"], wdf["place"])
            ax.set_title(f"{w.upper()}\nr={r:.3f}, p={p:.2e}, n={len(wdf)}", fontsize=12)
        else:
            ax.set_title(f"{w.upper()} (n={len(wdf)})", fontsize=12)

        ax.set_xlabel("Seed")
        ax.set_ylabel("Placement")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

    plt.suptitle("Seed vs. Placement at Div I NAC/Nationals (2024)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_seed_vs_placement_by_tournament(df, out_path):
    """Scatter with separate colors per tournament."""
    tournaments = sorted(df["tournament_name"].unique())
    cmap = plt.cm.get_cmap("tab10", len(tournaments))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, w in zip(axes, WEAPONS):
        wdf = df[df["weapon"] == w]
        for i, t in enumerate(tournaments):
            sub = wdf[wdf["tournament_name"] == t]
            if len(sub) == 0:
                continue
            short_name = t[:25] + "..." if len(t) > 28 else t
            ax.scatter(sub["seed"], sub["place"],
                       alpha=0.4, s=15, label=f"{short_name} ({len(sub)})",
                       color=cmap(i))

        max_val = max(wdf["seed"].max(), wdf["place"].max()) if len(wdf) > 0 else 100
        ax.plot([1, max_val], [1, max_val], "k--", alpha=0.4, linewidth=1)
        ax.set_title(f"{w.upper()} (n={len(wdf)})", fontsize=12)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Placement")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal")

    plt.suptitle("Seed vs. Placement by Tournament", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Approach 2: Regression ───────────────────────────────────────────────────

def run_regression(df):
    """OLS on log(placement) ~ seed + weapon + gender + weapon:gender.

    Returns (result_obj, summary_text).
    """
    rdf = df.copy()
    rdf["log_place"] = np.log(rdf["place"])
    rdf["log_seed"] = np.log(rdf["seed"])

    # Dummy variables
    rdf["weapon_foil"] = (rdf["weapon"] == "foil").astype(int)
    rdf["weapon_saber"] = (rdf["weapon"] == "saber").astype(int)
    rdf["gender_F"] = (rdf["gender"] == "F").astype(int)
    rdf["foil_F"] = rdf["weapon_foil"] * rdf["gender_F"]
    rdf["saber_F"] = rdf["weapon_saber"] * rdf["gender_F"]

    X = rdf[["log_seed", "weapon_foil", "weapon_saber", "gender_F", "foil_F", "saber_F"]]
    X = sm.add_constant(X)
    y = rdf["log_place"]

    try:
        model = sm.OLS(y, X)
        result = model.fit()
        summary_text = str(result.summary())
        return result, summary_text
    except Exception as e:
        return None, f"Regression failed: {e}"


def run_regression_by_slice(df):
    """Run simple OLS log(place) ~ log(seed) per weapon/gender slice. Returns summary dict."""
    results = []

    for w in WEAPONS:
        for g in GENDERS:
            sub = df[(df["weapon"] == w) & (df["gender"] == g)]
            if len(sub) < 20:
                continue

            X = sm.add_constant(np.log(sub["seed"]))
            y = np.log(sub["place"])
            try:
                res = sm.OLS(y, X).fit()
                results.append({
                    "weapon": w, "gender": g, "n": len(sub),
                    "r_squared": res.rsquared,
                    "seed_coef": res.params.iloc[1],
                    "seed_pvalue": res.pvalues.iloc[1],
                })
            except Exception:
                pass

    return pd.DataFrame(results) if results else pd.DataFrame()


# ── Approach 3: Upset Distance ───────────────────────────────────────────────

def compute_upset_distance(df):
    """Compute upset_distance = place - seed for each row."""
    df = df.copy()
    df["upset_distance"] = df["place"] - df["seed"]
    return df


def plot_upset_distance(df, out_path):
    """Violin/box plots of upset distance by weapon and gender."""
    df = compute_upset_distance(df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By weapon
    ax = axes[0]
    weapon_data = [df[df["weapon"] == w]["upset_distance"].values for w in WEAPONS]
    parts = ax.violinplot(weapon_data, positions=range(len(WEAPONS)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(WEAPON_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(WEAPONS)))
    ax.set_xticklabels([w.upper() for w in WEAPONS])
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Upset Distance (place - seed)")
    ax.set_title("Upset Distance by Weapon")

    # Add SD annotations
    for i, w in enumerate(WEAPONS):
        wdata = df[df["weapon"] == w]["upset_distance"]
        ax.text(i, ax.get_ylim()[1] * 0.9, f"SD={wdata.std():.1f}\nn={len(wdata)}",
                ha="center", fontsize=9)

    # By weapon x gender
    ax = axes[1]
    positions = []
    labels = []
    data_list = []
    colors = []
    pos = 0
    for w in WEAPONS:
        for g in GENDERS:
            sub = df[(df["weapon"] == w) & (df["gender"] == g)]
            if len(sub) > 0:
                data_list.append(sub["upset_distance"].values)
                positions.append(pos)
                labels.append(f"{w[:3].upper()}\n{g}")
                colors.append(WEAPON_COLORS[w])
                pos += 1
        pos += 0.5  # gap between weapons

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Upset Distance (place - seed)")
    ax.set_title("Upset Distance by Weapon & Gender")

    plt.suptitle("Upset Distance Distribution (place - seed)\nPositive = underperformed seed, Negative = overperformed",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_upset_histogram(df, out_path):
    """Overlaid histograms of upset distance by weapon."""
    df = compute_upset_distance(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.arange(df["upset_distance"].min() - 5, df["upset_distance"].max() + 5, 5)

    for w in WEAPONS:
        wdata = df[df["weapon"] == w]["upset_distance"]
        ax.hist(wdata, bins=bins, alpha=0.4, label=f"{w} (SD={wdata.std():.1f}, n={len(wdata)})",
                color=WEAPON_COLORS[w], edgecolor="white")

    ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="No change from seed")
    ax.set_xlabel("Upset Distance (place - seed)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Upset Distance by Weapon")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def run_levene_test(df):
    """Levene's test for equality of variances of upset_distance across weapons."""
    df = compute_upset_distance(df)
    groups = [df[df["weapon"] == w]["upset_distance"].values for w in WEAPONS if len(df[df["weapon"] == w]) >= 10]

    if len(groups) < 2:
        return None, None

    stat, p = stats.levene(*groups)
    return stat, p


# ── Approach 4: Top-N Survival ───────────────────────────────────────────────

def compute_topn_survival(df, thresholds=(8, 16, 32)):
    """For each threshold N: fraction of top-N seeds finishing in top-N (recall)
    and fraction of top-N finishers who were seeded top-N (precision).

    Returns DataFrame with columns: weapon, gender, threshold, recall, precision, n_events.
    """
    results = []

    slices = [("All", "All", df)]
    for w in WEAPONS:
        slices.append((w, "All", df[df["weapon"] == w]))
        for g in GENDERS:
            slices.append((w, g, df[(df["weapon"] == w) & (df["gender"] == g)]))
    for g in GENDERS:
        slices.append(("All", g, df[df["gender"] == g]))

    for weapon_label, gender_label, sub in slices:
        if len(sub) == 0:
            continue

        for n in thresholds:
            # Only consider events with field_size >= N
            valid_events = sub[sub["field_size"] >= n]
            if len(valid_events) == 0:
                continue

            top_seeded = valid_events[valid_events["seed"] <= n]
            top_finished = valid_events[valid_events["place"] <= n]

            # Recall: of top-N seeds, how many finished top-N?
            if len(top_seeded) > 0:
                recall = (top_seeded["place"] <= n).mean()
            else:
                recall = np.nan

            # Precision: of top-N finishers, how many were seeded top-N?
            if len(top_finished) > 0:
                precision = (top_finished["seed"] <= n).mean()
            else:
                precision = np.nan

            n_events = valid_events["event_id"].nunique()

            results.append({
                "weapon": weapon_label, "gender": gender_label,
                "threshold": n,
                "recall": recall, "precision": precision,
                "n_seeds": len(top_seeded), "n_finishers": len(top_finished),
                "n_events": n_events,
            })

    return pd.DataFrame(results)


def plot_topn_survival(survival_df, out_path):
    """Grouped bar chart: top-N survival by weapon for each threshold."""
    thresholds = sorted(survival_df["threshold"].unique())
    weapon_rows = survival_df[(survival_df["gender"] == "All") & (survival_df["weapon"].isin(WEAPONS))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in [(axes[0], "recall", "Recall: Top-N Seeds Finishing Top-N"),
                               (axes[1], "precision", "Precision: Top-N Finishers Seeded Top-N")]:
        x = np.arange(len(thresholds))
        width = 0.25

        for i, w in enumerate(WEAPONS):
            wdata = weapon_rows[weapon_rows["weapon"] == w]
            vals = []
            for t in thresholds:
                row = wdata[wdata["threshold"] == t]
                vals.append(row[metric].values[0] * 100 if len(row) > 0 and not np.isnan(row[metric].values[0]) else 0)
            bars = ax.bar(x + i * width, vals, width, label=w.upper(),
                          color=WEAPON_COLORS[w], edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Threshold (N)")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"Top {t}" for t in thresholds])
        ax.legend()
        ax.set_ylim(0, 105)

    plt.suptitle("Top-N Seed Survival Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_topn_by_gender(survival_df, out_path):
    """Top-N recall by weapon and gender."""
    thresholds = sorted(survival_df["threshold"].unique())

    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5))
    if len(thresholds) == 1:
        axes = [axes]

    for ax, t in zip(axes, thresholds):
        sub = survival_df[(survival_df["threshold"] == t) & (survival_df["weapon"].isin(WEAPONS)) &
                          (survival_df["gender"].isin(GENDERS))]

        x = np.arange(len(WEAPONS))
        width = 0.35
        for i, g in enumerate(GENDERS):
            gdata = sub[sub["gender"] == g]
            vals = []
            for w in WEAPONS:
                row = gdata[gdata["weapon"] == w]
                vals.append(row["recall"].values[0] * 100 if len(row) > 0 and not np.isnan(row["recall"].values[0]) else 0)
            color = "steelblue" if g == "M" else "coral"
            bars = ax.bar(x + i * width, vals, width, label=g, color=color, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([w.upper() for w in WEAPONS])
        ax.set_ylabel("Recall (%)")
        ax.set_title(f"Top-{t} Recall by Gender")
        ax.legend()
        ax.set_ylim(0, 105)

    plt.suptitle("Top-N Seed Survival by Gender", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Summary Writer ───────────────────────────────────────────────────────────

def write_findings(df, corr_df, regression_summary, slice_reg_df,
                   levene_stat, levene_p, survival_df, out_path):
    """Write findings_summary.txt synthesizing all four approaches."""
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS PART 2: SEED VS. PLACEMENT FINDINGS")
    lines.append("Div I NAC/Nationals Events, 2024")
    lines.append("=" * 70)
    lines.append("")

    # Data overview
    lines.append("DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total observations: {len(df)}")
    lines.append(f"Unique fencers: {df['fencer_id'].nunique()}")
    lines.append(f"Unique events (event_id): {df['event_id'].nunique()}")
    lines.append(f"Tournaments: {df['tournament_name'].nunique()}")
    lines.append("")
    lines.append("Counts by weapon x gender:")
    ct = pd.crosstab(df["weapon"], df["gender"], margins=True)
    lines.append(ct.to_string())
    lines.append("")
    lines.append("Tournaments represented:")
    for t in sorted(df["tournament_name"].unique()):
        n = len(df[df["tournament_name"] == t])
        lines.append(f"  {t}: {n} rows")
    lines.append("")

    # Approach 1: Correlation
    lines.append("=" * 70)
    lines.append("APPROACH 1: SPEARMAN RANK CORRELATION (seed vs. placement)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Higher r = more predictable (seed correlates strongly with placement)")
    lines.append("")
    lines.append(corr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")

    # Key findings from correlation
    weapon_corrs = corr_df[(corr_df["gender"] == "All") & (corr_df["weapon"].isin(WEAPONS))]
    if len(weapon_corrs) > 0:
        most_pred = weapon_corrs.loc[weapon_corrs["spearman_r"].idxmax()]
        least_pred = weapon_corrs.loc[weapon_corrs["spearman_r"].idxmin()]
        lines.append(f"Most predictable weapon: {most_pred['weapon']} (r={most_pred['spearman_r']:.3f})")
        lines.append(f"Least predictable weapon: {least_pred['weapon']} (r={least_pred['spearman_r']:.3f})")
    lines.append("")

    # Approach 2: Regression
    lines.append("=" * 70)
    lines.append("APPROACH 2: OLS REGRESSION — log(placement) ~ log(seed) + weapon + gender + interactions")
    lines.append("=" * 70)
    lines.append("")
    lines.append(regression_summary)
    lines.append("")

    if len(slice_reg_df) > 0:
        lines.append("Per-slice R-squared (log(place) ~ log(seed)):")
        lines.append(slice_reg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        most_r2 = slice_reg_df.loc[slice_reg_df["r_squared"].idxmax()]
        least_r2 = slice_reg_df.loc[slice_reg_df["r_squared"].idxmin()]
        lines.append(f"Highest R-squared: {most_r2['weapon']} {most_r2['gender']} (R2={most_r2['r_squared']:.3f})")
        lines.append(f"Lowest R-squared: {least_r2['weapon']} {least_r2['gender']} (R2={least_r2['r_squared']:.3f})")
    lines.append("")

    # Approach 3: Upset Distance
    lines.append("=" * 70)
    lines.append("APPROACH 3: UPSET DISTANCE (place - seed)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Positive = underperformed seed, Negative = overperformed seed")
    lines.append("Larger SD = more volatile/unpredictable")
    lines.append("")

    ud = compute_upset_distance(df)
    for w in WEAPONS:
        wdata = ud[ud["weapon"] == w]["upset_distance"]
        lines.append(f"  {w.upper():8s}: mean={wdata.mean():+.1f}, SD={wdata.std():.1f}, "
                      f"median={wdata.median():+.0f}, n={len(wdata)}")
    lines.append("")

    for w in WEAPONS:
        for g in GENDERS:
            sub = ud[(ud["weapon"] == w) & (ud["gender"] == g)]["upset_distance"]
            if len(sub) > 0:
                lines.append(f"  {w.upper():8s} {g}: mean={sub.mean():+.1f}, SD={sub.std():.1f}, "
                              f"median={sub.median():+.0f}, n={len(sub)}")
    lines.append("")

    if levene_stat is not None:
        lines.append(f"Levene's test for equality of variances across weapons:")
        lines.append(f"  W={levene_stat:.3f}, p={levene_p:.4f}")
        if levene_p < 0.05:
            lines.append("  -> Significant: volatility differs across weapons")
        else:
            lines.append("  -> Not significant: no evidence of different volatility across weapons")
    lines.append("")

    # Approach 4: Top-N Survival
    lines.append("=" * 70)
    lines.append("APPROACH 4: TOP-N SEED SURVIVAL")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Recall: of top-N seeds, what fraction finished top-N?")
    lines.append("Precision: of top-N finishers, what fraction were seeded top-N?")
    lines.append("")

    # Overall by weapon
    overall = survival_df[(survival_df["gender"] == "All") & (survival_df["weapon"].isin(WEAPONS + ["All"]))]
    lines.append(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    # By weapon x gender
    by_wg = survival_df[(survival_df["gender"].isin(GENDERS)) & (survival_df["weapon"].isin(WEAPONS))]
    if len(by_wg) > 0:
        lines.append("By weapon x gender:")
        lines.append(by_wg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    # Synthesis
    lines.append("=" * 70)
    lines.append("SYNTHESIS")
    lines.append("=" * 70)
    lines.append("")

    # Determine most/least predictable weapon
    if len(weapon_corrs) > 0:
        ranked = weapon_corrs.sort_values("spearman_r", ascending=False)
        lines.append("Predictability ranking (by Spearman correlation):")
        for _, row in ranked.iterrows():
            lines.append(f"  {row['weapon'].upper():8s}: r={row['spearman_r']:.3f} (n={int(row['n'])})")
        lines.append("")

    lines.append("Key questions answered:")
    lines.append("")

    lines.append("Q: Does initial seed predict final placement?")
    overall_r = corr_df[(corr_df["weapon"] == "All") & (corr_df["gender"] == "All")]
    if len(overall_r) > 0:
        r_val = overall_r.iloc[0]["spearman_r"]
        lines.append(f"A: Yes. Overall Spearman r = {r_val:.3f}. Seeds are meaningfully predictive of placement,")
        lines.append("   though substantial variation remains (many upsets and underperformances).")
    lines.append("")

    lines.append("Q: Does predictability vary by weapon?")
    if len(weapon_corrs) > 0:
        r_range = weapon_corrs["spearman_r"].max() - weapon_corrs["spearman_r"].min()
        lines.append(f"A: The spread in Spearman r across weapons is {r_range:.3f}.")
        most = weapon_corrs.loc[weapon_corrs["spearman_r"].idxmax()]
        least = weapon_corrs.loc[weapon_corrs["spearman_r"].idxmin()]
        lines.append(f"   {most['weapon'].upper()} is most predictable (r={most['spearman_r']:.3f}),")
        lines.append(f"   {least['weapon'].upper()} is least predictable (r={least['spearman_r']:.3f}).")
        if levene_p is not None and levene_p < 0.05:
            lines.append("   Levene's test confirms significantly different volatility across weapons.")
        elif levene_p is not None:
            lines.append("   However, Levene's test does not find significant variance differences.")
    lines.append("")

    lines.append("Q: Are there gender differences in predictability?")
    gender_corrs = corr_df[(corr_df["weapon"] == "All") & (corr_df["gender"].isin(GENDERS))]
    if len(gender_corrs) > 0:
        for _, row in gender_corrs.iterrows():
            lines.append(f"   {row['gender']}: r={row['spearman_r']:.3f} (n={int(row['n'])})")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote findings to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ANALYSIS PART 2: Seed vs. Placement")
    print("=" * 60)
    print()

    df = load_data()
    print()

    # Approach 1: Correlation
    print("Approach 1: Computing correlations...")
    corr_df = compute_correlations(df)
    corr_df.to_csv(os.path.join(BASE_DIR, "correlation_table.csv"), index=False)
    print(f"  Saved correlation_table.csv")
    print()

    print("Approach 1: Generating scatter plots...")
    plot_seed_vs_placement(df, os.path.join(BASE_DIR, "seed_vs_placement.png"))
    plot_seed_vs_placement_by_tournament(df, os.path.join(BASE_DIR, "seed_vs_placement_by_tournament.png"))
    print()

    # Approach 2: Regression
    print("Approach 2: Running regression...")
    reg_result, reg_summary = run_regression(df)
    slice_reg_df = run_regression_by_slice(df)
    if len(slice_reg_df) > 0:
        slice_reg_df.to_csv(os.path.join(BASE_DIR, "regression_by_slice.csv"), index=False)
        print(f"  Saved regression_by_slice.csv")
    print()

    # Approach 3: Upset Distance
    print("Approach 3: Upset distance analysis...")
    plot_upset_distance(df, os.path.join(BASE_DIR, "upset_distance.png"))
    plot_upset_histogram(df, os.path.join(BASE_DIR, "upset_distance_histogram.png"))
    levene_stat, levene_p = run_levene_test(df)
    print()

    # Approach 4: Top-N Survival
    print("Approach 4: Top-N survival analysis...")
    survival_df = compute_topn_survival(df)
    survival_df.to_csv(os.path.join(BASE_DIR, "topn_survival.csv"), index=False)
    print(f"  Saved topn_survival.csv")
    plot_topn_survival(survival_df, os.path.join(BASE_DIR, "topn_survival.png"))
    plot_topn_by_gender(survival_df, os.path.join(BASE_DIR, "topn_by_gender.png"))
    print()

    # Write findings summary
    print("Writing findings summary...")
    write_findings(df, corr_df, reg_summary, slice_reg_df,
                   levene_stat, levene_p, survival_df,
                   os.path.join(BASE_DIR, "findings_summary.txt"))
    print()

    # Final count
    png_count = len([f for f in os.listdir(BASE_DIR) if f.endswith(".png")])
    csv_count = len([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")])
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  PNGs: {png_count}")
    print(f"  CSVs: {csv_count}")
    print(f"  Summary: findings_summary.txt")


if __name__ == "__main__":
    main()
