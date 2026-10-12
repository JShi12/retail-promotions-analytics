"""Phase 2 figures. Style: blue / orange / aqua in fixed order, grey for reference marks,
thin marks, light grid, no top/right spines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, MUTED = "#1f1f1f", "#6b6b6b"


def _style(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)


def _title(ax, title: str, subtitle: str) -> None:
    """Title and subtitle at fixed point offsets above the axes, aligned to the figure's left edge."""
    pos = ("figure fraction", "axes fraction")
    ax.annotate(title, xy=(0.01, 1), xycoords=pos, xytext=(0, 30), textcoords="offset points",
                ha="left", fontsize=12, fontweight="semibold", color=INK)  # fmt: skip
    ax.annotate(subtitle, xy=(0.01, 1), xycoords=pos, xytext=(0, 14), textcoords="offset points",
                ha="left", fontsize=9, color=MUTED)  # fmt: skip


def _fig_title(fig, title: str, subtitle: str, wspace: float = 0.3) -> None:
    """Title and subtitle for multi-panel figures, at fixed inch offsets from the top edge."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.28 / h, title, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 1 - 0.52 / h, subtitle, ha="left", fontsize=9, color=MUTED)
    fig.subplots_adjust(top=1 - 1.0 / h, wspace=wspace, left=0.07, right=0.97, bottom=0.16)


def _pct(x, _=None) -> str:
    return f"{x:+.0%}" if x != 0 else "0%"


def save(fig, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return path


def raw_vs_model(raw: pd.DataFrame, results: pd.DataFrame):
    """Naive mailer-only vs no-promotion contrast next to the fixed-effects mailer-week lift."""
    naive = raw.set_index("status").loc["mailer only", "ratio_to_none"] - 1
    m = results.query("model == 'product' and scope == 'pooled' and term == 'mailer_share'").iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    labels = [
        "Raw: mailer-only weeks vs\nunpromoted weeks",
        "Model: product + category-week\nfixed effects",
    ]
    ax.barh([1, 0], [naive, m.lift], height=0.45, color=[GREY, BLUE])
    ax.errorbar(
        m.lift,
        0,
        xerr=[[m.lift - m.lift_low], [m.lift_high - m.lift]],
        fmt="none",
        ecolor=INK,
        lw=1,
        capsize=3,
    )
    ax.text(naive, 1, f"  {naive:+.0%}", va="center", fontsize=9, color=INK)
    ax.text(
        m.lift_high,
        0,
        f"  {m.lift:+.0%} ({m.lift_low:+.0%} to {m.lift_high:+.0%})",
        va="center",
        fontsize=9,
        color=INK,
    )
    ax.set_yticks([1, 0], labels, fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct))
    ax.set_xlim(0, naive * 1.25)
    _style(ax)
    _title(ax, "Comparing raw averages overstates the mailer lift about fivefold",
           "% change in weekly units in mailer weeks; bar whisker = 95% confidence interval")  # fmt: skip
    return fig


def _forest(
    ax, df: pd.DataFrame, value: str, low: str, high: str, ref: float, log_multiplier: bool
) -> None:
    df = df.sort_values(value)
    y = np.arange(len(df))
    if log_multiplier:  # plot units multiplier (1 + lift) on a log axis, label as % lift
        v, lo, hi = 1 + df[value], 1 + df[low], 1 + df[high]
        ax.set_xscale("log")
        ax.axvline(1, color=GREY, lw=0.8)
        ax.axvline(1 + ref, color=GREY, lw=0.8, ls="--")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _pct(x - 1)))
        ax.set_xticks([1, 1.5, 2, 3, 5, 10])
    else:
        v, lo, hi = df[value], df[low], df[high]
        ax.axvline(ref, color=GREY, lw=0.8, ls="--")
        ax.xaxis.set_major_formatter(FuncFormatter(_pct))
    ax.hlines(y, lo, hi, color=BLUE, lw=1.2)
    ax.plot(v, y, "o", color=BLUE, ms=5)
    ax.set_yticks(y, [c.title() for c in df["category"]], fontsize=9, color=INK)
    _style(ax)


def mailer_by_category(results: pd.DataFrame):
    cat = results.query("model == 'product' and scope == 'category' and term == 'mailer_share'")
    pooled = (
        results.query("model == 'product' and scope == 'pooled' and term == 'mailer_share'")
        .iloc[0]
        .lift
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    _forest(ax, cat, "lift", "lift_low", "lift_high", pooled, log_multiplier=True)
    ax.text(1 + pooled, len(cat) - 0.4, f" all products {pooled:+.0%}", fontsize=8, color=MUTED)
    _title(ax, "Mailer-week lift ranges from none detectable (beer) to several-fold (fresh meat, milk)",
           "% change in weekly units in mailer weeks, 95% CI; log scale; dashed = pooled over all products")  # fmt: skip
    return fig


def display_by_category(results: pd.DataFrame):
    cat = results.query("model == 'product' and scope == 'category' and term == 'display_share'")
    pooled = (
        results.query("model == 'product' and scope == 'pooled' and term == 'display_share'")
        .iloc[0]
        .lift
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    _forest(ax, cat, "lift", "lift_low", "lift_high", pooled, log_multiplier=False)
    ax.text(pooled, len(cat) - 0.4, f" all products {pooled:+.0%}", fontsize=8, color=MUTED)
    _title(ax, "In the stores that have it, a display goes with 2.8x to 4.6x normal sales",
           "Approximate % lift in the stores that have the display, 95% CI; dashed = pooled over all products")  # fmt: skip
    return fig


def product_vs_category(results: pd.DataFrame):
    prod = results.query("model == 'product' and scope == 'selected'").set_index("term")
    cat = results.query("model == 'category_net'").set_index("term")
    groups = [("Mailer", prod.loc["mailer_share"], cat.loc["cat_mailer_share"]),
              ("Display", prod.loc["display_share"], cat.loc["cat_display_share"])]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (_, p, c) in enumerate(groups):
        for j, (row, colour) in enumerate([(p, BLUE), (c, ORANGE)]):
            val = row.lift
            ax.bar(i + (j - 0.5) * 0.36, val, width=0.32, color=colour)
            ax.errorbar(i + (j - 0.5) * 0.36, val, yerr=[[val - row.lift_low], [row.lift_high - val]],
                        fmt="none", ecolor=INK, lw=1, capsize=3)  # fmt: skip
            ax.text(
                i + (j - 0.5) * 0.36,
                row.lift_high,
                f"{val:+.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks([0, 1], [g[0] for g in groups], fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.bar(0, 0, color=BLUE, label="Product level (pooled product model)")
    ax.bar(0, 0, color=ORANGE, label="Category level (category totals, net of switching)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, grid_axis="y")
    _title(ax, "Category totals point to a smaller mailer lift than product-level estimates",
           "Selected 20 categories; approximate % lift if all sales were promoted; 95% CI (the mailer intervals overlap)")  # fmt: skip
    return fig


def timing(results: pd.DataFrame):
    t = results.query("model == 'timing'").set_index("term")
    order = [
        ("Week before\nthe mailer", "mailer_lead"),
        ("Mailer week", "mailer_share"),
        ("Week after\nthe mailer", "mailer_lag"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for x, (_, term) in enumerate(order):
        r = t.loc[term]
        colour = BLUE if term == "mailer_share" else GREY
        ax.vlines(x, r.lift_low, r.lift_high, color=colour, lw=1.2)
        ax.plot(x, r.lift, "o", color=colour, ms=6)
        ax.text(x + 0.08, r.lift, f"{r.lift:+.1%}", va="center", fontsize=8, color=INK)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(3), [o[0] for o in order], fontsize=9, color=INK)
    ax.set_xlim(-0.5, 2.6)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    _style(ax, grid_axis="y")
    _title(ax, "Sales are slightly higher the week before and after a mailer, not lower",
           "% change in weekly units associated with a mailer in each week, 95% CI (one model)")  # fmt: skip
    return fig


# ---------------------------------------------------------------- Phase 3

SEGMENT_COLOURS = [BLUE, ORANGE, AQUA]


def _seg_ticks(profiles: pd.DataFrame) -> list[str]:
    return [
        f"{s}. {lab}" for s, lab in zip(profiles["segment"], profiles["segment_label"], strict=True)
    ]


def k_selection(summary: pd.DataFrame, chosen: int):
    """Silhouette against the shuffled baseline, and the least stable segment, for each k."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(summary["k"], summary["silhouette"], "o-", color=BLUE, lw=1.5, ms=5, label="Segments")
    a1.plot(
        summary["k"],
        summary["shuffled_silhouette"],
        "o--",
        color=GREY,
        lw=1.2,
        ms=4,
        label="Shuffled features",
    )
    a1.set_xticks(summary["k"])
    a1.set_xlabel("Number of segments (k)", fontsize=9, color=MUTED)
    a1.set_ylabel("Silhouette", fontsize=9, color=MUTED)
    a1.legend(frameon=False, fontsize=8)
    _style(a1, grid_axis="y")
    colours = [BLUE if k == chosen else GREY for k in summary["k"]]
    a2.bar(summary["k"], summary["min_segment_jaccard"], color=colours, width=0.6)
    a2.axhline(0.75, color=INK, lw=0.8, ls="--")
    a2.text(summary["k"].max() + 0.45, 0.75, "stable\n(0.75)", va="center", fontsize=8, color=MUTED)
    a2.set_xlabel("Number of segments (k)", fontsize=9, color=MUTED)
    a2.set_ylabel("Least stable segment (Jaccard)", fontsize=9, color=MUTED)
    a2.set_ylim(0, 1)
    _style(a2, grid_axis="y")
    _fig_title(fig, f"k = {chosen} gives the clearest gain over shuffled data with every segment stable",
               "Left: silhouette vs the same data with each feature shuffled. Right: lowest bootstrap Jaccard among the segments")  # fmt: skip
    return fig


def segment_centroids(centroids: pd.DataFrame):
    """Standardised centre of each segment on the three clustering inputs."""
    if len(centroids) > len(SEGMENT_COLOURS):
        raise ValueError("Plot style allows at most 3 colours; facet the chart for more segments.")
    names = {
        "deal_index": "Deal-proneness index",
        "pl_adjusted": "Own label (adjusted)",
        "log_coupon": "Coupon use",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(names))
    for i, row in centroids.reset_index(drop=True).iterrows():
        vals = [row[c] for c in names]
        ax.plot(vals, y + (i - 1) * 0.18, "o", color=SEGMENT_COLOURS[i], ms=7,
                label=f"{row['segment']}. {row['segment_label']}")  # fmt: skip
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y, list(names.values()), fontsize=9, color=INK)
    ax.set_xlabel("Standard deviations from the average household", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    _style(ax)
    _title(ax, "Each segment is defined by where it sits on three behaviours",
           "Segment centre on each clustering input, standardised (0 = average household)")  # fmt: skip
    return fig


def actual_vs_expected(profiles: pd.DataFrame):
    """Median actual vs expected (same categories, same stores) for each behaviour and segment."""
    panels = [
        ("promo", "Promotion share of spend"),
        ("disc", "Discount (% off shelf price)"),
        ("pl", "Own-label share of spend"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    x = np.arange(len(profiles))
    for ax, (m, label) in zip(axes, panels, strict=True):
        ax.bar(x - 0.18, profiles[f"{m}_expected"], width=0.34, color=GREY, label="Expected")
        ax.bar(x + 0.18, profiles[f"{m}_actual"], width=0.34, color=BLUE, label="Actual")
        for xi, v in zip(x, profiles[f"{m}_actual"], strict=True):
            ax.text(xi + 0.18, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, [f"Seg {s}" for s in profiles["segment"]], fontsize=8.5, color=INK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        _style(ax, grid_axis="y")
    h = fig.get_figheight()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.97, 1 - 0.38 / h),
    )
    _fig_title(
        fig,
        "Segments differ from what their own categories and stores would predict",
        "Median household; expected = same categories in the same stores, other households' rates",
    )
    return fig


def size_and_redemption(profiles: pd.DataFrame):
    """Share of households and of spend per segment, and campaign coupon redemption rate."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(profiles))
    hh_share = profiles["households"] / profiles["households"].sum()
    a1.bar(x - 0.18, hh_share, width=0.34, color=BLUE, label="Households")
    a1.bar(x + 0.18, profiles["share_of_spend"], width=0.34, color=ORANGE, label="Spend")
    for xi, h, s in zip(x, hh_share, profiles["share_of_spend"], strict=True):
        a1.text(xi - 0.18, h, f"{h:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
        a1.text(xi + 0.18, s, f"{s:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_xticks(x, [f"Seg {s}" for s in profiles["segment"]], fontsize=8.5, color=INK)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Share of households and of spend", fontsize=9, color=INK, loc="left")
    _style(a1, grid_axis="y")
    a2.bar(x, profiles["redemption_rate"], width=0.5, color=BLUE)
    for xi, r, n in zip(x, profiles["redemption_rate"], profiles["campaign_pairs"], strict=True):
        a2.text(xi, r, f"{r:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=7.5, color=INK)
    a2.set_xticks(x, [f"Seg {s}" for s in profiles["segment"]], fontsize=8.5, color=INK)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_ylim(0, profiles["redemption_rate"].max() * 1.35)
    a2.set_title(
        "Campaign coupon redemption rate (household-campaign pairs)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    _style(a2, grid_axis="y")
    _fig_title(
        fig,
        "The coupon-using segment redeems campaign coupons about five times as often",
        "Households with at least 10 baskets; redemption is descriptive, not a clustering input",
    )
    return fig
