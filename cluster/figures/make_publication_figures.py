import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CSV = ROOT / "commuting-regression" / "data_swindon_with_commute.csv"
GEO = ROOT / "data preprocessing+EDA" / "Shi" / "LSOA_boundaries.geojson"
GVA = "log_total_GVA_2023"


def save_pub(fig, stem, dpi=600):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    fig.savefig(HERE / f"{stem}.tiff", dpi=dpi, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


def load():
    df = pd.read_csv(CSV)
    features = [
        "log_voa_rv_2023", "rv_per_working_age", "sme_density", "qualification_index",
        "firm_size_diversity", "rv_per_employee", "sme_qual_interaction",
        "employment_quality", "modern_sector_leverage", "asset_growth_diversity",
        "msoa_out_commute_share", "msoa_same_msoa_work_share", "msoa_wfh_share",
        "msoa_in_commute_share", "msoa_local_worker_share",
    ]
    X = StandardScaler().fit_transform(df[features])
    df["cluster"] = KMeans(n_clusters=6, random_state=42, n_init=20).fit_predict(X)
    return df


def palette(df):
    order = df.groupby("cluster")[GVA].mean().sort_values().index.tolist()
    cmap = plt.cm.viridis(np.linspace(0.08, 0.95, len(order)))
    color = {cl: cmap[i] for i, cl in enumerate(order)}
    return order, color


def stats_block(df):
    groups = [g[GVA].values for _, g in df.groupby("cluster")]
    grand = df[GVA]
    ss_b = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    eta2 = ss_b / ((grand - grand.mean()) ** 2).sum()
    f, p_a = stats.f_oneway(*groups)
    h, p_k = stats.kruskal(*groups)
    return eta2, f, p_a, h, p_k


def p_txt(p):
    return "P < 0.001" if p < 1e-3 else f"P = {p:.3f}"


def fig_separation(df, order, color):
    eta2, f, p_a, h, p_k = stats_block(df)
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})

    xs = np.linspace(df[GVA].min() - 0.4, df[GVA].max() + 0.4, 400)
    top = list(reversed(order))
    overlap = 1.5
    gtrans = mtransforms.blended_transform_factory(axA.transAxes, axA.transData)
    for i, cl in enumerate(top):
        v = df.loc[df.cluster == cl, GVA].values
        kde = gaussian_kde(v)
        y = kde(xs)
        ymax = y.max()
        y = y / ymax
        base = i * (1 / overlap)
        axA.fill_between(xs, base, base + y, color=color[cl], alpha=0.85,
                         lw=0.8, edgecolor="white", zorder=2 * (len(top) - i))
        mh = float(kde(v.mean())[0]) / ymax
        axA.plot([v.mean(), v.mean()], [base, base + mh],
                 color="white", lw=0.9, zorder=2 * (len(top) - i) + 1)
        axA.text(-0.015, base + 0.04, f"Cluster {cl}  (n={len(v)})",
                 transform=gtrans, va="bottom", ha="right", fontsize=6.5, color="0.15")
    axA.set_xlim(xs[0], xs[-1])
    axA.set_yticks([])
    axA.set_xlabel("log total GVA, 2023")
    axA.spines["left"].set_visible(False)
    axA.set_title("a", loc="left", fontweight="bold", fontsize=9)
    axA.margins(y=0.02)

    pos = list(range(len(order)))
    parts = axB.violinplot([df.loc[df.cluster == cl, GVA].values for cl in order],
                           positions=pos, showextrema=False, widths=0.85)
    for b, cl in zip(parts["bodies"], order):
        b.set_facecolor(color[cl])
        b.set_alpha(0.35)
        b.set_edgecolor(color[cl])
        b.set_linewidth(0.8)
    for i, cl in enumerate(order):
        v = df.loc[df.cluster == cl, GVA].values
        jit = np.random.default_rng(cl).normal(0, 0.05, len(v))
        axB.scatter(i + jit, v, s=5, color=color[cl], edgecolor="white",
                    linewidth=0.2, zorder=3)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        axB.plot([i, i], [q1, q3], color="0.2", lw=2.2, solid_capstyle="round", zorder=4)
        axB.scatter(i, med, s=14, color="white", edgecolor="0.2", linewidth=0.6, zorder=5)
    axB.set_xticks(pos)
    axB.set_xticklabels([f"C{cl}" for cl in order])
    axB.set_xlabel("Cluster (ordered by mean GVA)")
    axB.set_ylabel("log total GVA, 2023")
    axB.set_title("b", loc="left", fontweight="bold", fontsize=9)
    txt = (r"$\eta^2$ = " + f"{eta2:.3f}\n"
           f"ANOVA: F = {f:.1f}, {p_txt(p_a)}\n"
           f"Kruskal–Wallis: H = {h:.1f}, {p_txt(p_k)}")
    axB.text(0.02, 0.98, txt, transform=axB.transAxes, va="top", ha="left",
             fontsize=6.2, linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.6))

    fig.tight_layout(w_pad=2.0)
    save_pub(fig, "fig1_cluster_gva_separation")
    plt.close(fig)
    return eta2


def fig_overall_decomposition(df, order, color):
    bins = np.histogram_bin_edges(df[GVA], bins=22)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)

    axA.hist(df[GVA], bins=bins, color="0.6", edgecolor="white", linewidth=0.5)
    axA.set_xlabel("log total GVA, 2023")
    axA.set_ylabel("Number of LSOAs")
    axA.set_title("a", loc="left", fontweight="bold", fontsize=9)
    axA.text(0.97, 0.95, "All LSOAs", transform=axA.transAxes, ha="right", va="top",
             fontsize=7, color="0.3")

    stack_order = list(order)
    data = [df.loc[df.cluster == cl, GVA].values for cl in stack_order]
    cols = [color[cl] for cl in stack_order]
    axB.hist(data, bins=bins, stacked=True, color=cols,
             edgecolor="white", linewidth=0.3)
    axB.set_xlabel("log total GVA, 2023")
    axB.set_title("b", loc="left", fontweight="bold", fontsize=9)

    means = df.groupby("cluster")[GVA].mean()
    handles = [Patch(facecolor=color[cl], edgecolor="white",
                     label=f"Cluster {cl} (n={int((df.cluster == cl).sum())})")
               for cl in reversed(order)]
    axB.legend(handles=handles, loc="upper right", fontsize=6, handlelength=1.0,
               handleheight=1.0, labelspacing=0.35, borderpad=0.5,
               title="Higher → lower GVA", title_fontsize=6.4)

    fig.tight_layout(w_pad=1.6)
    save_pub(fig, "fig3_gva_overall_vs_clusters")
    plt.close(fig)


def fig_map(df, order, color):
    geo = gpd.read_file(GEO)[["LSOA21CD", "geometry"]]
    m = geo.merge(df[["LSOA21CD", "cluster"]], on="LSOA21CD", how="inner").to_crs(27700)
    means = df.groupby("cluster")[GVA].mean()

    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    for cl in order:
        m[m.cluster == cl].plot(ax=ax, color=color[cl], edgecolor="white", linewidth=0.35)
    ax.set_axis_off()
    ax.set_aspect("equal")

    high_to_low = list(reversed(order))
    handles = [Patch(facecolor=color[cl], edgecolor="white",
                     label=f"Cluster {cl}  (n={int((df.cluster == cl).sum())}, "
                           f"mean log GVA {means[cl]:.2f})")
               for cl in high_to_low]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                    fontsize=6.2, title="Higher  →  lower GVA", title_fontsize=6.8,
                    handlelength=1.1, handleheight=1.1, borderpad=0.6, labelspacing=0.5,
                    frameon=True)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_edgecolor("0.8")
    leg.get_frame().set_linewidth(0.6)

    minx, miny, maxx, maxy = m.total_bounds
    bar = 2000
    x0 = maxx - bar - (maxx - minx) * 0.04
    y0 = miny + (maxy - miny) * 0.03
    ax.plot([x0, x0 + bar], [y0, y0], color="0.15", lw=1.6, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + (maxy - miny) * 0.012, "2 km",
            ha="center", va="bottom", fontsize=6)

    nx = maxx - (maxx - minx) * 0.05
    ny = maxy - (maxy - miny) * 0.01
    ax.annotate("N", xy=(nx, ny), xytext=(nx, ny - (maxy - miny) * 0.065),
                ha="center", va="center", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.2))

    ax.set_title("Swindon LSOA economic clusters (k = 6)", fontsize=8, pad=6)
    fig.tight_layout()
    save_pub(fig, "fig2_cluster_map")
    plt.close(fig)


def main():
    df = load()
    order, color = palette(df)
    eta2 = fig_separation(df, order, color)
    fig_overall_decomposition(df, order, color)
    fig_map(df, order, color)
    print(f"eta^2 = {eta2:.3f}")
    print("Saved to", HERE)
    for f in sorted(HERE.glob("fig*.*")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
