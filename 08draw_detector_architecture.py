"""
Draw the keypoint detector architecture (RPN_Detector) for thesis / slides.
Matches USIP-master/models/networks.py + layers.py (GeneralKNNFusionModule SE gate).

Usage:
  python draw_detector_architecture.py
Outputs:
  figures/usip_detector_architecture.png
  figures/usip_detector_architecture.pdf
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- style ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["mathtext.fontset"] = "stix"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

COLOR_IN = "#f0f4ff"
COLOR_PRE = "#d6e8ff"
COLOR_FEAT = "#c8e6c9"
COLOR_SE = "#fff3b0"
COLOR_HEAD = "#ffd9b3"
COLOR_LOSS = "#ffe0cc"
COLOR_EDGE = "#222222"
COLOR_TEXT = "#111111"


def rounded_box(ax, x, y, w, h, fc, ec=COLOR_EDGE, lw=1.2, rad=0.08):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rad}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, lw=1.0):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        facecolor=COLOR_EDGE,
        edgecolor=COLOR_EDGE,
    )
    ax.add_patch(a)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig_w, fig_h = 18.0, 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    y_mid = 3.75
    # ---- Title ----
    ax.text(
        9.0,
        7.05,
        "Keypoint Detector Network Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        9.0,
        6.55,
        r"Shared weights for both views; batch pairs $(\mathbf{X},\,\mathbf{X}')$",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#444444",
    )

    # ---- Column positions (left -> right) ----
    # 1 Inputs
    w_in, h_in = 1.35, 0.85
    x_in = 0.35
    y_top = y_mid + 1.15
    y_bot = y_mid - 1.15
    rounded_box(ax, x_in, y_top, w_in, h_in, COLOR_IN)
    ax.text(x_in + w_in / 2, y_top + h_in / 2, r"$\mathbf{X}$", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(x_in + w_in / 2, y_top - 0.35, r"$\mathbb{R}^{3\times N}$", ha="center", va="top", fontsize=9)

    rounded_box(ax, x_in, y_bot, w_in, h_in, COLOR_IN)
    ax.text(x_in + w_in / 2, y_bot + h_in / 2, r"$\mathbf{X}'$", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(x_in + w_in / 2, y_bot - 0.35, r"$\mathbb{R}^{3\times N}$", ha="center", va="top", fontsize=9)

    # merge arrow
    merge_x = 2.15
    arrow(ax, x_in + w_in, y_top + h_in / 2, merge_x - 0.05, y_mid + 0.35)
    arrow(ax, x_in + w_in, y_bot + h_in / 2, merge_x - 0.05, y_mid - 0.35)
    ax.plot([merge_x - 0.05, merge_x - 0.05], [y_mid + 0.35, y_mid - 0.35], color=COLOR_EDGE, lw=1)
    arrow(ax, merge_x - 0.05, y_mid, merge_x + 0.02, y_mid)

    # 2 FPS + PNG
    x2, w2, h2 = 2.2, 2.55, 2.9
    y2 = y_mid - h2 / 2
    rounded_box(ax, x2, y2, w2, h2, COLOR_PRE)
    ax.text(x2 + w2 / 2, y2 + h2 - 0.35, "Sampling \& grouping", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        x2 + w2 / 2,
        y2 + h2 - 0.75,
        "FPS node centers  $\\mathbf{S}\\in\\mathbb{R}^{3\\times M}$\n"
        "Point-to-node assignment, local decentering\n"
        "(normals optional)",
        ha="center",
        va="top",
        fontsize=9,
        linespacing=1.35,
    )

    # 3 First + second PointNet (stacked label)
    x3, w3, h3 = 5.05, 2.35, 2.9
    rounded_box(ax, x3, y2, w3, h3, COLOR_FEAT)
    ax.text(x3 + w3 / 2, y2 + h3 - 0.35, "Backbone PointNets", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        x3 + w3 / 2,
        y2 + h3 - 0.78,
        "1st PointNet on local patches\n"
        r"$\rightarrow$ per-node masked Max $\rightarrow$ scatter",
        ha="center",
        va="top",
        fontsize=9,
        linespacing=1.3,
    )
    ax.text(
        x3 + w3 / 2,
        y2 + 0.95,
        "2nd PointNet on fused tokens\n" r"$\rightarrow$ per-node masked Max",
        ha="center",
        va="top",
        fontsize=9,
        linespacing=1.3,
    )

    # 4 KNN Fusion + SE (large yellow)
    x4, w4, h4 = 7.65, 3.15, 2.9
    rounded_box(ax, x4, y2, w4, h4, COLOR_SE)
    ax.text(x4 + w4 / 2, y2 + h4 - 0.35, r"$\mathrm{KNN}$ fusion module", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        x4 + w4 / 2,
        y2 + h4 - 0.78,
        r"$K$-NN on node centers $\cdot$ shared MLP on neighbors",
        ha="center",
        va="top",
        fontsize=9,
    )
    # mini SE strip inside
    se_x0 = x4 + 0.35
    se_y0 = y2 + 0.55
    ax.text(se_x0 + 1.15, se_y0 + 0.55, "SE channel gate", ha="center", va="bottom", fontsize=9, fontweight="bold")
    rounded_box(ax, se_x0, se_y0, 0.55, 0.45, "#fffef5", lw=0.9)
    ax.text(se_x0 + 0.275, se_y0 + 0.22, r"$\max_K$", ha="center", va="center", fontsize=8)
    arrow(ax, se_x0 + 0.55, se_y0 + 0.22, se_x0 + 0.85, se_y0 + 0.22)
    rounded_box(ax, se_x0 + 0.85, se_y0, 0.75, 0.45, "#fffef5", lw=0.9)
    ax.text(se_x0 + 1.225, se_y0 + 0.22, "FC-ReLU-FC", ha="center", va="center", fontsize=8)
    arrow(ax, se_x0 + 1.6, se_y0 + 0.22, se_x0 + 1.95, se_y0 + 0.22)
    ax.text(se_x0 + 2.05, se_y0 + 0.22, r"$\sigma(\cdot)$", ha="center", va="center", fontsize=10)
    arrow(ax, se_x0 + 2.25, se_y0 + 0.22, se_x0 + 2.55, se_y0 + 0.22)
    ax.text(se_x0 + 2.75, se_y0 + 0.22, r"$\otimes$", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(se_x0 + 1.15, se_y0 - 0.08, r"then Max over $K$ $\rightarrow$ node feature", ha="center", va="top", fontsize=8, style="italic")

    # 5 Head
    x5, w5, h5 = 11.05, 1.85, 2.9
    rounded_box(ax, x5, y2, w5, h5, COLOR_HEAD)
    ax.text(x5 + w5 / 2, y2 + h5 - 0.35, "Keypoint head (MLP)", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        x5 + w5 / 2,
        y2 + h5 - 0.78,
        "Concat node \& KNN features\nEquivariant MLP layers",
        ha="center",
        va="top",
        fontsize=9,
        linespacing=1.25,
    )
    ax.text(
        x5 + w5 / 2,
        y2 + 0.55,
        r"$\mathbf{Q}_i = \mathbf{S}_i + \Delta\mathbf{Q}_i$" "\n" r"$\sigma_i = \mathrm{softplus}(\cdot)+\epsilon$",
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.4,
    )

    # 6 Loss
    x6, w6, h6 = 13.15, 2.55, 2.9
    rounded_box(ax, x6, y2, w6, h6, COLOR_LOSS)
    ax.text(x6 + w6 / 2, y2 + h6 - 0.35, "Training objectives", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        x6 + w6 / 2,
        y2 + h6 - 0.78,
        r"$L_{\mathrm{chamfer}}$: probabilistic Chamfer on $(\mathbf{Q},\mathbf{Q}')$"
        "\nwith batch overlap coefficients $\\gamma_{\\mathrm{src}},\\gamma_{\\mathrm{dst}}$",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
    )
    ax.text(
        x6 + w6 / 2,
        y2 + 0.95,
        r"$L_{\mathrm{surf}}$: keypoints on surface (single-side Chamfer)",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
    )
    ax.text(x6 + w6 / 2, y2 + 0.35, r"$L = L_{\mathrm{chamfer}} + L_{\mathrm{surf}}$", ha="center", va="center", fontsize=10, fontweight="bold")

    # Arrows between main blocks
    arrow(ax, x2 + w2, y_mid, x3 - 0.02, y_mid)
    arrow(ax, x3 + w3, y_mid, x4 - 0.02, y_mid)
    arrow(ax, x4 + w4, y_mid, x5 - 0.02, y_mid)
    arrow(ax, x5 + w5, y_mid, x6 - 0.02, y_mid)

    # Feedback dashed arrow from loss to left (optional training)
    ax.annotate(
        "",
        xy=(3.4, y2 - 0.35),
        xytext=(x6 + w6 * 0.35, y2 - 0.35),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#888888",
            lw=1.0,
            linestyle=(0, (4, 3)),
            mutation_scale=9,
            connectionstyle="arc3,rad=-0.25",
        ),
    )
    ax.text(6.2, y2 - 0.55, "backprop", ha="center", va="top", fontsize=8, color="#666666", style="italic")

    out_png = os.path.join(OUT_DIR, "usip_detector_architecture.png")
    out_pdf = os.path.join(OUT_DIR, "usip_detector_architecture.pdf")
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
