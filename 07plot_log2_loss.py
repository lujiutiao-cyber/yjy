import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log2(path: Path):
    epochs = []
    losses = []
    line_pat = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+gpu\d+_")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = line_pat.match(line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))

    if not epochs:
        raise ValueError("No valid epoch/loss lines parsed from log2.txt")
    return epochs, losses


def main():
    base = Path(__file__).resolve().parent
    log_path = base / "log0.txt"
    out_path = base / "usip_train_loss_from_log2.png"

    epochs, losses = parse_log2(log_path)
    best_i = min(range(len(losses)), key=lambda i: losses[i])

    plt.figure(figsize=(11, 6), dpi=160)
    plt.plot(epochs, losses, color="#2b6cb0", linewidth=1.8, label="Test loss")
    plt.scatter([epochs[best_i]], [losses[best_i]], color="#e53e3e", s=40, zorder=3, label="Best")
    plt.annotate(
        f"Best: epoch={epochs[best_i]}, loss={losses[best_i]:.6f}",
        xy=(epochs[best_i], losses[best_i]),
        xytext=(10, -18),
        textcoords="offset points",
        color="#c53030",
        fontsize=10,
    )

    plt.title("USIP Training Curve from log0.txt")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
