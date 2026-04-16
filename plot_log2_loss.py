import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log2(path: Path, max_epoch: int = 300):
    epochs = []
    losses = []
    line_pat = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+gpu\d+_")

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = line_pat.match(line)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > max_epoch:
            continue
        epochs.append(ep)
        losses.append(float(m.group(2)))

    if not epochs:
        raise ValueError("No valid epoch/loss lines parsed from log2.txt")
    return epochs, losses


def main():
    base = Path(__file__).resolve().parent
    log_path = base / "log0.txt"
    out_path = base / "usip_train_loss_from_log2.png"

    epochs, losses = parse_log2(log_path, max_epoch=300)
    plt.figure(figsize=(11, 6), dpi=160)
    plt.plot(epochs, losses, color="#2b6cb0", linewidth=1.8, label="Test loss")

    plt.title("USIP Training Curve from log2.txt")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.xlim(0, 300)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
