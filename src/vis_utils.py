import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path

def plot_stable_weights(metadata: List[Dict[str, float]], out_path: Path, title_suffix: str = "") -> None:
    if not metadata:
        print("No metadata to plot.")
        return

    weight_norms = np.array([m["weight_norm"] for m in metadata], dtype=np.float32)
    Vmax = np.array([m["V_max"] for m in metadata], dtype=np.float32)
    viol = np.array([m["constraint_violation"] for m in metadata], dtype=np.float32)
    scales = np.array([m["initial_scale"] for m in metadata], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(weight_norms, bins=15, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Weight norm ||w0||")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Stable weight norms" + title_suffix)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(Vmax, viol, alpha=0.6, s=60)
    axes[0, 1].set_xlabel("V_max")
    axes[0, 1].set_ylabel("Constraint violation")
    axes[0, 1].set_title("V_max vs constraint" + title_suffix)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(scales, weight_norms, alpha=0.6, s=60)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Init scale")
    axes[1, 0].set_ylabel("Weight norm ||w0||")
    axes[1, 0].set_title("Scale -> norm" + title_suffix)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(viol, bins=15, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Constraint violation")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Constraint distribution" + title_suffix)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")