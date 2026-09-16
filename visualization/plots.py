"""Publication-quality static figures (trends, distributions, comparisons)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class PlotGenerator:
    """Generates matplotlib figures summarizing morphology results over time."""

    def plot_index_trend(
        self, years: list[int], values_by_index: dict[str, list[float]], output_path: Path, title: str = "Spectral index trends"
    ) -> Path:
        """Plot mean spectral index values across study years.

        Args:
            years: Ordered study years.
            values_by_index: Mapping of index name -> list of mean values
                aligned with ``years``.
            output_path: Destination ``.png`` path.
            title: Figure title.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for name, values in values_by_index.items():
            ax.plot(years, values, marker="o", label=name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean index value")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved index trend plot: %s", output_path)
        return output_path

    def plot_lcz_distribution(self, class_counts: dict[str, int], output_path: Path, title: str = "LCZ class distribution") -> Path:
        """Bar chart of LCZ (or morphology) class pixel/cell counts.

        Args:
            class_counts: Mapping of class name -> count.
            output_path: Destination ``.png`` path.
            title: Figure title.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(class_counts.keys())
        counts = list(class_counts.values())
        ax.bar(names, counts, color="steelblue")
        ax.set_ylabel("Cell count")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved class distribution plot: %s", output_path)
        return output_path

    def plot_confusion_matrix(self, matrix: list[list[int]], labels: list[str], output_path: Path) -> Path:
        """Heatmap of a classification confusion matrix.

        Args:
            matrix: Square confusion matrix as nested lists.
            labels: Class labels matching matrix rows/columns.
            output_path: Destination ``.png`` path.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.array(matrix)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix plot: %s", output_path)
        return output_path
