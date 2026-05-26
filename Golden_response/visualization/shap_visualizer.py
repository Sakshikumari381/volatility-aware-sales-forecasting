"""
SHAP explainability visualizations.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)


class SHAPVisualizer:
    """Generate and save SHAP summary and top-feature plots."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.output_dir = self.settings.outputs_shap_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        prefix: str = "summary",
    ) -> Path:
        """Save SHAP beeswarm summary plot."""
        path = self.output_dir / f"{prefix}.png"
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                features=None,
                feature_names=feature_names,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Saved SHAP summary: %s", path)
        except Exception as exc:
            logger.warning("SHAP summary plot failed: %s", exc)
            self._fallback_bar(shap_values, feature_names, path)
        return path

    def save_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 15,
        prefix: str = "top_features",
    ) -> Path:
        """Save bar chart of top K features by mean |SHAP|."""
        path = self.output_dir / f"{prefix}.png"
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        top_names = [feature_names[i] for i in order]
        top_vals = mean_abs[order]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_names[::-1], top_vals[::-1], color="steelblue")
        ax.set_title(f"Top {top_k} SHAP Features")
        ax.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved SHAP top features: %s", path)
        return path

    def _fallback_bar(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        path: Path,
    ) -> None:
        """Fallback bar plot when summary_plot fails."""
        self.save_top_features(
            shap_values,
            feature_names,
            top_k=self.settings.shap_top_k,
            prefix=path.stem,
        )
