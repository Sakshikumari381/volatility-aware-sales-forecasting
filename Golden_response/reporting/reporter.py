"""
Generate HTML and text forecast reports.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)


class ForecastReporter:
    """Build structured training and evaluation reports."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.output_dir = self.settings.outputs_reports_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        comparison_df: Optional[pd.DataFrame] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write HTML report with metrics table.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        if comparison_df is None:
            rows = []
            for model, m in metrics.items():
                rows.append({"model": model, **m})
            comparison_df = pd.DataFrame(rows)

        table_html = comparison_df.to_html(index=False, float_format="%.4f")
        extra_html = ""
        if extra:
            extra_html = "<pre>" + "\n".join(f"{k}: {v}" for k, v in extra.items()) + "</pre>"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Forecast Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ color: #1a5276; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #2874a6; color: white; }}
  </style>
</head>
<body>
  <h1>Volatility-Aware Sales Forecast Report</h1>
  <p>Generated: {timestamp}</p>
  <h2>Model Comparison</h2>
  {table_html}
  <h2>Configuration</h2>
  {extra_html}
</body>
</html>"""

        path = self.output_dir / f"forecast_report_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    def generate_text_summary(
        self,
        metrics: Dict[str, Dict[str, float]],
    ) -> Path:
        """Plain-text metrics summary."""
        lines = ["=== Model Evaluation Summary ===", ""]
        ranked = sorted(metrics.items(), key=lambda x: x[1].get("rmsle", 999))
        for rank, (model, m) in enumerate(ranked, 1):
            lines.append(f"{rank}. {model.upper()}")
            lines.append(f"   RMSLE: {m.get('rmsle', 0):.4f}")
            lines.append(f"   RMSE:  {m.get('rmse', 0):.4f}")
            lines.append(f"   R²:    {m.get('r2', 0):.4f}")
            lines.append("")
        path = self.output_dir / "summary.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
