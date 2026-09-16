"""Bundles figures/maps/tables into a single self-contained HTML report."""
from __future__ import annotations

from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background:#fafafa; }}
h1 {{ border-bottom: 3px solid #2c6e49; padding-bottom: .5rem; }}
h2 {{ margin-top: 2.5rem; color: #2c6e49; }}
.section {{ background: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
img {{ max-width: 100%; border-radius: 4px; margin-top: .5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
.meta {{ color: #666; font-size: .9rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {generated_at}</p>
{sections}
</body>
</html>
"""


class HTMLReportBuilder:
    """Assembles a static HTML summary report from generated figures."""

    def __init__(self, title: str = "Urban Morphology Report — Hyderabad") -> None:
        self.title = title
        self._sections: list[str] = []

    def add_image_section(self, heading: str, image_path: Path, caption: str = "") -> "HTMLReportBuilder":
        """Add a section embedding a figure by relative path.

        Args:
            heading: Section heading text.
            image_path: Path to the image (embedded via relative ``src``).
            caption: Optional caption text below the image.

        Returns:
            ``self``, to allow chaining.
        """
        self._sections.append(
            f'<div class="section"><h2>{heading}</h2>'
            f'<img src="{image_path.name}" alt="{heading}">'
            f'<p class="meta">{caption}</p></div>'
        )
        return self

    def add_table_section(self, heading: str, rows: list[dict[str, object]]) -> "HTMLReportBuilder":
        """Add a section rendering a list of dict rows as an HTML table.

        Args:
            heading: Section heading text.
            rows: List of row dicts (all rows should share keys).

        Returns:
            ``self``, to allow chaining.
        """
        if not rows:
            return self
        cols = list(rows[0].keys())
        header = "".join(f"<th>{c}</th>" for c in cols)
        body_rows = "".join(
            "<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>" for row in rows
        )
        table_html = f"<table><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"
        self._sections.append(f'<div class="section"><h2>{heading}</h2>{table_html}</div>')
        return self

    def build(self, output_path: Path) -> Path:
        """Render and write the final HTML report.

        Args:
            output_path: Destination ``.html`` path.

        Returns:
            The path written to.
        """
        from datetime import datetime, timezone

        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _TEMPLATE.format(
            title=self.title,
            generated_at=datetime.now(timezone.utc).isoformat(),
            sections="\n".join(self._sections),
        )
        output_path.write_text(html, encoding="utf-8")
        logger.info("Saved HTML report: %s", output_path)
        return output_path
