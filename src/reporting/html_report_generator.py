"""
src/reporting/html_report_generator.py
Renders the Jinja2 HTML template and optionally converts to PDF via WeasyPrint.
Falls back to saving the HTML directly if WeasyPrint is not installed.
"""

import os
from pathlib import Path
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class HTMLReportGenerator:
    """
    Renders a full-page HTML report from pipeline results.
    Can optionally convert to PDF using WeasyPrint.
    """

    TEMPLATE_PATH = Path(__file__).parent / "report_templates" / "report_template.html"

    def generate_html(self, results: Dict, output_path: str) -> str:
        """Render and save the HTML report. Returns output path."""
        try:
            from jinja2 import Template
        except ImportError:
            logger.warning("jinja2 not installed — saving plain HTML")
            return self._save_plain(results, output_path)

        template_str = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        template     = Template(template_str)

        # Enrich incidents with formatted fields
        incidents = []
        for inc in results.get("incidents", []):
            enriched = dict(inc)
            enriched["start_fmt"] = _fmt_time(inc["start_time"])
            enriched["end_fmt"]   = _fmt_time(inc["end_time"])
            from src.summarization.prompt_templates import get_recommendation
            enriched.setdefault("recommendation", get_recommendation(inc["type"]))
            incidents.append(enriched)

        duration = results.get("duration", 0)
        context  = {
            **results,
            "incidents":      incidents,
            "duration_fmt":   _fmt_time(duration),
            "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_version": results.get("system_version", "1.0.0"),
            "risk_level":     results.get("risk_level", "UNKNOWN"),
        }

        html = template.render(**context)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved: {output_path}")
        return output_path

    def generate_pdf(self, results: Dict, output_path: str) -> str:
        """Render HTML then convert to PDF via WeasyPrint. Falls back to HTML."""
        html_path = output_path.replace(".pdf", "_report.html")
        self.generate_html(results, html_path)

        try:
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(output_path)
            logger.info(f"PDF (WeasyPrint) saved: {output_path}")
            return output_path
        except ImportError:
            logger.warning("WeasyPrint not installed — returning HTML report instead")
            return html_path
        except Exception as e:
            logger.error(f"WeasyPrint error: {e} — returning HTML")
            return html_path

    def _save_plain(self, results: Dict, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            f"<h1>VIGIL Report</h1><pre>{results}</pre>", encoding="utf-8"
        )
        return output_path
