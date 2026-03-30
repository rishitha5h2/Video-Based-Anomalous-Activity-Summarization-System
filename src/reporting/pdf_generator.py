import os
import re
from pathlib import Path
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _fmt(secs):
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _clean_summary(text: str) -> str:
    """Remove section headers like CHRONOLOGICAL ACCOUNT:, THREAT ASSESSMENT etc.
    and reformat as clean bullet points."""
    if not text:
        return ""

    # Remove all-caps section labels with colon
    text = re.sub(
        r'(CHRONOLOGICAL ACCOUNT|THREAT ASSESSMENT\s*\[.*?\]|THREAT ASSESSMENT|'
        r'RECOMMENDED ACTIONS|SCENE SETTING|INCIDENT DEEP DIVE|'
        r'NORMAL ACTIVITY CONFIRMATION|PARAGRAPH \d+[^:]*)\s*:?\s*',
        '', text, flags=re.IGNORECASE
    )

    # Split into sentences and clean up
    text = text.strip()
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _to_bullets(text: str) -> list:
    """Convert a block of text into a clean bullet list."""
    # Split on newlines or sentence boundaries after periods
    lines = []
    for para in re.split(r'\n\n+', text):
        para = para.strip()
        if para:
            lines.append(para)
    return lines


class PDFReportGenerator:
    """Generate clean, professional PDF incident reports."""

    def __init__(self):
        self._check_deps()

    def _check_deps(self):
        try:
            from reportlab.lib.pagesizes import A4
            self.available = True
        except ImportError:
            logger.warning("reportlab not installed. PDF generation disabled.")
            self.available = False

    def generate(self, results: Dict, output_path: str) -> str:
        if not self.available:
            return self._generate_text_fallback(results, output_path)

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.lib.colors import HexColor, white
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable, Image as RLImage,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2.5*cm,
        )

        # ── Colours ────────────────────────────────────────────────────────
        RED    = HexColor("#e8453c")
        TEAL   = HexColor("#00c9b1")
        AMBER  = HexColor("#f5a623")
        GRAY   = HexColor("#8da4bf")
        LGRAY  = HexColor("#f4f6f9")
        DGRAY  = HexColor("#444444")
        LBLUE  = HexColor("#e8f0fe")

        # ── Styles ─────────────────────────────────────────────────────────
        H1 = ParagraphStyle("H1", fontName="Helvetica-Bold",
                            fontSize=20, textColor=RED, spaceAfter=2)
        H2 = ParagraphStyle("H2", fontName="Helvetica-Bold",
                            fontSize=13, textColor=TEAL,
                            spaceBefore=14, spaceAfter=6)
        H3 = ParagraphStyle("H3", fontName="Helvetica-Bold",
                            fontSize=10, textColor=AMBER,
                            spaceBefore=10, spaceAfter=4)
        BODY = ParagraphStyle("Body", fontName="Helvetica",
                              fontSize=9, textColor=DGRAY,
                              leading=15, spaceAfter=4)
        BULLET = ParagraphStyle("Bullet", fontName="Helvetica",
                                fontSize=9, textColor=DGRAY,
                                leading=15, leftIndent=14,
                                firstLineIndent=-14, spaceAfter=3)
        SUB = ParagraphStyle("Sub", fontName="Helvetica",
                             fontSize=11, textColor=GRAY, spaceAfter=4)
        FOOTER = ParagraphStyle("Footer", fontName="Helvetica",
                                fontSize=7, textColor=GRAY,
                                alignment=TA_CENTER)
        VERDICT_ANOM = ParagraphStyle(
            "VA", fontName="Helvetica-Bold", fontSize=12,
            textColor=RED, alignment=TA_CENTER,
            backColor=HexColor("#fde8e8"), borderPad=10, spaceAfter=12)
        VERDICT_OK = ParagraphStyle(
            "VK", fontName="Helvetica-Bold", fontSize=12,
            textColor=HexColor("#1a7a1a"), alignment=TA_CENTER,
            backColor=HexColor("#e8fde8"), borderPad=10, spaceAfter=12)

        story = []
        incidents = results.get("incidents", [])
        n_inc     = len(incidents)
        dur       = results.get("duration", 0)
        video_name = results.get("video_name", "Unknown")
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── HEADER ─────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Video Anomaly Detection Report", H1))
        story.append(Paragraph("Surveillance Analysis System", SUB))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=RED, spaceAfter=10))

        # ── META TABLE ─────────────────────────────────────────────────────
        meta = [
            ["Video File",    video_name,
             "Analysis Date", generated_at],
            ["Duration",      _fmt(dur),
             "Incidents",     str(n_inc)],
            ["Resolution",    results.get("resolution", "N/A"),
             "Status",        "ANOMALOUS" if n_inc > 0 else "NORMAL"],
        ]
        mt = Table(meta, colWidths=[2.8*cm, 7*cm, 3.2*cm, 4.5*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), LBLUE),
            ("BACKGROUND",  (2, 0), (2, -1), LBLUE),
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
            ("FONTNAME",    (3, 0), (3, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [HexColor("#f9f9f9"), white]),
            ("PADDING",     (0, 0), (-1, -1), 6),
            # Colour STATUS cell
            ("TEXTCOLOR",   (3, 2), (3, 2),
             RED if n_inc > 0 else HexColor("#1a7a1a")),
            ("FONTNAME",    (3, 2), (3, 2), "Helvetica-Bold"),
        ]))
        story.append(mt)
        story.append(Spacer(1, 0.15*inch))

        # ── VERDICT BANNER ─────────────────────────────────────────────────
        if n_inc > 0:
            story.append(Paragraph(
                f"ANOMALOUS ACTIVITY DETECTED — {n_inc} incident(s) found",
                VERDICT_ANOM))
        else:
            story.append(Paragraph(
                "NO ANOMALOUS ACTIVITY DETECTED — Recording is clear",
                VERDICT_OK))

        # ── SUMMARY ────────────────────────────────────────────────────────
        story.append(Paragraph("Summary", H2))
        raw_summary = results.get("summary", "No summary available.")
        clean = _clean_summary(raw_summary)
        bullets = _to_bullets(clean)
        if len(bullets) <= 1:
            story.append(Paragraph(clean, BODY))
        else:
            for b in bullets:
                story.append(Paragraph(f"• {b}", BULLET))
        story.append(Spacer(1, 0.1*inch))

        # ── INCIDENT TIMELINE TABLE ────────────────────────────────────────
        if n_inc > 0:
            story.append(Paragraph("Incident Timeline", H2))
            tl = [["#", "Start", "End", "Duration", "Type",
                   "Confidence", "Persons"]]
            for inc in incidents:
                tl.append([
                    str(inc["id"]),
                    _fmt(inc["start_time"]),
                    _fmt(inc["end_time"]),
                    f"{inc['duration']:.0f}s",
                    inc["type"].capitalize(),
                    f"{inc['confidence']:.0%}",
                    str(inc.get("num_persons", "?")),
                ])
            tlt = Table(tl, colWidths=[1*cm, 2*cm, 2*cm,
                                        2.5*cm, 4*cm, 3*cm, 2.5*cm])
            tlt.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), HexColor("#e8453c")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("GRID",        (0, 0), (-1, -1), 0.3,
                 HexColor("#dddddd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [white, HexColor("#fef9f9")]),
                ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                ("PADDING",     (0, 0), (-1, -1), 5),
            ]))
            story.append(tlt)
            story.append(Spacer(1, 0.2*inch))

            # ── PER-INCIDENT DETAILS ───────────────────────────────────────
            story.append(Paragraph("Incident Details", H2))
            for inc in incidents:
                story.append(Paragraph(
                    f"Incident #{inc['id']} — "
                    f"{inc['type'].capitalize()}  "
                    f"({_fmt(inc['start_time'])} → {_fmt(inc['end_time'])})",
                    H3
                ))

                # Details row — removed Peak Score
                det = Table(
                    [[
                        "Confidence:", f"{inc['confidence']:.1%}",
                        "Duration:",   f"{inc['duration']:.1f}s",
                        "Persons:",    str(inc.get("num_persons", "?")),
                    ]],
                    colWidths=[2.5*cm, 3*cm, 2.5*cm, 3*cm,
                               2.5*cm, 3.5*cm]
                )
                det.setStyle(TableStyle([
                    ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
                    ("FONTNAME",   (4, 0), (4, -1), "Helvetica-Bold"),
                    ("FONTSIZE",   (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, -1),
                     HexColor("#fef5f5")),
                    ("GRID",       (0, 0), (-1, -1), 0.3,
                     HexColor("#dddddd")),
                    ("PADDING",    (0, 0), (-1, -1), 5),
                ]))
                story.append(det)

                # Narrative as bullet points
                narrative = inc.get("narrative", "")
                if narrative:
                    cleaned_narr = _clean_summary(narrative)
                    narr_bullets = _to_bullets(cleaned_narr)
                    if len(narr_bullets) <= 1:
                        story.append(Paragraph(cleaned_narr, BODY))
                    else:
                        for b in narr_bullets:
                            story.append(Paragraph(f"• {b}", BULLET))

                # Snapshot images
                frame_paths = inc.get("frame_paths", [])
                if frame_paths:
                    img_cells = []
                    for fpath in frame_paths[:3]:
                        if os.path.exists(fpath):
                            try:
                                img_cells.append(
                                    RLImage(fpath, width=5.5*cm,
                                            height=4*cm,
                                            kind="proportional")
                                )
                            except Exception:
                                pass
                    if img_cells:
                        # Pad to 3 cols
                        while len(img_cells) < 3:
                            img_cells.append("")
                        img_table = Table(
                            [img_cells],
                            colWidths=[5.8*cm, 5.8*cm, 5.8*cm]
                        )
                        img_table.setStyle(TableStyle([
                            ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
                            ("PADDING", (0, 0), (-1, -1), 3),
                        ]))
                        story.append(img_table)

                story.append(Spacer(1, 0.12*inch))

        # ── FOOTER ─────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.1*inch))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=GRAY, spaceAfter=5))
        story.append(Paragraph(
            f"Report generated on {generated_at} — Confidential",
            FOOTER
        ))

        doc.build(story)
        logger.info(f"PDF report saved: {output_path}")
        return output_path

    def _generate_text_fallback(self, results: Dict,
                                output_path: str) -> str:
        txt = output_path.replace(".pdf", ".txt")
        Path(txt).parent.mkdir(parents=True, exist_ok=True)
        dur  = results.get("duration", 0)
        incs = results.get("incidents", [])
        lines = [
            "VIDEO ANOMALY DETECTION REPORT",
            "=" * 50,
            f"Video    : {results.get('video_name', 'Unknown')}",
            f"Duration : {_fmt(dur)}",
            f"Incidents: {len(incs)}",
            f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "", "SUMMARY:", _clean_summary(results.get("summary", "")),
            "", "INCIDENTS:",
        ]
        for inc in incs:
            lines += [
                f"  #{inc['id']} {inc['type'].capitalize()} "
                f"{_fmt(inc['start_time'])}–{_fmt(inc['end_time'])} "
                f"({inc['confidence']:.0%} confidence)",
                f"  {_clean_summary(inc.get('narrative', ''))}",
                "",
            ]
        with open(txt, "w") as f:
            f.write("\n".join(lines))
        return txt