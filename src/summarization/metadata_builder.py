"""src/summarization/metadata_builder.py — Build structured metadata dict for reports."""

from typing import Dict, List
from datetime import datetime


def build_report_metadata(
    results:        Dict,
    system_version: str = "1.0.0",
) -> Dict:
    """
    Enrich pipeline results with report-ready metadata.

    Adds:
      - generated_at timestamp
      - risk_level classification
      - incident_type_counts breakdown
      - recommendations per incident
      - overall_recommendation
    """
    from src.summarization.prompt_templates import get_recommendation

    incidents = results.get("incidents", [])

    # Risk level
    max_conf = results.get("max_confidence", 0.0)
    if not incidents:
        risk = "LOW"
    elif max_conf >= 0.85:
        risk = "CRITICAL"
    elif max_conf >= 0.70:
        risk = "HIGH"
    elif max_conf >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Type breakdown
    type_counts: Dict[str, int] = {}
    for inc in incidents:
        t = inc.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Per-incident recommendations
    for inc in incidents:
        inc.setdefault("recommendation", get_recommendation(inc.get("type", "")))

    # Dominant type
    dominant_type = (
        max(type_counts, key=type_counts.get) if type_counts else "none"
    )

    metadata = {
        **results,
        "generated_at":        datetime.now().isoformat(),
        "system_version":      system_version,
        "risk_level":          risk,
        "incident_type_counts": type_counts,
        "dominant_type":       dominant_type,
        "overall_recommendation": get_recommendation(dominant_type),
        "total_persons_involved": sum(
            inc.get("num_persons", 0) for inc in incidents
        ),
        "avg_confidence": (
            round(sum(i["confidence"] for i in incidents) / len(incidents), 4)
            if incidents else 0.0
        ),
    }

    return metadata


def incidents_to_table_rows(incidents: List[Dict]) -> List[List[str]]:
    """Convert incident dicts to table rows for PDF rendering."""
    rows = [["#", "Type", "Start", "End", "Duration", "Conf", "Persons", "Risk"]]
    for inc in incidents:
        ms, ss = divmod(int(inc["start_time"]), 60)
        me, se = divmod(int(inc["end_time"]), 60)
        conf   = inc["confidence"]
        risk   = "HIGH" if conf >= 0.75 else "MEDIUM" if conf >= 0.5 else "LOW"
        rows.append([
            str(inc["id"]),
            inc["type"].upper(),
            f"{ms:02d}:{ss:02d}",
            f"{me:02d}:{se:02d}",
            f"{inc['duration']:.0f}s",
            f"{conf:.0%}",
            str(inc.get("num_persons", "?")),
            risk,
        ])
    return rows
