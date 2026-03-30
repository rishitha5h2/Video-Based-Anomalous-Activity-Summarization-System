"""
src/summarization/prompt_templates.py
All prompt templates used by llm_summarizer.py.
"""

# ── per-incident prompt ───────────────────────────────────────────────────────
INCIDENT_SYSTEM = (
    "You are a professional CCTV security analyst writing concise, factual incident reports. "
    "Use plain language. Do not speculate beyond the data provided. Be direct and action-oriented."
)

INCIDENT_USER = """\
Analyse the following incident data and write a 2-3 sentence professional incident report entry.

Video file  : {video_name}
Incident #  : {incident_id}
Type        : {incident_type}
Timeframe   : {start_time} → {end_time}
Duration    : {duration}s
Confidence  : {confidence:.0%}
Persons     : {num_persons}
Anomaly peak: {peak_score:.3f}

Write: what likely occurred, severity level (LOW / MEDIUM / HIGH / CRITICAL), recommended immediate action.
"""

# ── full-video executive summary prompt ──────────────────────────────────────
SUMMARY_SYSTEM = (
    "You are a senior security operations manager writing executive briefing reports. "
    "Write in professional English. Be concise, factual, and actionable."
)

SUMMARY_USER = """\
Write a professional executive summary (3-4 paragraphs) for the following surveillance analysis.

Video       : {video_name}
Duration    : {duration_fmt}
Total frames: {frame_count}
Incidents   : {num_incidents}
Types found : {incident_types}

Incident list:
{incident_list}

Cover:
1. Overview of analysis findings
2. Most significant incident and its implications
3. Risk level assessment (LOW / MEDIUM / HIGH / CRITICAL)
4. Specific, actionable recommendations

Be direct. Do not use hedging language.
"""

# ── batch summary prompt ──────────────────────────────────────────────────────
BATCH_SUMMARY_USER = """\
Summarise the following multi-video surveillance analysis in 2-3 paragraphs.

Total videos analysed : {total_videos}
Anomalous videos      : {anomalous_count}
Total incidents found : {total_incidents}
Most common type      : {most_common_type}

Video breakdown:
{video_list}

Provide: overall risk assessment, pattern observations across videos, recommended system-wide actions.
"""

# ── recommendation lookup ─────────────────────────────────────────────────────
RECOMMENDATIONS: dict[str, str] = {
    "fighting":
        "Dispatch security personnel immediately. Separate parties and administer first aid if needed. "
        "Preserve footage for law-enforcement handover.",
    "burglary":
        "Lock down affected entry points. Alert on-site security and contact police. "
        "Review access-control logs and check for additional intrusion points.",
    "robbery":
        "Do not intervene physically. Contact police immediately (emergency line). "
        "Preserve footage and secure area as a crime scene.",
    "assault":
        "Dispatch security and first aid. Contact emergency services. "
        "Do not disturb the area until law enforcement arrives.",
    "shoplifting":
        "Alert loss-prevention staff. Document incident for potential prosecution. "
        "Review POS logs for the same time window.",
    "stealing":
        "Secure remaining assets. Contact police and preserve footage. "
        "Review access logs to identify entry method.",
    "vandalism":
        "Photograph damage for insurance. Review footage to identify perpetrators. "
        "File police report and preserve evidence.",
    "explosion":
        "Evacuate immediately. Contact emergency services (fire, police, ambulance). "
        "Do not re-enter until area is declared safe.",
    "arson":
        "Evacuate and contact fire services immediately. "
        "Preserve footage for arson investigation.",
    "abuse":
        "Contact appropriate authorities and victim support services immediately. "
        "Preserve all footage as legal evidence.",
    "arrest":
        "Coordinate with responding law enforcement. "
        "Preserve footage and document chain of custody.",
    "roadaccidents":
        "Contact emergency services. Manage traffic if safe to do so. "
        "Preserve footage for insurance and legal purposes.",
    "shooting":
        "Initiate lockdown protocol immediately. Contact police (emergency). "
        "Do not approach the area. Preserve footage.",
    "loitering":
        "Dispatch security for a welfare check. "
        "Monitor subject and escalate if behaviour becomes threatening.",
    "default":
        "Review incident details with security supervisor. "
        "Escalate to law enforcement if criminal activity is confirmed.",
}


def get_recommendation(incident_type: str) -> str:
    return RECOMMENDATIONS.get(incident_type.lower(), RECOMMENDATIONS["default"])


def format_incident_list(incidents: list) -> str:
    lines = []
    for inc in incidents:
        m_s, s_s = divmod(int(inc["start_time"]), 60)
        m_e, s_e = divmod(int(inc["end_time"]),   60)
        lines.append(
            f"  • Incident #{inc['id']} [{inc['type'].upper()}] "
            f"{m_s:02d}:{s_s:02d}–{m_e:02d}:{s_e:02d} "
            f"conf={inc['confidence']:.0%} persons={inc.get('num_persons', '?')}"
        )
    return "\n".join(lines) if lines else "  (none)"
