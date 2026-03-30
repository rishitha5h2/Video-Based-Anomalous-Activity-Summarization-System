import os
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# ── Prompts ───────────────────────────────────────────────────────────────────

INCIDENT_PROMPT = """You are a professional CCTV security analyst writing an official incident report.

You have been given detection data from an AI surveillance system for a specific incident.
Your job is to write a vivid, detailed narrative — as if you are describing exactly what
the surveillance camera captured — telling the COMPLETE STORY of what happened.

─── INCIDENT DATA ───
Video File    : {video_name}
Incident #    : {incident_id}
Detected Type : {incident_type}
Start Time    : {start_time}
End Time      : {end_time}
Duration      : {duration} seconds
Confidence    : {confidence:.0%}
Persons       : {num_persons} individual(s) detected
Peak Score    : {peak_score:.3f}
Prior Context : {prior_context}
─────────────────────

Write 3-4 clear sentences describing:
- What was happening at {start_time} and how it developed over {duration} seconds
- Who was involved and what made this activity suspicious or dangerous
- What the confidence level of {confidence:.0%} means for the severity
- What security personnel should do about this specific incident

Write in professional past tense. Be specific and factual.
Do NOT use section headers or labels — just clear flowing sentences."""


VIDEO_SUMMARY_PROMPT = """You are a senior security analyst writing a surveillance incident summary.

You have been given analysis data from a surveillance recording.

─── VIDEO ANALYSIS DATA ───
Video File      : {video_name}
Recording Length: {duration_fmt}
Resolution      : {resolution}
Frames Analysed : {frame_count}
Total Incidents : {num_incidents}
Incident Types  : {incident_types}
Max Confidence  : {max_confidence:.0%}
Overall Status  : {status}

─── INCIDENT TIMELINE ───
{incident_timeline}

─── INDIVIDUAL NARRATIVES ───
{individual_narratives}
───────────────────────────

Write a clear professional summary in 4 paragraphs. No section headers. No all-caps labels.
No words like CHRONOLOGICAL ACCOUNT or THREAT ASSESSMENT anywhere in your response.

Paragraph 1: Describe the recording — what video, how long, how many incidents found and what types.

Paragraph 2: Tell the story of what happened chronologically using specific timestamps.
Describe who was involved and what they were doing during each incident.

Paragraph 3: Explain the severity based on confidence scores and activity types.
Is this a serious threat? Why or why not?

Paragraph 4: Give specific actions for security personnel — what to review, who to notify,
what documentation to preserve, what follow-up is needed.

Write in professional past tense. Factual and clear. Just plain paragraphs."""


NORMAL_VIDEO_PARAGRAPH = """PARAGRAPH 6 — NORMAL ACTIVITY CONFIRMATION
Since no incidents were detected, describe what normal activity was observed and
confirm the area was clear throughout the recording."""

NO_INCIDENT_PROMPT = """You are a professional security analyst.

A surveillance video was analysed and NO anomalous activity was detected.

─── VIDEO DATA ───
Video File     : {video_name}
Duration       : {duration_fmt}
Frames Analysed: {frame_count}
Resolution     : {resolution}
──────────────────

Write a brief (2–3 paragraph) clearance report confirming:
1. The recording was reviewed and found to be clear
2. What normal activity patterns were observed (describe plausible normal activity
   for a surveillance camera — people walking, normal movement patterns, etc.)
3. Confirmation that no security intervention is required

Write in professional prose. Be reassuring but factual."""


class LLMSummarizer:
    """Generate rich story-style summaries using Google Gemini 2.5 Flash."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-2.5-flash-preview-04-17"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model   = model
        self.client  = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. Summaries will use templates.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Gemini client initialized — model: {self.model}")
        except ImportError:
            logger.warning(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}. Using template summaries.")

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize_incident(self, incident: Dict, video_name: str,
                           all_incidents: Optional[List[Dict]] = None) -> str:
        """Generate a vivid story-style narrative for a single incident."""
        mins_s, secs_s = divmod(int(incident["start_time"]), 60)
        mins_e, secs_e = divmod(int(incident["end_time"]), 60)

        # Build prior context — what happened before this incident
        prior = "This is the first detected incident in the recording."
        if all_incidents:
            earlier = [i for i in all_incidents
                       if i["id"] < incident["id"]]
            if earlier:
                prior_parts = [
                    f"Incident #{i['id']} ({i['type']}) at "
                    f"{int(i['start_time'])//60:02d}:{int(i['start_time'])%60:02d}"
                    for i in earlier
                ]
                prior = f"Earlier incidents in this recording: {'; '.join(prior_parts)}."

        prompt = INCIDENT_PROMPT.format(
            video_name    = video_name,
            incident_id   = incident["id"],
            incident_type = incident["type"].upper(),
            start_time    = f"{mins_s:02d}:{secs_s:02d}",
            end_time      = f"{mins_e:02d}:{secs_e:02d}",
            duration      = incident["duration"],
            confidence    = incident["confidence"],
            num_persons   = incident.get("num_persons", "unknown number of"),
            peak_score    = incident.get("peak_anomaly_score", 0),
            prior_context = prior,
        )

        if self.client:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 500,
                        "temperature":       0.5,
                    }
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini error (incident narrative): {e}")

        return self._template_incident(incident, video_name)

    def summarize_video(self, results: Dict) -> str:
        """Generate a complete story-style executive summary for the full video."""
        incidents = results.get("incidents", [])
        dur       = results.get("duration", 0)
        dur_fmt   = f"{int(dur)//60:02d}:{int(dur)%60:02d}"

        if not incidents:
            return self._summarize_normal_video(results, dur_fmt)

        # Build rich incident timeline
        timeline_lines = []
        for i in incidents:
            ts = f"{int(i['start_time'])//60:02d}:{int(i['start_time'])%60:02d}"
            te = f"{int(i['end_time'])//60:02d}:{int(i['end_time'])%60:02d}"
            timeline_lines.append(
                f"  [{ts} → {te}]  #{i['id']} {i['type'].upper():<15} "
                f"conf:{i['confidence']:.0%}  persons:{i.get('num_persons','?')}  "
                f"duration:{i['duration']:.1f}s  peak_score:{i.get('peak_anomaly_score',0):.3f}"
            )

        # Build individual narratives block
        narrative_lines = []
        for i in incidents:
            ts = f"{int(i['start_time'])//60:02d}:{int(i['start_time'])%60:02d}"
            narrative_lines.append(
                f"Incident #{i['id']} ({i['type'].upper()}) at {ts}:\n"
                f"{i.get('narrative', 'No narrative available.')}"
            )

        types      = list(set(i["type"] for i in incidents))
        max_conf   = max(i["confidence"] for i in incidents)
        status     = "ANOMALOUS — REVIEW REQUIRED" if incidents else "CLEAR"

        prompt = VIDEO_SUMMARY_PROMPT.format(
            video_name           = results.get("video_name", "Unknown"),
            duration_fmt         = dur_fmt,
            duration_secs        = f"{dur:.0f}",
            resolution           = results.get("resolution", "Unknown"),
            frame_count          = results.get("frame_count", "Unknown"),
            num_incidents        = len(incidents),
            incident_types       = ", ".join(types),
            max_confidence       = max_conf,
            status               = status,
            incident_timeline    = "\n".join(timeline_lines),
            individual_narratives= "\n\n".join(narrative_lines),
            normal_paragraph     = "",
        )

        if self.client:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 1500,
                        "temperature":       0.5,
                    }
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini error (video summary): {e}")

        return self._template_video(results)

    def _summarize_normal_video(self, results: Dict, dur_fmt: str) -> str:
        """Summary for videos with no detected incidents."""
        prompt = NO_INCIDENT_PROMPT.format(
            video_name  = results.get("video_name", "Unknown"),
            duration_fmt= dur_fmt,
            frame_count = results.get("frame_count", "Unknown"),
            resolution  = results.get("resolution", "Unknown"),
        )
        if self.client:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 400,
                        "temperature":       0.4,
                    }
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini error (normal video summary): {e}")

        return self._template_video(results)

    # ── Template fallbacks (no API key / error) ───────────────────────────────

    def _template_incident(self, incident: Dict, video_name: str) -> str:
        mins_s, secs_s = divmod(int(incident["start_time"]), 60)
        mins_e, secs_e = divmod(int(incident["end_time"]), 60)
        t    = incident["type"].capitalize()
        conf = incident["confidence"]
        dur  = incident["duration"]
        n    = incident.get("num_persons", "multiple")
        peak = incident.get("peak_anomaly_score", 0)

        severity = "critical" if conf > 0.8 else "moderate" if conf > 0.6 else "low-level"

        return (
            f"At {mins_s:02d}:{secs_s:02d} in {video_name}, the surveillance system "
            f"flagged a {severity} {t} incident lasting {dur:.0f} seconds "
            f"({mins_s:02d}:{secs_s:02d} to {mins_e:02d}:{secs_e:02d}). "
            f"The detection involved {n} person(s) with a peak anomaly score of {peak:.3f}, "
            f"indicating {self._severity_desc(conf)}. "
            f"The system registered {conf:.0%} confidence in this classification. "
            f"Security personnel should immediately review this segment and take "
            f"{'urgent action' if conf > 0.7 else 'appropriate follow-up action'} "
            f"as warranted by the footage."
        )

    def _template_video(self, results: Dict) -> str:
        incidents = results.get("incidents", [])
        name      = results.get("video_name", "the recording")
        dur       = results.get("duration", 0)
        dur_fmt   = f"{int(dur)//60:02d}:{int(dur)%60:02d}"

        if not incidents:
            return (
                f"Security review of {name} covering {dur_fmt} of footage found no "
                f"anomalous activity. The {results.get('frame_count', 'full')} frames "
                f"analysed showed normal surveillance conditions throughout the recording. "
                f"No security intervention is required. The area remained clear for the "
                f"entire duration and no persons of interest were identified."
            )

        types    = list(set(i["type"] for i in incidents))
        max_conf = max(i["confidence"] for i in incidents)
        total_dur= sum(i["duration"] for i in incidents)
        n_persons= max((i.get("num_persons", 0) or 0) for i in incidents)

        # Build chronological story
        story_parts = []
        for inc in incidents:
            ts = f"{int(inc['start_time'])//60:02d}:{int(inc['start_time'])%60:02d}"
            te = f"{int(inc['end_time'])//60:02d}:{int(inc['end_time'])%60:02d}"
            story_parts.append(
                f"At {ts}, a {inc['type']} incident was detected involving "
                f"{inc.get('num_persons','multiple')} person(s) and lasting "
                f"{inc['duration']:.0f} seconds until {te} "
                f"(confidence: {inc['confidence']:.0%})"
            )

        threat = "HIGH" if max_conf > 0.8 else "MEDIUM" if max_conf > 0.6 else "LOW"

        # Build clean chronological account
        return (
            f"Surveillance analysis of {name} covering {dur_fmt} of footage identified "
            f"{len(incidents)} incident(s) of concern, including {', '.join(types)}. "
            f"The recording was reviewed across {results.get('frame_count','all available')} frames.\n\n"

            f"{'. '.join(story_parts)}. "
            f"Anomalous activity totalled {total_dur:.0f} seconds across the recording, "
            f"with up to {n_persons} person(s) involved at peak.\n\n"

            f"The overall threat level is assessed as {threat} based on a maximum detection "
            f"confidence of {max_conf:.0%}. "
            f"{'The confidence scores strongly suggest genuine security events requiring immediate attention.' if max_conf > 0.7 else 'The confidence scores indicate activity that warrants review.'} "
            f"{'The pattern of multiple incidents across the recording suggests sustained or repeated suspicious behaviour.' if len(incidents) > 1 else ''}\n\n"

            f"Security personnel should review flagged segments between "
            f"{int(incidents[0]['start_time'])//60:02d}:{int(incidents[0]['start_time'])%60:02d} and "
            f"{int(incidents[-1]['end_time'])//60:02d}:{int(incidents[-1]['end_time'])%60:02d}. "
            f"{'Relevant authorities should be notified given the confidence level of the detections.' if max_conf > 0.75 else ''} "
            f"All incident footage should be preserved and access logs for the affected timeframe reviewed."
        )

    @staticmethod
    def _severity_desc(conf: float) -> str:
        if conf > 0.85: return "a very high probability of genuine criminal activity"
        if conf > 0.70: return "a strong likelihood of suspicious behaviour"
        if conf > 0.55: return "moderate probability of anomalous behaviour"
        return "low-level unusual activity that warrants review"