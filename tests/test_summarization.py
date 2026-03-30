"""tests/test_summarization.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.summarization.llm_summarizer  import LLMSummarizer
from src.summarization.prompt_templates import (
    get_recommendation, format_incident_list, RECOMMENDATIONS
)
from src.reporting.json_exporter import JSONExporter
import json, tempfile, pathlib


def _sample_incident(idx=1):
    return {
        "id": idx, "start_time": 12.5, "end_time": 34.0,
        "duration": 21.5, "type": "fighting", "confidence": 0.87,
        "num_persons": 2, "peak_anomaly_score": 0.91,
        "frame_paths": [], "narrative": "",
    }


def _sample_results():
    return {
        "video_name": "Fighting001.mp4",
        "duration": 120.0, "resolution": "1280x720",
        "fps": 25, "frame_count": 600,
        "incidents": [_sample_incident()],
        "incident_count": 1, "is_anomalous": True,
        "max_confidence": 0.87, "summary": "",
    }


class TestLLMSummarizer:
    def setup_method(self):
        # No API key → always uses template fallback
        self.summ = LLMSummarizer(api_key=None)

    def test_summarize_incident_returns_string(self):
        inc = _sample_incident()
        s   = self.summ.summarize_incident(inc, "test.mp4")
        assert isinstance(s, str) and len(s) > 20

    def test_summarize_video_returns_string(self):
        res = _sample_results()
        s   = self.summ.summarize_video(res)
        assert isinstance(s, str) and len(s) > 20

    def test_normal_video_summary(self):
        res = {**_sample_results(), "incidents": [], "incident_count": 0, "is_anomalous": False}
        s   = self.summ.summarize_video(res)
        assert isinstance(s, str)


class TestPromptTemplates:
    def test_known_recommendation(self):
        r = get_recommendation("fighting")
        assert isinstance(r, str) and len(r) > 10

    def test_unknown_falls_back_to_default(self):
        r = get_recommendation("alien_invasion")
        assert r == RECOMMENDATIONS["default"]

    def test_format_incident_list(self):
        incs = [_sample_incident(1), _sample_incident(2)]
        s    = format_incident_list(incs)
        assert "#1" in s and "#2" in s

    def test_format_empty_list(self):
        s = format_incident_list([])
        assert "none" in s.lower()


class TestJSONExporter:
    def test_export_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out  = os.path.join(tmpdir, "report.json")
            exp  = JSONExporter()
            path = exp.export(_sample_results(), out)
            assert pathlib.Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert "video_name"    in data
            assert "generated_at"  in data
            assert "vigil_version" in data

    def test_export_incidents_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out  = os.path.join(tmpdir, "r.json")
            exp  = JSONExporter()
            exp.export(_sample_results(), out)
            with open(out) as f:
                data = json.load(f)
            assert len(data["incidents"]) == 1
            assert data["incidents"][0]["type"] == "fighting"
