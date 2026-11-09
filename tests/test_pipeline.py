import pandas as pd
import unittest

from src.validation import validate_timeseries


class PipelineQualityTests(unittest.TestCase):
    def test_timeseries_quality_report(self):
        data = pd.DataFrame(
        {
            "station_code": ["TEST", "TEST"],
            "datetime_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z"],
            "H_m": [1.0, 1.1],
            "Q_m3_s": [2.0, 2.1],
            "H_m_missing": [False, True],
            "Q_m3_s_missing": [False, False],
        }
    )
        report = validate_timeseries(data, {"resample_frequency": "10min"})
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["missing_rate"]["H_m_missing"], 0.5)

    def test_timeseries_rejects_duplicate_timestamps(self):
        data = pd.DataFrame(
        {
            "station_code": ["TEST", "TEST"],
            "datetime_utc": ["2026-01-01T00:00:00Z"] * 2,
            "H_m": [1.0, 1.1],
            "Q_m3_s": [2.0, 2.1],
            "H_m_missing": [False, False],
            "Q_m3_s_missing": [False, False],
        }
    )
        with self.assertRaisesRegex(ValueError, "dupliqués"):
            validate_timeseries(data, {"resample_frequency": "10min"})
