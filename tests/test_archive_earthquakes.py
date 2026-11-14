import os
import unittest
from unittest.mock import patch

os.environ.setdefault("BRAMBLEY_LAT", "47.0")
os.environ.setdefault("BRAMBLEY_LON", "-124.0")

import archive_earthquakes as archive


class ArchiveStateTests(unittest.TestCase):
    def test_migrates_legacy_no_data_error(self):
        processed = {
            "event": {
                "status": "error",
                "processed_utc": "2026-09-22T05:10:21Z",
                "error": "FDSNNoDataException: No data available; HTTP 204",
            }
        }

        self.assertTrue(archive.migrate_no_data_errors(processed))
        self.assertEqual(
            processed["event"],
            {
                "status": "unavailable",
                "processed_utc": "2026-09-22T05:10:21Z",
                "reason": "waveform_not_available",
            },
        )

    def test_migration_leaves_genuine_error_retryable(self):
        processed = {
            "event": {
                "status": "error",
                "error": "TimeoutError: service did not respond",
            }
        }

        self.assertFalse(archive.migrate_no_data_errors(processed))
        self.assertEqual(processed["event"]["status"], "error")

    @patch("archive_earthquakes.utc_now_string", return_value="2026-09-22T06:00:00Z")
    def test_genuine_error_becomes_terminal_at_retry_limit(self, _now):
        retryable = archive.processing_error_record(TimeoutError("temporary"), 1)
        terminal = archive.processing_error_record(TimeoutError("still unavailable"), 2)

        self.assertEqual(retryable["status"], "error")
        self.assertEqual(retryable["attempts"], 2)
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["attempts"], 3)
        self.assertEqual(terminal["error_type"], "TimeoutError")

    @patch("archive_earthquakes.Client")
    def test_fetch_waveform_preserves_fdsn_no_data_exception(self, client_class):
        client_class.return_value.get_waveforms.side_effect = archive.FDSNNoDataException(
            "No data"
        )

        with self.assertRaises(archive.FDSNNoDataException):
            archive.fetch_waveform(
                {"fetch_start": archive.UTCDateTime(0), "fetch_end": archive.UTCDateTime(1)},
                "regional",
            )


if __name__ == "__main__":
    unittest.main()
