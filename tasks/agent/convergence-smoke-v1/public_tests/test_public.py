import unittest

from convergence_project import build_record


class PublicConvergenceTests(unittest.TestCase):
    def test_record_normalizes_label_and_totals_values(self):
        self.assertEqual(
            build_record("  Release Candidate  ", [2, 3, 5]),
            {"label": "release candidate", "total": 10, "count": 3},
        )

    def test_boolean_values_are_rejected(self):
        with self.assertRaises(ValueError):
            build_record("ok", [1, True, 3])


if __name__ == "__main__":
    unittest.main()
