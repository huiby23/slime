import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c4v8/workspace/experiments/c4v8_rollout_bridge"))
from logical_kv import ForwardAddresses, RequestSpan
from hif4_checks import check_before_return, check_invalid


class DecodeChecks(unittest.TestCase):
    def test_deferred_failure_precedes_forward_return_and_context_resets(self):
        with self.assertRaisesRegex(FloatingPointError, "bad Q"):
            with check_before_return():
                check_invalid(torch.tensor(False), "good")
                with check_before_return():
                    check_invalid(torch.tensor(True), "bad Q")
                check_invalid(torch.tensor(True), "bad output")
        with self.assertRaisesRegex(FloatingPointError, "standalone"):
            check_invalid(torch.tensor(True), "standalone")

    def test_per_forward_snapshot_contains_only_indices(self):
        with torch.inference_mode():
            table = torch.tensor([[1, 2, 19], [1, 2, 6]], dtype=torch.int32)
            spans = (RequestSpan(0, 2, 1), RequestSpan(1, 2, 1))
            locations = torch.tensor([19, 6], dtype=torch.int32)
            pool = object()
            first = ForwardAddresses(table, locations, spans, 20, pool)
            first.verify(table, locations, 20, pool)
            table[0, 2] = 7
            locations[0] = 7
            second = ForwardAddresses(table, locations, spans, 20, pool)
            self.assertEqual(first.logical[0].tolist(), [1, 2, 19])
            self.assertEqual(first.locations.tolist(), [19, 6])
            self.assertEqual(second.locations.tolist(), [7, 6])
            with self.assertRaises(RuntimeError):
                first.verify(table, locations, 19, pool)

    def test_rejects_invalid_and_overlapping_writes(self):
        cases = [
            ([[1, 1, 5], [1, 3, 6]], [5, 6]),
            ([[1, 2, 5], [1, 3, 5]], [5, 5]),
            ([[1, 2, 5], [5, 3, 6]], [5, 6]),
            ([[1, 2, 5], [1, 3, 6]], [6, 5]),
            ([[1, 2, 20], [1, 3, 6]], [20, 6]),
            ([[1, -2, 5], [1, 3, 6]], [5, 6]),
        ]
        for rows, writes in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ForwardAddresses(torch.tensor(rows), torch.tensor(writes),
                                (RequestSpan(0, 2, 1), RequestSpan(1, 2, 1)), 20, object())

    def test_chunked_prefill_validates_all_current_writes(self):
        table = torch.tensor([[2, 3, 6, 1], [2, 3, 7, 4]])
        spans = (RequestSpan(0, 2, 2), RequestSpan(1, 2, 2))
        addresses = ForwardAddresses(table, torch.tensor([6,1,7,4]), spans, 8, object())
        self.assertEqual(addresses.locations.tolist(), [6,1,7,4])
        with self.assertRaisesRegex(ValueError, "prefix"):
            ForwardAddresses(torch.tensor([[2, 3, 6, 1], [1, 3, 7, 4]]), torch.tensor([6,1,7,4]), spans, 8, object())

    def test_tracked_mapping_mutation_detected(self):
        table = torch.tensor([[1, 2]], dtype=torch.int32)
        pool = object()
        addresses = ForwardAddresses(table, None, (RequestSpan(0, 1, 1),), 3, pool)
        table[0, 0] = 0
        with self.assertRaises(RuntimeError):
            addresses.verify(table, None, 3, pool)


if __name__ == "__main__":
    unittest.main()
