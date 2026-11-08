import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nvfp4_mxfp8_attention_sim'))
from attention import AttentionConfig, attention
from logical_kv import RequestSpan, spans_from_lengths, token_cache_view, forward_cached


class LogicalKVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(1908)
        self.config = AttentionConfig(query_block=5, key_block=8)
        self.hq, self.hk, self.d = 4, 2, 17

    def values(self, n):
        return tuple(torch.randn((n, h, self.d), generator=self.generator).bfloat16()
                     for h in (self.hq, self.hk, self.hk))

    def buffers(self, count=128):
        return (torch.zeros((count, self.hk, self.d), dtype=torch.bfloat16),
                torch.zeros((count, self.hk, self.d), dtype=torch.bfloat16),
                torch.full((8, 64), -1, dtype=torch.int64))

    def call(self, q, kc, vc, table, spans, **kwargs):
        return forward_cached(q, kc, vc, table, spans, core=attention,
                              config=self.config, scale=1 / math.sqrt(self.d), **kwargs)

    def direct(self, q, k, v, prefix=0):
        result = attention(q.transpose(0, 1).unsqueeze(0),
                           k.transpose(0, 1).unsqueeze(0),
                           v.transpose(0, 1).unsqueeze(0), self.config, query_offset=prefix)
        return result.squeeze(0).transpose(0, 1)

    def test_packed_requests_gqa_random_addresses(self):
        kc, vc, table = self.buffers()
        a, b = self.values(13), self.values(7)
        slots = torch.randperm(kc.shape[0], generator=self.generator)
        table[3, :13] = slots[:13]; table[6, :7] = slots[13:20]
        kc[slots[:13]], vc[slots[:13]] = a[1], a[2]
        kc[slots[13:20]], vc[slots[13:20]] = b[1], b[2]
        out = self.call(torch.cat((a[0], b[0])), kc, vc, table,
                        [RequestSpan(3, 0, 13), RequestSpan(6, 0, 7)])
        torch.testing.assert_close(out, torch.cat((self.direct(*a), self.direct(*b))), atol=0, rtol=0)
        vc[slots[13:20]] += 100
        changed = self.call(torch.cat((a[0], b[0])), kc, vc, table,
                            [RequestSpan(3, 0, 13), RequestSpan(6, 0, 7)])
        torch.testing.assert_close(changed[:13], out[:13], atol=0, rtol=0)

    def test_prefill_prefix_chunks_and_decode_equal_full(self):
        q, k, v = self.values(23)
        kc, vc, table = self.buffers()
        slots = torch.randperm(kc.shape[0], generator=self.generator)[:23]
        table[2, :23] = slots
        chunks = [(0, 7), (7, 13), (13, 19)] + [(n, n + 1) for n in range(19, 23)]
        pieces = []
        for start, end in chunks:
            pieces.append(self.call(q[start:end], kc, vc, table,
                          [RequestSpan(2, start, end - start)], new_key=k[start:end],
                          new_value=v[start:end], out_cache_locations=slots[start:end]))
        torch.testing.assert_close(torch.cat(pieces), self.direct(q, k, v), atol=0, rtol=0)
        # Persistent cache stores original BF16 values, never QDQ results.
        torch.testing.assert_close(kc[slots], k, atol=0, rtol=0)
        torch.testing.assert_close(vc[slots], v, atol=0, rtol=0)

    def test_head_sharding_and_replicated_kv_preserve_output(self):
        q, k, v = self.values(13)
        full = self.direct(q, k, v)
        # Analogue of TP4 versus TP8: two Q heads share each KV head; splitting
        # to one Q head per shard replicates that KV head without changing scales.
        pieces = []
        for head in range(self.hq):
            kvhead = head // (self.hq // self.hk)
            partial = attention(q[:,head:head+1].transpose(0,1).unsqueeze(0),
                       k[:,kvhead:kvhead+1].transpose(0,1).unsqueeze(0),
                       v[:,kvhead:kvhead+1].transpose(0,1).unsqueeze(0), self.config)
            pieces.append(partial.squeeze(0).transpose(0,1))
        torch.testing.assert_close(torch.cat(pieces,1), full, atol=0, rtol=0)

    def test_shared_prefix_reads_and_new_write_isolation(self):
        q, k, v = self.values(8)
        kc, vc, table = self.buffers()
        table[1, :8] = torch.arange(8)
        table[5, :8] = torch.tensor([0, 1, 2, 3, 14, 15, 16, 17])
        kc[:4], vc[:4] = k[:4], v[:4]
        out = self.call(torch.cat((q[4:], q[4:])), kc, vc, table,
               [RequestSpan(1, 4, 4), RequestSpan(5, 4, 4)],
               new_key=torch.cat((k[4:], k[4:])), new_value=torch.cat((v[4:], v[4:])),
               out_cache_locations=torch.tensor([4, 5, 6, 7, 14, 15, 16, 17]))
        expected = self.direct(q[4:], k, v, 4)
        torch.testing.assert_close(out[:4], expected, atol=0, rtol=0)
        torch.testing.assert_close(out[4:], expected, atol=0, rtol=0)

    def test_explicit_page_layout_preserves_token_addresses(self):
        q, k, v = self.values(9)
        kp = torch.zeros((8, 4, self.hk, self.d), dtype=torch.bfloat16)
        vp = torch.zeros_like(kp)
        kc, vc = token_cache_view(kp, page_size=4), token_cache_view(vp, page_size=4)
        table = torch.full((1, 16), -1, dtype=torch.int32)
        locations = torch.tensor([7, 4, 5, 6, 12, 13, 14, 15, 23], dtype=torch.int32)
        table[0, :9] = locations
        out = self.call(q, kc, vc, table, [RequestSpan(0, 0, 9)], new_key=k, new_value=v,
                        out_cache_locations=locations)
        torch.testing.assert_close(out, self.direct(q, k, v), atol=0, rtol=0)
        torch.testing.assert_close(kp[1, 3], k[0], atol=0, rtol=0)
        with self.assertRaises(ValueError): token_cache_view(kp)
        with self.assertRaises(ValueError): token_cache_view(kp, page_size=8)

    def test_write_locations_are_validated_before_mutation(self):
        q, k, v = self.values(4)
        kc, vc, table = self.buffers()
        table[0, :4] = torch.tensor([4, 7, 8, 9])
        before = kc.clone()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.call(q, kc, vc, table, [RequestSpan(0, 0, 4)], new_key=k, new_value=v,
                      out_cache_locations=torch.tensor([7, 4, 8, 9]))
        torch.testing.assert_close(kc, before, atol=0, rtol=0)

    def test_rejects_cross_request_write_to_shared_prefix(self):
        q, k, v = self.values(2)
        kc, vc, table = self.buffers()
        table[0, :2] = torch.tensor([0, 3]); table[1, :2] = torch.tensor([3, 4])
        with self.assertRaisesRegex(ValueError, 'cached prefix'):
            self.call(q, kc, vc, table, [RequestSpan(0, 1, 1), RequestSpan(1, 1, 1)],
                      new_key=k, new_value=v, out_cache_locations=torch.tensor([3, 4]))
        self.assertEqual(torch.count_nonzero(kc).item(), 0)

    def test_bad_mapping_and_duplicate_requests_fail(self):
        q, _, _ = self.values(2)
        kc, vc, table = self.buffers()
        for row in ([-1, 1], [0, 128], [5, 5]):
            table[0, :2] = torch.tensor(row)
            with self.assertRaises(ValueError):
                self.call(q, kc, vc, table, [RequestSpan(0, 0, 2)])
        with self.assertRaises(ValueError): spans_from_lengths([0, 0], [2, 2], [1, 1])
        with self.assertRaises(TypeError): spans_from_lengths(torch.tensor([0]), [2], [1])
        with self.assertRaises(ValueError): spans_from_lengths([0], [1], [2])

    def test_nondefault_math_and_quantized_cache_rejected(self):
        q, _, _ = self.values(2)
        kc, vc, table = self.buffers(); table[0, :2] = torch.tensor([0, 1])
        spans = [RequestSpan(0, 0, 2)]
        for args in ({'sliding_window': 128}, {'logit_softcap': 30.0}):
            with self.assertRaises(NotImplementedError): self.call(q, kc, vc, table, spans, **args)
        with self.assertRaises(NotImplementedError):
            forward_cached(q, kc, vc, table, spans, core=attention, config=self.config, scale=0.7)
        with self.assertRaisesRegex(ValueError, 'BF16'):
            self.call(q.float(), kc.float(), vc.float(), table, spans)

    def test_idle_and_cpu_metadata(self):
        kc, vc, table = self.buffers()
        q = torch.empty((0, self.hq, self.d), dtype=torch.bfloat16)
        self.assertEqual(self.call(q, kc, vc, table, []).shape, q.shape)
        spans = spans_from_lengths([2, 5], [9, 13], [1, 4])
        self.assertEqual(spans, (RequestSpan(2, 8, 1), RequestSpan(5, 9, 4)))


if __name__ == '__main__':
    unittest.main()
