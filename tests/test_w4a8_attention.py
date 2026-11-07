"""CPU contracts for the four W4A8 attention precision routes."""

from types import SimpleNamespace

import pytest
import torch

from slime_plugins.w4a8_sim.attention import ROUTES, _SimulatedAttention, _attention_forward, fake_quant

NUM_GPUS = 0


def _packed(cu):
    return SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        cp_group=None,
        local_cp_size=1,
    )


def _inputs():
    torch.manual_seed(123)
    query = torch.randn(5, 4, 8).bfloat16().requires_grad_()
    key = torch.randn(5, 2, 8).bfloat16().requires_grad_()
    value = torch.randn(5, 2, 8).bfloat16().requires_grad_()
    return query, key, value, torch.tensor([0, 3, 5], dtype=torch.int32)


@pytest.mark.parametrize("route", ["naive-c8", "naive-c4", "attn-qat-c4"])
def test_software_routes_have_expected_forward_and_gradient_contract(route):
    query, key, value, cu = _inputs()
    output = _SimulatedAttention.apply(query, key, value, cu, 8**-0.5, ROUTES[route].bits, ROUTES[route].qat)
    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.isfinite().all()
    output.float().square().mean().backward()
    assert query.grad is not None and query.grad.isfinite().all()
    assert key.grad is not None and key.grad.isfinite().all()
    assert value.grad is not None and value.grad.isfinite().all()


def test_attn_qat_preserves_forward_and_changes_backward_rule():
    query, key, value, cu = _inputs()
    qat_output = _SimulatedAttention.apply(query, key, value, cu, 8**-0.5, 4, True)
    qat_output.float().square().mean().backward()
    qat_grads = tuple(x.grad.detach().clone() for x in (query, key, value))

    query, key, value, cu = _inputs()
    naive_output = _SimulatedAttention.apply(query, key, value, cu, 8**-0.5, 4, False)
    torch.testing.assert_close(qat_output, naive_output, rtol=0, atol=0)
    naive_output.float().square().mean().backward()
    assert any(not torch.equal(a, b.grad) for a, b in zip(qat_grads, (query, key, value), strict=True))


def test_causal_probability_quantization_ignores_future_tokens_and_packed_sequences():
    query, key, value, cu = _inputs()
    _, records = _attention_forward(query, key, value, cu, 8**-0.5, 4, True)
    for _, _, _, probabilities, probability_fake, _, _ in records:
        assert torch.count_nonzero(probabilities.triu(1)) == 0
        assert torch.count_nonzero(probability_fake.triu(1)) == 0

    first_query, first_key, first_value = query[:3], key[:3], value[:3]
    _, first_records = _attention_forward(
        first_query, first_key, first_value, torch.tensor([0, 3], dtype=torch.int32), 8**-0.5, 4, True
    )
    torch.testing.assert_close(records[0][4], first_records[0][4], rtol=0, atol=0)


def test_naive_c8_quantizes_probability_forward_but_uses_unquantized_probability_for_dv():
    query, key, value, cu = _inputs()
    output = _SimulatedAttention.apply(query, key, value, cu, 8**-0.5, 8, False)
    grad_output = torch.randn_like(output)
    value_grad = torch.autograd.grad(output, value, grad_output)[0]

    _, records = _attention_forward(query, key, value, cu, 8**-0.5, 8, False)
    expected_value_grad = torch.zeros_like(value, dtype=torch.float32)
    for (start, end), (_, _, _, probabilities, probability_fake, _, _) in zip(
        ((0, 3), (3, 5)), records, strict=True
    ):
        assert not torch.equal(probability_fake, probabilities)
        expected_per_query_head = torch.einsum(
            "hlm,lhd->mhd", probabilities, grad_output[start:end].float()
        )
        groups = query.shape[1] // key.shape[1]
        expected = expected_per_query_head.reshape(key.shape[1], groups, end - start, key.shape[2]).sum(1)
        expected_value_grad[start:end] = expected.transpose(0, 1)

    torch.testing.assert_close(value_grad, expected_value_grad.to(value.dtype), rtol=0, atol=0)


def test_fake_quant_formats_and_empty_last_axis():
    tensor = torch.tensor([[0.25, -1.0, 5.5, 0.0]], dtype=torch.bfloat16)
    for bits in (4, 8, 16):
        quantized = fake_quant(tensor, bits)
        assert quantized.dtype == torch.bfloat16
        assert quantized.shape == tensor.shape
        assert quantized.isfinite().all()
    empty = torch.empty((2, 0), dtype=torch.bfloat16)
    assert fake_quant(empty, 4).shape == empty.shape


def test_route_metadata_keeps_c16_and_c4_variants_explicit():
    assert set(ROUTES) == {"c16", "naive-c8", "naive-c4", "attn-qat-c4"}
    assert ROUTES["c16"].bits == 16
    assert ROUTES["naive-c4"].qat is False
    assert ROUTES["attn-qat-c4"].qat is True
    assert ROUTES["naive-c4"].block_size == ROUTES["attn-qat-c4"].block_size == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
