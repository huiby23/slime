"""Exact standalone SG configuration, sent through the same training shim."""
def server_payload(parent_pid):
    args = dict(model_path='/root/hby/models/Qwen3-30B-A3B',
        tokenizer_path='/root/hby/models/Qwen3-30B-A3B', weight_version='0',
        host='127.0.0.1', port=15115, tp_size=8, device='npu', dtype='bfloat16',
        kv_cache_dtype='auto', attention_backend='hby_c4v8_sim',
        prefill_attention_backend='hby_c4v8_sim', decode_attention_backend='hby_c4v8_sim',
        sampling_backend='pytorch', disable_cuda_graph=True, enable_torch_compile=False,
        enable_mixed_chunk=False, speculative_algorithm=None, quantization=None,
        context_length=8192, max_running_requests=4, max_total_tokens=12288,
        chunked_prefill_size=256, mem_fraction_static=0.4, random_seed=1234,
        enable_memory_saver=True, disable_overlap_schedule=True)
    return dict(server_args=args,parent_pid=parent_pid)
