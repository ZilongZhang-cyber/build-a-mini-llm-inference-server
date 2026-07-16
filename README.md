# Build a Mini LLM Inference Server

Construct a complete LLM inference stack from scratch: sampling, tokenization, a tiny transformer with KV caching, a paged attention allocator, continuous batching with scheduling, a streaming serving API, and a throughput/latency benchmark harness. This mirrors the architecture of modern serving systems like vLLM at a digestible scale.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** stable_softmax
- [x] **2.** apply_temperature
- [x] **3.** top_k_filter
- [x] **4.** top_p_filter
- [x] **5.** sample_from_probs
- [x] **6.** greedy_select
- [x] **7.** build_vocab
- [x] **8.** encode_prompt
- [x] **9.** decode_tokens
- [x] **10.** embed_tokens
- [x] **11.** linear_projection
- [x] **12.** init_kv_cache
- [x] **13.** append_kv
- [x] **14.** causal_attention
- [x] **15.** model_prefill
- [x] **16.** model_decode_step
- [x] **17.** blocks_needed
- [x] **18.** init_block_allocator
- [x] **19.** allocate_block
- [x] **20.** free_block
- [x] **21.** append_to_paged_cache
- [x] **22.** gather_kv_from_blocks
- [x] **23.** paged_attention_step
- [x] **24.** free_sequence_blocks
- [x] **25.** kv_blocks_in_use
- [x] **26.** make_request
- [x] **27.** init_sequence_state
- [x] **28.** sequence_decode_step
- [x] **29.** is_sequence_done
- [x] **30.** generate_single_sequence
- [x] **31.** build_batch_step_input
- [x] **32.** batched_decode_step
- [x] **33.** static_batch_generate
- [x] **34.** has_free_capacity
- [x] **35.** continuous_batch_step
- [x] **36.** run_continuous_batching
- [x] **37.** priority_queue_push
- [x] **38.** priority_queue_pop
- [x] **39.** select_admissions
- [x] **40.** preempt_sequence
- [x] **41.** schedule_step
- [x] **42.** format_stream_chunk
- [x] **43.** submit_request
- [x] **44.** drive_until_complete
- [x] **45.** collect_request_output
- [x] **46.** build_completion_response
- [x] **47.** time_to_first_token
- [x] **48.** inter_token_latency
- [x] **49.** aggregate_throughput
- [x] **50.** latency_percentiles
- [x] **51.** run_throughput_latency_benchmark

---

## Project Architecture (51 Steps)

The project is organized into 4 layers, each solving a specific problem:

### Layer 1: Sampling & Tokenization (Steps 1-9)
The outermost IO layer — convert between text and token ids, control randomness of generation.
- Stable softmax, temperature, top-k, top-p filtering, sampling, greedy selection
- Build vocab, encode prompt, decode tokens

### Layer 2: Single-Sequence Inference (Steps 10-16)
Core transformer inference for one request at a time.
- Token embedding, linear projection
- Causal attention (prefill + decode)
- Contiguous KV cache: pre-allocate a fixed-size buffer

### Layer 3: PagedAttention (Steps 17-29)
Dynamic KV cache allocation — the key idea behind vLLM.
- Split KV cache into fixed-size blocks (block_size=8)
- Block allocator: allocate/free on demand
- Paged attention step, gather KV from multiple blocks
- Sequence state: tracks blocks, position, output per request

### Layer 4: Serving & Benchmarking (Steps 30-51)
Multi-request scheduling, continuous batching, and performance measurement.
- Static batch: all requests start and end together
- Continuous batch: requests enter/leave dynamically (no waiting for stragglers)
- Priority queue, preemption, admission control
- Streaming format, request lifecycle
- Metrics: TTFT, ITL, throughput, latency percentiles (P50/P90/P99)

## Core Concepts

### Inference Chain (Single Request)
```
User input → tokenize → embed → attention (with KV cache) → linear projection → logits → sample → token → decode → output
```

### Evolution of Ideas

| Step | Idea | Problem It Solves |
|------|------|-------------------|
| 10-11 | embed + linear | Basic model computation |
| 12-13 | KV cache | Avoid recomputing K/V every step |
| 14 | Causal attention | Auto-regressive generation |
| 15-16 | Prefill + Decode | Two phases of generation |
| 17-25 | PagedAttention | Dynamic KV cache allocation |
| 30 | Single sequence | One complete request lifecycle |
| 31-33 | Static batching | Process multiple requests together |
| 34-41 | Continuous batching | No waiting for stragglers |
| 42-46 | Serving API | Stream results, manage lifecycle |
| 47-51 | Benchmark | Quantify performance |

### Key Trade-off: Latency vs Throughput

- **Low `max_running`** (e.g. 1): Fast TTFT, low throughput — good for interactive use
- **High `max_running`** (e.g. 16): High throughput, higher TTFT — good for batch processing
- **Smaller `block_size`**: Less memory waste, more management overhead
- **More blocks**: Serve more concurrent requests, higher memory usage

The optimal balance depends on the business scenario — there is no one-size-fits-all answer.

## Performance Metrics

| Metric | Meaning | What It Measures |
|--------|---------|-----------------|
| TTFT | Time to First Token | How fast the server starts responding |
| ITL | Inter-Token Latency | Generation speed (tokens/second per request) |
| Throughput | Total tokens / total time | System capacity |
| P50/P90/P99 | Latency percentiles | Worst-case user experience |

## Limitations (What This Project Does NOT Cover)

- Model parallelism (TP/PP) — splitting a large model across GPUs
- Quantization (INT8/FP8) — reducing model size/speed
- Speculative decoding — using a draft model to accelerate
- Training / fine-tuning / RLHF — this is inference only

---

Built on Deep-ML.
