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

## 项目架构（51 步分 4 层）

### 第 1 层：采样 & 分词（Steps 1-9）
最外层的输入输出处理——文本和 token id 互转，控制生成的随机性。
- stable_softmax, temperature, top-k, top-p, 采样, greedy 选择
- 构建词表、编码 prompt、解码 token

### 第 2 层：单序列推理（Steps 10-16）
一次处理一个请求的完整 transformer 推理。
- Token embedding、线性投影
- Causal attention（prefill + decode）
- 连续 KV cache：预先分配一块固定大小的缓冲区

### 第 3 层：PagedAttention（Steps 17-29）
动态 KV cache 分配——vLLM 的核心思想。
- 把 KV cache 拆成固定大小的 block（block_size=8）
- Block 分配器：按需分配/回收
- Paged attention 计算、从多个 block 中收集 K/V
- Sequence state：记录每个请求的 blocks、位置、输出

### 第 4 层：服务调度 & 基准测试（Steps 30-51）
多请求调度、连续批处理、性能测量。
- 静态批处理：所有请求一起开始一起结束
- 连续批处理：请求随时进出，不等慢的
- 优先级队列、抢占、准入控制
- 流式格式、请求生命周期管理
- 指标：TTFT、ITL、吞吐量、延迟百分位（P50/P90/P99）

## 核心概念

### 单请求推理链路
```
用户输入 → 分词 → embedding → attention（带 KV cache）→ 线性投影 → logits → 采样 → token → 解码 → 输出
```

### 思路演进

| 步骤 | 概念 | 解决什么问题 |
|------|------|------------|
| 10-11 | embedding + 线性投影 | 基本的模型计算 |
| 12-13 | KV cache | 避免每步重新算 K/V |
| 14 | Causal attention | 自回归生成 |
| 15-16 | Prefill + Decode | 生成的两个阶段 |
| 17-25 | PagedAttention | 动态分配 KV cache |
| 30 | 单序列生成 | 一条请求的完整生命周期 |
| 31-33 | 静态批处理 | 同时处理多个请求 |
| 34-41 | 连续批处理 | 不等人齐，完成即走 |
| 42-46 | 服务 API | 流式输出、管理生命周期 |
| 47-51 | 基准测试 | 量化服务器性能 |

### 核心取舍：延迟 vs 吞吐量

- **`max_running` 小**（如 1）：TTFT 低，吞吐量低——适合实时交互
- **`max_running` 大**（如 16）：吞吐量高，TTFT 升高——适合批量处理
- **`block_size` 小**：内存碎片少，管理开销大
- **block 总数多**：能服务更多并发，内存占用高

最优值取决于业务场景，没有标准答案。

## 性能指标

| 指标 | 含义 | 衡量什么 |
|------|------|---------|
| TTFT | 首 token 延迟 | 服务器多久开始出字 |
| ITL | token 间延迟 | 生成速度（每秒多少 token） |
| Throughput | 总 token / 总时间 | 系统容量 |
| P50/P90/P99 | 延迟百分位 | 最差情况下的用户体验 |

## 本项目未覆盖的内容

- 模型并行（TP/PP）——把大模型切到多张 GPU 上
- 量化（INT8/FP8）——减小模型大小/加快速度
- Speculative decoding——用小模型辅助大模型加速
- 训练 / 微调 / RLHF——本项目只做了推理

---

Built on Deep-ML.

---

## 项目复盘

### 架构总览

```
用户输入                       输出
  │                             ↑
  ├─① 分词（encode_prompt）      ├─⑨ 解码（decode_tokens）
  │                             │
  ▼                             │
② embedding → ③ attention ──→ ④ linear_projection → logits
                  │
                  ▼
           ⑤ KV cache（普通版或分页版）
                  │
                  ▼
           ⑥ 批处理调度（静态或连续）
                  │
                  ▼
           ⑦ 采样（选下一个词）
```

### 核心概念演进链

每阶段的"问题"就是下一阶段的"动机"：

```
单请求推理（10-16）
  └── 问题：K/V 每次都要重算，浪费
        ↓
KV cache（12-13）
  └── 问题：预先分配一大块内存，用不完浪费，用完了不够
        ↓
PagedAttention（17-29）
  └── 问题：多个请求一起处理更高效，但怎么管理？
        ↓
静态批处理（30-33）
  └── 问题：等人齐了才一起跑，快的等慢的
        ↓
连续批处理（34-41）
  └── 问题：怎么衡量服务器性能？
        ↓
基准测试（47-51）
```

### 面试考点

| 考点 | 对应内容 |
|------|---------|
| Transformer 推理过程 | prefill vs decode、attention 公式、KV cache |
| PagedAttention 原理 | 为什么分 block、怎么分配/回收、vLLM 核心思想 |
| 连续批处理 | 为什么比静态批处理好、何时接新人、何时踢人 |
| 采样策略 | temperature、top-k、top-p、greedy 的区别和 trade-off |
| 推理性能指标 | TTFT（首 token 延迟）、ITL（token 间隔）、吞吐量、P50/P90/P99 |
| 服务调度 | 优先级队列、抢占、block 容量检查 |

### 核心取舍：延迟 vs 吞吐量

- **`max_running` 小**（如 1）：TTFT 低，吞吐量低——适合实时交互
- **`max_running` 大**（如 16）：吞吐量高，TTFT 升高——适合批量处理
- **`block_size` 小**：内存碎片少，管理开销大
- **block 总数多**：能服务更多并发，内存占用高

| 调大参数 | 好处 | 坏处 |
|----------|------|------|
| `max_running` ↑ | 吞吐量 ↑ | TTFT ↑，block 竞争 ↑ |
| `block_size` ↑ | 管理 overhead ↓ | 内存浪费 ↑（碎片） |
| `num_blocks` ↑ | 能同时服务更多人 | 内存占用 ↑ |
| 抢占（preempt）| 高优请求能插队 | 低优先级的被踢，浪费之前算的 K/V |

PagedAttention 就是把 KV cache 分页管理：

```
普通 KV cache： [________________________]  ← 预先挖一个大坑，用不完浪费
PagedAttention： [block0][block1]...[blockN] ← 按需分配，用多少拿多少
```

就像去食堂吃饭：普通 KV cache = 先买一整本饭票，吃不完也浪费；PagedAttention = 吃一碗买一碗，不够再加。

### 性能指标解读

| 指标 | 含义 | 衡量什么 |
|------|------|---------|
| TTFT | 首 token 延迟 | 用户提问后多久开始出字（0.3s 流畅，3s 卡顿）|
| ITL | token 间延迟 | 生成速度，像打字速度（0.05s/字很流畅）|
| Throughput | 总 token / 总时间 | 系统容量，决定能支撑多少用户 |
| P50/P90/P99 | 延迟百分位 | 最差情况下的用户体验 |

不能只看平均——平均 1s 但 P99 是 10s，说明有 1% 的人等了 10 秒。

### Bug 总结

绝大部分 bug 是语法/命名问题，而非逻辑问题：

- 命名不一致（`max_new_token` vs `max_new_tokens`、`kv_block_in_use` vs `kv_blocks_in_use`）
- 拼写错误（`state["block"]` 少 s、`server_state["stream"]` 少 s、`.item()` 写成 `.items()`）
- 缩进错误（decode 部分意外缩进到 `for req in admissions` 循环内部）
- API 不匹配（`time.time()` 精度不够，改为 `time.perf_counter()`）
- Prefill 逻辑缺陷（只填了 `prompt_ids[:-1]` 导致 K/V 少一行）

### 本项目未覆盖的内容

- 模型并行（TP/PP）——把大模型切到多张 GPU 上
- 量化（INT8/FP8）——减小模型大小、加快推理速度
- Speculative decoding——用小模型辅助大模型加速
- 训练 / 微调 / RLHF——本项目仅针对推理