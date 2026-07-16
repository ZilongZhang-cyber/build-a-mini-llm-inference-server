"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np
import time

# Step 1 - stable_softmax
# TODO: implement
def stable_softmax(logits):
    max_val = logits.max(axis=-1,keepdims=True)
    shifted = logits-max_val

    exp_vals = np.exp(shifted)
    
    sum_vals = exp_vals.sum(axis=-1,keepdims=True)
    probs = exp_vals / sum_vals
    return probs

# Step 2 - apply_temperature
# TODO: implement
def apply_temperature(logits,temperature):
    return logits / temperature

# Step 3 - top_k_filter
# TODO: implement
def top_k_filter(logits, k):
    sorted_logits = np.sort(logits,axis=-1)
    threshold = sorted_logits[...,-k]
    threshold = threshold[...,np.newaxis]

    result = np.where(logits < threshold,-np.inf,logits)
    return result

# Step 4 - top_p_filter
# TODO: implement
def top_p_filter(logits,p):
    probs = stable_softmax(logits)

    sorted_idx = np.argsort(-probs,axis=-1)
    sorted_probs = np.take_along_axis(probs,sorted_idx,axis=-1)

    cumsum = np.cumsum(sorted_probs,axis=-1)

    sorted_keep = (cumsum - sorted_probs) < p

    keep_mask = np.zeros_like(probs,dtype=bool)
    np.put_along_axis(keep_mask,sorted_idx,sorted_keep,axis=-1)

    return np.where(keep_mask,logits,-np.inf)

# Step 5 - sample_from_probs
# TODO: implement
def sample_from_probs(probs,rng):
    return rng.choice(len(probs),p=probs)
# Step 6 - greedy_select
# TODO: implement
def greedy_select(logits):
    return logits.argmax(axis = -1)
# Step 7 - build_vocab
# TODO: implement
def build_vocab(corpus,specials):
    #拆分句子
    all_words = []
    for sentence in corpus:
        all_words.extend(sentence.split())
    
    #去重
    seen = set()
    unique_words = []
    for word in all_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    #特殊符号token编号
    token_to_id = {}
    id_to_token = {}
    for i,tok in enumerate(specials):
        token_to_id[tok] = i
        id_to_token[i]   = tok
    
    #普通单词后面编号
    for word in unique_words:
        idx = len(token_to_id)
        token_to_id[word] = idx
        id_to_token[idx]  = word
    
    return {"token_to_id":token_to_id,"id_to_token":id_to_token}


# Step 8 - encode_prompt
# TODO: implement
def encode_prompt(prompt,vocab,add_bos = True):
    token_to_id = vocab["token_to_id"]
    unk_id = token_to_id["<unk>"]

    words = prompt.split()
    
    ids = []
    for word in words:
        if word in token_to_id:
            ids.append(token_to_id[word])
        else:
            ids.append(unk_id)
    
    if(add_bos):
        bos_id = token_to_id["<bos>"]
        ids = [bos_id] + ids
    
    return ids

# Step 9 - decode_tokens
# TODO: implement
def decode_tokens(token_ids,vocab,skip_special=True):
    id_to_token = vocab["id_to_token"]
    specials = {"<pad>","<bos>","<eos>","<unk>"}

    words = []
    for tid in token_ids:
        token = id_to_token[tid]
        if skip_special and token in specials:
            continue
        words.append(token)
    
    return " ".join(words)

# Step 10 - embed_tokens
# TODO: implement
def embed_tokens(token_ids,embedding_matrix):
    return embedding_matrix[token_ids]
# Step 11 - linear_projection
# TODO: implement
def linear_projection(hidden,W_out):
    return hidden @ W_out
# Step 12 - init_kv_cache
# TODO: implement
def init_kv_cache(max_seq_len,d_model):
    return {
        "K":np.zeros((max_seq_len,d_model)),
        "V":np.zeros((max_seq_len,d_model))
    }

# Step 13 - append_kv
# TODO: implement
def append_kv(cache,K_new,V_new,pos):
    cache["K"][pos] = K_new
    cache["V"][pos] = V_new
    return cache

# Step 14 - causal_attention
# TODO: implement
def causal_attention(X,Wq,Wk,Wv,Wo,cache=None,pos=0):
    d_model = X.shape[-1]
    seq_len = X.shape[0]

    Q     = X @ Wq
    K_new = X @ Wk
    V_new = X @ Wv

    if cache is not None and pos>0 :
        K_all = np.concatenate([cache["K"][:pos],K_new],axis = 0)
        V_all = np.concatenate([cache["V"][:pos],V_new],axis = 0)
    else:
        K_all = K_new
        V_all = V_new
    
    scale = 1.0 / np.sqrt(d_model)
    scores = Q @ K_all.T * scale

    q_positions = np.arange(pos,pos + seq_len).reshape(-1,1)#列向量控制的是行数
    k_positions = np.arange(K_all.shape[0]).reshape(1,-1)   #行向量控制的是列数
    mask = k_positions > q_positions                        #自动进行广播，广播成矩阵
    scores = np.where(mask,-np.inf,scores)                  #where(条件，A，B)

    attn = stable_softmax(scores)
    hidden = attn @ V_all
    output = hidden @ Wo
    
    if cache is not None:
        for i in range(seq_len):
            cache["K"][pos+i] = K_new[i]
            cache["V"][pos+i] = V_new[i]

    return output

# Step 15 - model_prefill
# TODO: implement
def model_prefill(token_ids,params,cache):
    X = embed_tokens(token_ids,params["embedding"])
    output = causal_attention(X,
                              params["Wq"],params["Wk"],params["Wv"],params["Wo"],
                              cache = cache,
                              pos = 0)
    last_hidden = output[-1]
    logits = linear_projection(last_hidden,params["W_out"])
    return logits

# Step 16 - model_decode_step
# TODO: implement
def model_decode_step(token_id,params,cache,pos):
    X = embed_tokens(np.array([token_id]),params["embedding"])
    output = causal_attention(
        X,
        params["Wq"],params["Wk"],params["Wv"],params["Wo"],
        cache = cache,
        pos = pos
    )
    logits = linear_projection(output[0],params["W_out"])
    return logits

# Step 17 - blocks_needed
# TODO: implement
def blocks_needed(seq_len,block_size):
    return (seq_len + block_size -1) // block_size

# Step 18 - init_block_allocator
# TODO: implement
def init_block_allocator(num_blocks,block_size,d_model):
    return {
        "free_blocks" : list(range(num_blocks)),
        "block_size" : block_size,
        "d_model" : d_model,
        "num_blocks" : num_blocks
    }

# Step 19 - allocate_block
# TODO: implement
def allocate_block(allocator):
    if not allocator["free_blocks"]:
        return None
    block_id = allocator["free_blocks"].pop(0)

    return {
        "id" : block_id,
        "K" : np.zeros((allocator["block_size"],allocator["d_model"])),
        "V" : np.zeros((allocator["block_size"],allocator["d_model"]))
    }

# Step 20 - free_block
# TODO: implement
def free_block(allocator,block):
    block_id = block["id"]
    allocator["free_blocks"].append(block_id)

# Step 21 - append_to_paged_cache
# TODO: implement
def append_to_paged_cache(blocks,K_new,V_new,pos,block_size):
    block_id = pos // block_size
    offset = pos % block_size
    blocks[block_id]["K"][offset] = K_new
    blocks[block_id]["V"][offset] = V_new
 
# Step 22 - gather_kv_from_blocks
# TODO: implement
def gather_kv_from_blocks(blocks,seq_len,block_size):
    K_parts = []
    V_parts = []
    remaining = seq_len
    for block in blocks:
        if remaining <= 0:
            break
        take = min(remaining,block_size)
        K_parts.append(block["K"][:take])
        V_parts.append(block["V"][:take])
        remaining -= take
    
    return np.concatenate(K_parts,axis = 0),np.concatenate(V_parts,axis = 0)

# Step 23 - paged_attention_step
# TODO: implement
def paged_attention_step(X,Wq,Wk,Wv,Wo,blocks,seq_len,block_size):
    Q = X @ Wq
    K_new = X @ Wk
    V_new = X @ Wv

    K_hist,V_hist = gather_kv_from_blocks(blocks,seq_len,block_size)

    K_all = np.concatenate([K_hist,K_new],axis = 0)
    V_all = np.concatenate([V_hist,V_new],axis = 0)

    scale = 1.0 / np.sqrt(X.shape[-1])
    scores = Q @ K_all.T *scale

    attn = stable_softmax(scores)
    hidden = attn @ V_all
    output = hidden @ Wo

    append_to_paged_cache(blocks,K_new[0],V_new[0],seq_len,block_size)

    return output

# Step 24 - free_sequence_blocks
# TODO: implement
def free_sequence_blocks(allocator,blocks):
    for block in blocks:
        free_block(allocator,block)

# Step 25 - kv_blocks_in_use
# TODO: implement
def kv_blocks_in_use(allocator):
    return allocator["num_blocks"] - len(allocator["free_blocks"])

# Step 26 - make_request
# TODO: implement
def make_request(request_id,prompt_ids,max_new_tokens,sampling_params):
    return {
        "id" : request_id,
        "prompt_ids" : prompt_ids,
        "max_new_tokens" : max_new_tokens,
        "sampling_params" : sampling_params
    }

# Step 27 - init_sequence_state
# TODO: implement
def init_sequence_state(request,allocator,eos_id):
    prompt_ids = request["prompt_ids"]
    block_size = allocator["block_size"]

    n_blocks = blocks_needed(len(prompt_ids),block_size)
    
    blocks = []
    for _ in range(n_blocks):
        blk = allocate_block(allocator)
        blocks.append(blk)
    
    return {
        "request" : request,
        "blocks" : blocks,
        "pos" : len(prompt_ids),
        "output_ids" : [],
        "done" : False,
        "eos_token_id" : eos_id
    }

# Step 28 - sequence_decode_step
# TODO: implement
def sequence_decode_step(token_id,sequence_state,params):
    X = embed_tokens(np.array([token_id]),params["embedding"])

    hidden = paged_attention_step(
        X,
        params["Wq"],params["Wk"],params["Wv"],params["Wo"],
        blocks = sequence_state["blocks"],
        seq_len = sequence_state["pos"],
        block_size = sequence_state["blocks"][0]["K"].shape[0]
    )
    
    logits = linear_projection(hidden[0],params["W_out"])

    sequence_state["pos"] += 1
    return logits
    
# Step 29 - is_sequence_done
# TODO: implement
def is_sequence_done(sequence_state):
    return sequence_state["done"]
    
# Step 30 - generate_single_sequence
# TODO: implement
def generate_single_sequence(request,params,eos_id,rng):
    cache = init_kv_cache(params["max_seq_len"],params["d_model"])

    logits = model_prefill(request["prompt_ids"],params,cache)
    next_tok = greedy_select(logits)

    pos = len(request["prompt_ids"])
    output_ids = []
    
    while len(output_ids) < request["max_new_tokens"]:
        output_ids.append(next_tok)

        logits = model_decode_step(next_tok,params,cache,pos)
        pos += 1

        sampling = request["sampling_params"]
        logits = apply_temperature(logits,sampling["temperature"])
        logits = top_k_filter(logits,sampling["top_k"])
        logits = top_p_filter(logits,sampling["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs,rng)

        if next_tok == eos_id:
            break
    return output_ids

# Step 31 - build_batch_step_input
# TODO: implement
def build_batch_step_input(states):
    token_ids = []
    for state in states:
        if not state["done"]:
            token_ids.append(state["output_ids"][-1])
    if not token_ids:
        return None
    return np.array(token_ids)

# Step 32 - batched_decode_step
# TODO: implement
def batched_decode_step(batch_tokens,states,params):
    all_logits = []
    j = 0
    for i,state in enumerate(states):
        if state["done"]:
            all_logits.append(None)
            continue
        token_id = batch_tokens[j]
        j += 1
        logits = sequence_decode_step(token_id,state,params)
        all_logits.append(logits)
    return all_logits

# Step 33 - static_batch_generate
# TODO: implement
def static_batch_generate(requests,params,allocator,eos_id,rng):
    states = [init_sequence_state(req,allocator,eos_id) for req in requests]

    for state in states:
        prompt_ids = state["request"]["prompt_ids"]
        block_size = state["blocks"][0]["K"].shape[0]

        for i,tid in enumerate(prompt_ids):
            X = embed_tokens(np.array([tid]),params["embedding"])
            append_to_paged_cache(state["blocks"],(X @ params["Wk"])[0],(X @ params["Wv"])[0],i,block_size)

        last_tid = prompt_ids[-1]
        X = embed_tokens(np.array([last_tid]),params["embedding"])
        K_hist,V_hist = gather_kv_from_blocks(state["blocks"],len(prompt_ids),block_size)
        Q = X @ params["Wq"]
        scale = 1.0 / np.sqrt(X.shape[-1])
        scores = Q @ K_hist.T * scale
        attn = stable_softmax(scores)
        hidden = attn @ V_hist
        logits = linear_projection(hidden[0],params["W_out"])

        sampling = state["request"]["sampling_params"]
        logits = apply_temperature(logits,sampling["temperature"])
        logits = top_k_filter(logits,sampling["top_k"])
        logits = top_p_filter(logits,sampling["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs,rng)

        state["output_ids"] = [next_tok]

        if next_tok == eos_id:
            state["done"] = True
    
    while not all(s["done"] for s in states):
        batch_tokens = build_batch_step_input(states)
        if batch_tokens is None:
            break
        all_logits = batched_decode_step(batch_tokens,states,params)

        for i,state in enumerate(states):
            if state["done"]:
                continue

            logits = all_logits[i]
            sampling = state["request"]["sampling_params"]
            logits = apply_temperature(logits,sampling["temperature"])    
            logits = top_k_filter(logits,sampling["top_k"])
            logits = top_p_filter(logits,sampling["top_p"])
            probs = stable_softmax(logits)
            next_tok = sample_from_probs(probs,rng)

            state["output_ids"].append(next_tok)

            if next_tok == eos_id or len(state["output_ids"]) >=state["request"]["max_new_tokens"]:
                state["done"] = True
    
    results = []
    for state in states:
        results.append(state["output_ids"][:])
        free_sequence_blocks(allocator,state["blocks"])

    return results
# Step 34 - has_free_capacity
# TODO: implement
def has_free_capacity(server_state):
    return len(server_state["running"]) < server_state["max_running"]

# Step 35 - continuous_batch_step
# TODO: implement
def continuous_batch_step(server_state,params,allocator,sampling_cfg):
    eos_id = server_state["eos_token_id"]

    still_running = []
    for state in server_state["running"]:
        if state["done"]:
            free_sequence_blocks(allocator,state["blocks"])
            rid = state["request"]["id"]
            server_state["outputs"][rid] = state["output_ids"]
        else:
            still_running.append(state)
    server_state["running"] = still_running

    while has_free_capacity(server_state) and server_state["waiting_heap"]:
        req = server_state["waiting_heap"].pop(0)
        state = init_sequence_state(req,allocator,eos_id)

        prompt_ids = state["request"]["prompt_ids"]
        block_size = state["blocks"][0]["K"].shape[0]
        for i,tid in enumerate(prompt_ids):
            X = embed_tokens(np.array([tid]),params["embedding"])
            append_to_paged_cache(
                state["blocks"],
                (X @ params["Wk"])[0],
                (X @ params["Wv"])[0],
                i,block_size,
            )

        last_tid = prompt_ids[-1]
        X = embed_tokens(np.array([last_tid]),params["embedding"])
        K_hist,V_hist = gather_kv_from_blocks(state["blocks"],len(prompt_ids),block_size)
        Q = X @ params["Wq"]
        scale = 1.0 / np.sqrt(X.shape[-1])
        scores = Q @ K_hist.T * scale
        attn = stable_softmax(scores)
        hidden = attn @ V_hist
        logits = linear_projection(hidden[0],params["W_out"])

        logits = apply_temperature(logits, sampling_cfg["temperature"])
        logits = top_k_filter(logits, sampling_cfg["top_k"])
        logits = top_p_filter(logits, sampling_cfg["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs, sampling_cfg["rng"])

        state["output_ids"] = [next_tok]
        if next_tok == eos_id:
            state["done"] = True

        server_state["running"].append(state)

    if not server_state["running"]:
        return

    batch_tokens = build_batch_step_input(server_state["running"])
    if server_state["tick"] < 5 and batch_tokens is not None:
        print(f"DEBUG decode: batch_tokens={batch_tokens.tolist()} states={[s['request']['id'] for s in server_state['running']]}")
    if batch_tokens is None:
        return

    all_logits = batched_decode_step(batch_tokens,server_state["running"],params)

    for i,state in enumerate(server_state["running"]):
        if state["done"]:
            continue
        logits = all_logits[i]
        logits = apply_temperature(logits,sampling_cfg["temperature"])
        logits = top_k_filter(logits, sampling_cfg["top_k"])
        logits = top_p_filter(logits, sampling_cfg["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs, sampling_cfg["rng"])

        state["output_ids"].append(next_tok)

        if next_tok == eos_id or len(state["output_ids"]) >= state["request"]["max_new_tokens"]:
            state["done"] = True
    server_state["tick"] += 1

# Step 36 - run_continuous_batching
# TODO: implement
def run_continuous_batching(server_state,params,allocator,sampling_cfg,max_steps):
    for step in range(max_steps):
        continuous_batch_step(server_state,params,allocator,sampling_cfg)

        if not server_state["waiting_heap"] and not server_state["running"]:
            break
    
# Step 37 - priority_queue_push
# TODO: implement
def priority_queue_push(server_state,request,priority):
    heap = server_state["waiting_heap"]
    request["priority"] = priority
    i=0
    while i<len(heap) and heap[i]["priority"] >= priority:
        i+=1
    heap.insert(i,request)

# Step 38 - priority_queue_pop
# TODO: implement
def priority_queue_pop(server_state):
    if not server_state["waiting_heap"]:
        return None
    return server_state["waiting_heap"].pop(0)

# Step 39 - select_admissions
# TODO: implement
def select_admissions(server_state,allocator):
    admissions = []
    free_blocks = allocator["num_blocks"] - kv_blocks_in_use(allocator)
    while has_free_capacity(server_state) and server_state["waiting_heap"]:
        req = server_state["waiting_heap"][0]
        n_blocks = blocks_needed(len(req["prompt_ids"]),allocator["block_size"])

        if n_blocks <= free_blocks:
            free_blocks -= n_blocks
            admissions.append(server_state["waiting_heap"].pop(0))
        else:
            break

    return admissions

# Step 40 - preempt_sequence
# TODO: implement
def preempt_sequence(server_state,allocator):
    running = server_state["running"]
    if not running:
        return None
    
    lowest = min(running,key=lambda s: s["request"].get("priority",0))
    free_sequence_blocks(allocator,lowest["blocks"])
    running.remove(lowest)
    server_state["waiting_heap"].insert(0,lowest["request"])

    return lowest["request"]["id"]

# Step 41 - schedule_step
# TODO: implement
def schedule_step(server_state,params,allocator,sampling_cfg):
    eos_id = server_state["eos_token_id"]

    still_running = []
    for state in server_state["running"]:
        if state["done"]:
            free_sequence_blocks(allocator,state["blocks"])
            rid = state["request"]["id"]
            server_state["outputs"][rid] = state["output_ids"]
            if "timestamps" in server_state and rid in server_state["timestamps"]:
                server_state["timestamps"][rid]["completed_at"] = time.perf_counter()
        else:
            still_running.append(state)
    server_state["running"] = still_running

    admissions = select_admissions(server_state,allocator)
    for req in admissions:
        state = init_sequence_state(req,allocator,eos_id)
        prompt_ids = state["request"]["prompt_ids"]
        block_size = state["blocks"][0]["K"].shape[0]

        # 手动填入所有 prompt token 的 K/V
        for i,tid in enumerate(prompt_ids):
            X = embed_tokens(np.array([tid]),params["embedding"])
            append_to_paged_cache(
                state["blocks"],
                (X @ params["Wk"])[0],
                (X @ params["Wv"])[0],
                i,block_size,
            )

        # 手动算最后一个 prompt token 的 attention（不用 paged_attention_step）
        last_tid = prompt_ids[-1]
        X = embed_tokens(np.array([last_tid]),params["embedding"])
        K_hist, V_hist = gather_kv_from_blocks(state["blocks"], len(prompt_ids), block_size)
        Q = X @ params["Wq"]
        scale = 1.0 / np.sqrt(X.shape[-1])
        scores = Q @ K_hist.T * scale
        attn = stable_softmax(scores)
        hidden = attn @ V_hist
        logits = linear_projection(hidden[0],params["W_out"])

        logits = apply_temperature(logits, sampling_cfg["temperature"])
        logits = top_k_filter(logits, sampling_cfg["top_k"])
        logits = top_p_filter(logits, sampling_cfg["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs, sampling_cfg["rng"])

        state["output_ids"] = [next_tok]
        if next_tok == eos_id:
            state["done"] = True

        # 记录首 token 时间
        rid = state["request"]["id"]
        if "timestamps" in server_state and rid in server_state["timestamps"]:
            server_state["timestamps"][rid]["first_token_at"] = time.perf_counter()

        server_state["running"].append(state)

    if not server_state["running"]:
        server_state["tick"] += 1
        return
    batch_tokens = build_batch_step_input(server_state["running"])
    if batch_tokens is None:
        server_state["tick"] += 1
        return
    all_logits = batched_decode_step(batch_tokens,server_state["running"],params)

    for i, state in enumerate(server_state["running"]):
        if state["done"]:
            continue

        logits = all_logits[i]
        logits = apply_temperature(logits, sampling_cfg["temperature"])
        logits = top_k_filter(logits, sampling_cfg["top_k"])
        logits = top_p_filter(logits, sampling_cfg["top_p"])
        probs = stable_softmax(logits)
        next_tok = sample_from_probs(probs, sampling_cfg["rng"])

        state["output_ids"].append(next_tok)

        if next_tok == eos_id or len(state["output_ids"]) >= state["request"]["max_new_tokens"]:
            state["done"] = True
    server_state["tick"] += 1
# Step 42 - format_stream_chunk
# TODO: implement
def format_stream_chunk(request_id,token,done):
    return {
        "id":request_id,
        "token":token,
        "done":done,
    }
# Step 43 - submit_request
# TODO: implement
def submit_request(server_state,prompt,max_new_tokens,priority,vocab):
    rid = f"req-{server_state['next_request_id']}"
    server_state["next_request_id"] += 1

    prompt_ids = encode_prompt(prompt,vocab,add_bos=True)

    sampling_params = {
        "temperature" : 1.0,
        "top_k" : 5,
        "top_p" : 0.9,
        "rng" : server_state["rng"],
    }
    
    req = make_request(rid,prompt_ids,max_new_tokens,sampling_params)
    priority_queue_push(server_state,req,priority)
    server_state["streams"][rid] = []

    if "timestamps" not in server_state:
        server_state["timestamps"] = {}
    server_state["timestamps"][rid] = {"submitted_at": time.perf_counter()}

    return rid
# Step 44 - drive_until_complete
# TODO: implement
def drive_until_complete(server_state,params,allocator,sampling_cfg,vocab,max_steps):
    for _ in range(max_steps):
        schedule_step(server_state,params,allocator,sampling_cfg)
        
        if not server_state["waiting_heap"] and not server_state["running"]:
            break
# Step 45 - collect_request_output
# TODO: implement
def collect_request_output(server_state,request_id):
    output_ids = server_state["outputs"].get(request_id)
    if output_ids is None:
        return None
    return {
        "request_id": request_id,
        "output_ids": output_ids,
    }

# Step 46 - build_completion_response
# TODO: implement
def build_completion_response(server_state,request_id,vocab):
    output = collect_request_output(server_state,request_id)
    if output is None:
        return None
    text = decode_tokens(output["output_ids"],vocab)
    return {
        "request_id":request_id,
        "text": text,
        "token_count": len(output["output_ids"]),
    }
# Step 47 - time_to_first_token
# TODO: implement
def time_to_first_token(server_state,request_id):
    timestamps = server_state.get("timestamps",{}).get(request_id)
    if timestamps is None:
        return None
    return timestamps.get("first_token_at",0) - timestamps.get("submitted_at",0)
# Step 48 - inter_token_latency
# TODO: implement
def inter_token_latency(server_state,request_id):
    timestamps = server_state.get("timestamps",{}).get(request_id)
    output = server_state["outputs"].get(request_id)
    if timestamps is None or output is None:
        return
    
    completed = timestamps.get("completed_at")
    first = timestamps.get("first_token_at")
    if completed is None or first is None:
        return 
    total_time = completed - first
    n_intervals = len(output)-1
    if n_intervals < 1:
        return 
    
    return total_time / n_intervals
# Step 49 - aggregate_throughput
# TODO: implement
def aggregate_throughput(server_state):
    outputs = server_state.get("outputs",{})
    timestamps = server_state.get("timestamps",{})
    if not outputs or not timestamps:
        return None
    total_tokens = 0
    min_submit = float("inf")
    max_complete = 0

    for rid,ids in outputs.items():
        total_tokens += len(ids)
        ts = timestamps.get(rid)
        if ts:
            if ts.get("submitted_at",float("inf")) < min_submit:
                min_submit = ts["submitted_at"]
            if ts.get("completed_at",0) > max_complete:
                max_complete = ts["completed_at"]
    
    total_time = max_complete - min_submit
    if total_time <= 0:
        return None
    
    return {
        "total_tokens": total_tokens,
        "total_time": round(total_time,4),
        "throughput": round(total_tokens / total_time,2),
        "num_requests": len(outputs),
    }
# Step 50 - latency_percentiles
# TODO: implement
def latency_percentiles(server_state):
    timestamps = server_state.get("timestamps",{})
    latencies = []
    for _,ts in timestamps.items():
        sub = ts.get("submitted_at")
        com = ts.get("completed_at")
        if sub is not None and com is not None:
            latencies.append(com - sub)
    if not latencies:
        return
    
    latencies.sort()
    n = len(latencies)

    def percentile(p):
        k = (n-1)*p/100
        f = int(k)
        c = k - f
        if f+1 < n:
            return latencies[f] * (1-c) + latencies[f+1] * c
        return latencies[f]
    
    return {
        "p50":round(percentile(50),4),
        "p90":round(percentile(90),4),
        "p99":round(percentile(99),4),
        "min":round(latencies[0],4),
        "max":round(latencies[-1],4)
    }
# Step 51 - run_throughput_latency_benchmark
# TODO: implement
def run_throughput_latency_benchmark(params,allocator,vocab,prompts,sampling_cfg,
                                     max_new_tokens,max_steps):
    eos_id = vocab["token_to_id"]["<eos>"]
    server_state = {
        "waiting_heap": [],
        "running": [],
        "next_request_id":0,
        "outputs":{},
        "streams":{},
        "timestamps":{},
        "events":[],
        "eos_token_id":eos_id,
        "block_size":allocator["block_size"],
        "max_running":4,
        "rng":sampling_cfg["rng"],
        "tick":0,
    }

    for i,p in enumerate(prompts):
        submit_request(server_state,p,max_new_tokens,i,vocab)
    drive_until_complete(server_state,params,allocator,sampling_cfg,vocab,max_steps)
    ttft_list = []
    itl_list = []
    for rid in server_state["outputs"]:
        ttft = time_to_first_token(server_state,rid)
        itl = inter_token_latency(server_state,rid)
        if ttft is not None:
            ttft_list.append(ttft)
        if itl is not None:
            itl_list.append(itl)
    
    agg = aggregate_throughput(server_state)
    perc = latency_percentiles(server_state)

    return {
        "avg_ttft": round(sum(ttft_list)/len(ttft_list),4) if ttft_list else None,
        "avg_itl": round(sum(itl_list)/len(itl_list),4) if itl_list else None,
        "throughput":agg["throughput"] if agg else None,
        "p50_latency": perc["p50"] if perc else None,
        "p90_latency": perc["p90"] if perc else None,
        "p99_latency": perc["p99"] if perc else None,
        "num_requests": len(prompts),
        "total_tokens":agg["total_tokens"] if agg else 0,
    }