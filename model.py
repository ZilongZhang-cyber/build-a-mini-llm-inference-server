"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

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
# Step 17 - blocks_needed (not yet solved)
# TODO: implement

# Step 18 - init_block_allocator (not yet solved)
# TODO: implement

# Step 19 - allocate_block (not yet solved)
# TODO: implement

# Step 20 - free_block (not yet solved)
# TODO: implement

# Step 21 - append_to_paged_cache (not yet solved)
# TODO: implement

# Step 22 - gather_kv_from_blocks (not yet solved)
# TODO: implement

# Step 23 - paged_attention_step (not yet solved)
# TODO: implement

# Step 24 - free_sequence_blocks (not yet solved)
# TODO: implement

# Step 25 - kv_blocks_in_use (not yet solved)
# TODO: implement

# Step 26 - make_request (not yet solved)
# TODO: implement

# Step 27 - init_sequence_state (not yet solved)
# TODO: implement

# Step 28 - sequence_decode_step (not yet solved)
# TODO: implement

# Step 29 - is_sequence_done (not yet solved)
# TODO: implement

# Step 30 - generate_single_sequence (not yet solved)
# TODO: implement

# Step 31 - build_batch_step_input (not yet solved)
# TODO: implement

# Step 32 - batched_decode_step (not yet solved)
# TODO: implement

# Step 33 - static_batch_generate (not yet solved)
# TODO: implement

# Step 34 - has_free_capacity (not yet solved)
# TODO: implement

# Step 35 - continuous_batch_step (not yet solved)
# TODO: implement

# Step 36 - run_continuous_batching (not yet solved)
# TODO: implement

# Step 37 - priority_queue_push (not yet solved)
# TODO: implement

# Step 38 - priority_queue_pop (not yet solved)
# TODO: implement

# Step 39 - select_admissions (not yet solved)
# TODO: implement

# Step 40 - preempt_sequence (not yet solved)
# TODO: implement

# Step 41 - schedule_step (not yet solved)
# TODO: implement

# Step 42 - format_stream_chunk (not yet solved)
# TODO: implement

# Step 43 - submit_request (not yet solved)
# TODO: implement

# Step 44 - drive_until_complete (not yet solved)
# TODO: implement

# Step 45 - collect_request_output (not yet solved)
# TODO: implement

# Step 46 - build_completion_response (not yet solved)
# TODO: implement

# Step 47 - time_to_first_token (not yet solved)
# TODO: implement

# Step 48 - inter_token_latency (not yet solved)
# TODO: implement

# Step 49 - aggregate_throughput (not yet solved)
# TODO: implement

# Step 50 - latency_percentiles (not yet solved)
# TODO: implement

# Step 51 - run_throughput_latency_benchmark (not yet solved)
# TODO: implement

