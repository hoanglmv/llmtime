import torch
import numpy as np
from jax import grad, vmap
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

loaded = {}

# Mapping từ tên gọi trong code sang đường dẫn HuggingFace
MODEL_MAPPING = {
    # Llama 2
    "7b": "meta-llama/Llama-2-7b-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "70b": "meta-llama/Llama-2-70b-hf",
    "7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    
    # Llama 3.1 & 3.2
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}

def get_hf_path(model_name):
    """Trả về đường dẫn HuggingFace dựa trên tên model"""
    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]
    
    # Fallback: Trả về nguyên gốc nếu người dùng truyền path trực tiếp
    if "/" in model_name: 
        return model_name
        
    name_parts = model_name.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1
    chat_str = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat_str}hf"

def get_tokenizer(model_name):
    hf_path = get_hf_path(model_name)

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)
    except:
        # Fallback nếu model mới bắt buộc dùng fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=True)

    # Xử lý Special Tokens cho Llama 3
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer

def get_model_and_tokenizer(model_name, cache_model=False):
    if model_name in loaded:
        return loaded[model_name]
    
    hf_path = get_hf_path(model_name)
    tokenizer = get_tokenizer(model_name)

    print(f"Loading model: {hf_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        device_map="auto",   
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    if cache_model:
        loaded[model_name] = model, tokenizer
    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)

def llama_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1, cache_model=True):
    """ 
    Tính NLL (Negative Log Likelihood) cho chuỗi target.
    Logic đã được cập nhật để tương thích với Llama 3 và sửa lỗi Index Masking.
    """
    model_obj, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    # 1. Serialize dữ liệu
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_str = input_str + target_str
    
    # 2. Tokenize Full String
    batch = tokenizer(
        [full_str], 
        return_tensors="pt",
        add_special_tokens=True 
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    # 3. Tính độ dài Token của Target
    target_tokens_len = len(tokenizer(target_str, add_special_tokens=False)['input_ids'])

    # 4. Forward Pass
    with torch.no_grad():
        out = model_obj(**batch)

    # 5. Masking (ĐÃ SỬA LỖI INDEX)
    logits = out.logits
    vocab_size = logits.shape[-1]
    
    # Danh sách các ký tự cho phép
    tokens_to_allow = list("0123456789" + settings.time_sep) + ['-', '.', ' ', '\n']
    
    good_tokens_ids = []
    for token in tokens_to_allow:
        tid = tokenizer.convert_tokens_to_ids(token)
        # Fix: Đảm bảo tid luôn là int, nếu là list thì lấy phần tử đầu
        if isinstance(tid, int):
            good_tokens_ids.append(tid)
        elif isinstance(tid, list) and len(tid) > 0:
            good_tokens_ids.append(tid[0])
            
    # Loại bỏ trùng lặp và convert sang Tensor để tránh lỗi index
    good_tokens_ids = list(set(good_tokens_ids))
    good_tokens_tensor = torch.tensor(good_tokens_ids, dtype=torch.long, device=logits.device)
    
    # Tạo mask: True là bad token
    bad_token_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
    # Dùng Tensor để index, đảm bảo an toàn tuyệt đối
    bad_token_mask[good_tokens_tensor] = False
    
    # Gán logit của bad token thành -inf
    logits[:, :, bad_token_mask] = -float('inf')

    # 6. Tính Logprobs
    input_ids = batch['input_ids'][0][1:] 
    shifted_logits = logits[0][:-1]
    
    log_softmax = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    
    token_logprobs = log_softmax[torch.arange(len(input_ids)), input_ids]
    token_logprobs = token_logprobs.cpu().numpy()

    # 7. Slicing
    if target_tokens_len > 0:
        slice_len = min(len(token_logprobs), target_tokens_len)
        target_logprobs = token_logprobs[-slice_len:]
    else:
        target_logprobs = np.array([])
    
    # 8. Tính kết quả
    if len(target_logprobs) > 0:
        BPD = -target_logprobs.sum() / len(target_arr)
    else:
        BPD = 0 

    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    
    return transformed_nll - avg_logdet_dydx

def llama_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9, 
    top_p=0.9,
    cache_model=True
):
    model_obj, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    # Ước lượng số token cần sinh
    input_tokens = tokenize_fn(input_str, model)['input_ids']
    avg_tokens_per_step = len(input_tokens) / len(input_str.split(settings.time_sep))
    max_new_tokens = int(avg_tokens_per_step * steps) + 20
    
    gen_strs = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches), desc="Generating", leave=False):
        batch_inputs = tokenizer([input_str], return_tensors="pt").to(model_obj.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        terminators = [t for t in terminators if isinstance(t, int)]

        with torch.no_grad():
            outputs = model_obj.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
                num_return_sequences=min(batch_size, num_samples - len(gen_strs))
            )
        
        input_length = batch_inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        decoded_batch = tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gen_strs.extend(decoded_batch)
        
        if len(gen_strs) >= num_samples:
            break
            
    return gen_strs[:num_samples]