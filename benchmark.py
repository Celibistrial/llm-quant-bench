import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def measure_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0

def benchmark_model(model_name, precision):
    print(f"--- Benchmarking {model_name} in {precision} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    kwargs = {"device_map": "auto"}
    if precision == "FP32":
        kwargs["torch_dtype"] = torch.float32
    elif precision == "INT8":
        kwargs["load_in_8bit"] = True
    elif precision == "INT4":
        kwargs["load_in_4bit"] = True
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        print(f"Failed to load model in {precision}: {e}")
        return

    load_time = time.time() - start_time
    memory_mb = measure_memory()
    
    print(f"Model Load Time: {load_time:.2f} s")
    print(f"Peak Memory: {memory_mb:.2f} MB")
    
    # Inference Latency
    input_text = "The future of artificial intelligence is"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
        
    # Measure latency
    latencies = []
    for _ in range(5):
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.time() - start_time)
        
    avg_latency = np.mean(latencies)
    print(f"Average Inference Latency (20 tokens): {avg_latency:.4f} s")
    
    # Perplexity on WikiText-2 subset
    print("Calculating perplexity on WikiText-2 subset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Use a small subset for quick benchmarking
    encodings = tokenizer("\n\n".join(dataset["text"][:150]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    if not hasattr(model.config, 'max_position_embeddings') or max_length is None:
        max_length = 2048
        
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if nlls:
        ppl = torch.exp(torch.stack(nlls).mean())
        print(f"Perplexity (WikiText-2 subset): {ppl.item():.2f}")
    else:
        print("Perplexity calculation failed or sequence too short.")
        
    print("\n")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    model_id = "facebook/opt-125m"
    benchmark_model(model_id, "FP32")
    if torch.cuda.is_available():
        benchmark_model(model_id, "INT8")
        benchmark_model(model_id, "INT4")
    else:
        print("CUDA not available. Skipping INT8 and INT4 quantization benchmarks which require a GPU and bitsandbytes.")
