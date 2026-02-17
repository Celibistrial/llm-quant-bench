import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def measure_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif torch.backends.mps.is_available():
        # MPS memory measurement is less direct, using current allocated
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0

def benchmark_model(model_name, precision, device):
    print(f"--- Benchmarking {model_name} in {precision} on {device} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    kwargs = {}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        if precision == "FP32":
            kwargs["torch_dtype"] = torch.float32
        elif precision == "INT8":
            kwargs["load_in_8bit"] = True
        elif precision == "INT4":
            kwargs["load_in_4bit"] = True
    elif device == "mps":
        kwargs["torch_dtype"] = torch.float32 if precision == "FP32" else torch.float16
        # bitsandbytes doesn't support MPS yet
        if precision in ["INT8", "INT4"]:
            print(f"Skipping {precision} on MPS: bitsandbytes requires CUDA.")
            return None
    else:
        kwargs["torch_dtype"] = torch.float32

    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device == "mps":
            model = model.to("mps")
    except Exception as e:
        print(f"Failed to load model in {precision}: {e}")
        return None

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
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        latencies.append(time.time() - start_time)
        
    avg_latency = np.mean(latencies)
    print(f"Average Inference Latency (20 tokens): {avg_latency:.4f} s")
    
    # Perplexity on WikiText-2 subset
    print("Calculating perplexity on WikiText-2 subset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"][:150]), return_tensors="pt")
    
    max_length = 2048
    if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings:
        max_length = model.config.max_position_embeddings
        
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

    ppl = torch.exp(torch.stack(nlls).mean()).item() if nlls else None
    if ppl:
        print(f"Perplexity: {ppl:.2f}")
    
    result = {
        "precision": precision,
        "load_time": load_time,
        "memory_mb": memory_mb,
        "latency": avg_latency,
        "perplexity": ppl
    }
    
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
        
    return result

def plot_results(results):
    df = pd.DataFrame(results)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LLM Quantization Benchmark Results (OPT-125M)', fontsize=16)

    sns.barplot(x='precision', y='latency', data=df, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Inference Latency (s) - Lower is Better')
    
    sns.barplot(x='precision', y='memory_mb', data=df, ax=axes[0, 1], palette='magma')
    axes[0, 1].set_title('Peak Memory Usage (MB) - Lower is Better')
    
    sns.barplot(x='precision', y='perplexity', data=df, ax=axes[1, 0], palette='rocket')
    axes[1, 0].set_title('Perplexity - Lower is Better')
    
    sns.barplot(x='precision', y='load_time', data=df, ax=axes[1, 1], palette='cubehelix')
    axes[1, 1].set_title('Model Load Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark_results.png')
    print("Plot saved as benchmark_results.png")

if __name__ == "__main__":
    model_id = "facebook/opt-125m"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    precisions = ["FP32"]
    if device == "cuda":
        precisions += ["INT8", "INT4"]
    elif device == "mps":
        precisions += ["FP16"] # MPS supports FP16 well
    
    results = []
    for p in precisions:
        res = benchmark_model(model_id, p, device)
        if res:
            results.append(res)
    
    if results:
        plot_results(results)
