import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import os

def measure_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return 0

def get_weight_stats(model):
    """Collects weight distribution stats for technical depth."""
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights) if weights else np.array([])

def compute_accuracy(model, tokenizer, dataset_text, k=1):
    """Computes Top-k accuracy on a validation subset."""
    model.eval()
    encodings = tokenizer(dataset_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids.to(model.device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        logits = outputs.logits
    
    # Shift so that tokens predict the next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    _, top_k_indices = shift_logits.topk(k, dim=-1)
    correct = (top_k_indices == shift_labels.unsqueeze(-1)).any(dim=-1)
    accuracy = correct.float().mean().item()
    return accuracy

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
        if precision in ["INT8", "INT4"]:
            print(f"Skipping {precision} on MPS: bitsandbytes requires CUDA.")
            return None
    
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device == "mps":
            model = model.to("mps")
    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    load_time = time.time() - start_time
    weight_data = get_weight_stats(model)
    
    # Inference Latency (Throughput focus)
    input_text = "Analysis of neural network quantization shows"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    latencies = []
    for _ in range(10): # More iterations for stability
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        if device == "cuda": torch.cuda.synchronize()
        elif device == "mps": torch.mps.synchronize()
        latencies.append((time.time() - start_time) / 32) # seconds per token
        
    avg_latency = np.mean(latencies) * 1000 # convert to ms/token
    
    # Accuracy & Perplexity
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(dataset["text"][:50])
    
    top1_acc = compute_accuracy(model, tokenizer, test_text, k=1)
    top5_acc = compute_accuracy(model, tokenizer, test_text, k=5)
    
    # Perplexity (Fuller slice)
    encodings = tokenizer("\n\n".join(dataset["text"][:100]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    stride = 512
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + 512, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nlls.append(outputs.loss)
    ppl = torch.exp(torch.stack(nlls).mean()).item()

    res = {
        "precision": precision,
        "latency_ms_per_token": avg_latency,
        "memory_mb": measure_memory(),
        "perplexity": ppl,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "weight_data": weight_data
    }
    
    del model
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()
    return res

def plot_complex_results(results):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'weight_data'} for r in results])
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)

    # 1. Latency vs Accuracy Trade-off
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x="latency_ms_per_token", y="top1_accuracy", hue="precision", s=200, ax=ax1)
    ax1.set_title("Inference Efficiency vs. Top-1 Accuracy", fontweight='bold')
    
    # 2. Memory Footprint
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df, x="precision", y="memory_mb", ax=ax2)
    ax2.set_title("Peak VRAM Usage (MB)", fontweight='bold')

    # 3. Perplexity Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df, x="precision", y="perplexity", ax=ax3)
    ax3.set_title("Language Modeling Perplexity (Lower is Better)", fontweight='bold')

    # 4. Top-5 Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=df, x="precision", y="top5_accuracy", ax=ax4)
    ax4.set_title("Top-5 Token Prediction Accuracy", fontweight='bold')

    # 5. Weight Distribution (Technical Depth)
    ax5 = fig.add_subplot(gs[2, :])
    for r in results:
        sns.kdeplot(r["weight_data"], label=r["precision"], ax=ax5, bw_adjust=0.5)
    ax5.set_title("Model Weight Distribution (Global Tensors)", fontweight='bold')
    ax5.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    print("Sophisticated plots saved as benchmark_results.png")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    precisions = ["FP32"]
    if device == "cuda": precisions += ["INT8", "INT4"]
    elif device == "mps": precisions += ["FP16"]
    
    all_res = [res for p in precisions if (res := benchmark_model("facebook/opt-125m", p, device))]
    if all_res: plot_complex_results(all_res)
