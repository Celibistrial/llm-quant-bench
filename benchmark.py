import torch
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
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

def benchmark_model(model_name, precision, device, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
    print(f"\n--- Benchmarking {model_name} in {precision} on {device.upper()} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    kwargs = {}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        if precision == "FP32":
            kwargs["torch_dtype"] = torch.float32
        elif precision == "FP16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "BF16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "INT8":
            kwargs["load_in_8bit"] = True
        elif precision == "INT4":
            kwargs["load_in_4bit"] = True
        elif precision == "NF4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    elif device == "mps":
        kwargs["torch_dtype"] = torch.float32 if precision == "FP32" else torch.float16
        if precision in ["INT8", "INT4", "NF4", "BF16"]:
            print(f"Skipping {precision} on MPS: BitsAndBytes INT/NF formats and BF16 are not fully supported or optimized on MPS.")
            return None
    else: # cpu
        kwargs["torch_dtype"] = torch.float32
        if precision != "FP32":
            print(f"Skipping {precision} on CPU: Only FP32 is benchmarked on CPU in this script.")
            return None
            
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device == "mps":
            model = model.to("mps")
    except Exception as e:
        print(f"Failed to load {model_name} in {precision}: {e}")
        return None

    load_time = time.time() - start_time
    weight_data = get_weight_stats(model)
    
    # Inference Latency (Throughput focus)
    input_text = "Analysis of neural network quantization shows"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print("Warming up model generation...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        
    latencies = []
    for _ in tqdm(range(10), desc="Measuring Latency", unit="iter"):
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        if device == "cuda": torch.cuda.synchronize()
        elif device == "mps": torch.mps.synchronize()
        latencies.append((time.time() - start_time) / 32)
        
    avg_latency = np.mean(latencies) * 1000 # ms/token
    
    # Accuracy & Perplexity
    print(f"Evaluating Perplexity & Accuracy on {dataset_name} ({dataset_config})...")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="test")
        test_text = "\n\n".join(dataset["text"][:50])
        
        top1_acc = compute_accuracy(model, tokenizer, test_text, k=1)
        top5_acc = compute_accuracy(model, tokenizer, test_text, k=5)
        
        encodings = tokenizer("\n\n".join(dataset["text"][:100]), return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        nlls = []
        stride = 512
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL", unit="batch"):
            end_loc = min(begin_loc + 512, seq_len)
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                nlls.append(outputs.loss)
        ppl = torch.exp(torch.stack(nlls).mean()).item()
    except Exception as e:
        print(f"Dataset evaluation failed: {e}")
        top1_acc, top5_acc, ppl = 0, 0, 0

    memory_usage = measure_memory()
    print(f"Result -> Memory: {memory_usage:.2f} MB | Latency: {avg_latency:.2f} ms/token | PPL: {ppl:.2f}")

    res = {
        "model": model_name,
        "precision": precision,
        "latency_ms_per_token": avg_latency,
        "memory_mb": memory_usage,
        "perplexity": ppl,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "load_time_s": load_time,
        "weight_data": weight_data
    }
    
    del model
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()
    return res

def plot_complex_results(results, output_file='benchmark_results.png'):
    if not results:
        print("No results to plot.")
        return
        
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'weight_data'} for r in results])
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2)

    # 1. Latency vs Accuracy Trade-off
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=df, x="latency_ms_per_token", y="top1_accuracy", hue="precision", style="model", s=200, ax=ax1)
    ax1.set_title("Inference Efficiency vs. Top-1 Accuracy", fontweight='bold')
    ax1.set_xlabel("Latency (ms/token)")
    ax1.set_ylabel("Top-1 Accuracy")
    
    # 2. Memory Footprint
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df, x="precision", y="memory_mb", hue="model", ax=ax2)
    ax2.set_title("Peak VRAM Usage (MB)", fontweight='bold')
    ax2.set_ylabel("Memory (MB)")

    # 3. Perplexity Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df, x="precision", y="perplexity", hue="model", ax=ax3)
    ax3.set_title("Language Modeling Perplexity (Lower is Better)", fontweight='bold')
    ax3.set_ylabel("Perplexity")

    # 4. Top-5 Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=df, x="precision", y="top5_accuracy", hue="model", ax=ax4)
    ax4.set_title("Top-5 Token Prediction Accuracy", fontweight='bold')
    ax4.set_ylabel("Top-5 Accuracy")

    # 5. Weight Distribution (Technical Depth)
    ax5 = fig.add_subplot(gs[2, :])
    for r in results:
        label = f'{r["model"]} - {r["precision"]}'
        if len(r["weight_data"]) > 0:
            sns.kdeplot(r["weight_data"], label=label, ax=ax5, bw_adjust=0.5)
    ax5.set_title("Model Weight Distribution (Global Tensors)", fontweight='bold')
    ax5.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Sophisticated plots saved as {output_file}")

def save_results_json(results, output_file='benchmark_results.json'):
    if not results:
        return
    data = [{k: v for k, v in r.items() if k != 'weight_data'} for r in results]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Raw metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced LLM Quantization Benchmark Suite")
    parser.add_argument("--models", nargs="+", default=["facebook/opt-125m"], help="HuggingFace model IDs to benchmark")
    parser.add_argument("--precisions", nargs="+", default=["FP32", "FP16", "BF16", "INT8", "INT4", "NF4"], help="Precisions to test")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device to force")
    parser.add_argument("--dataset", type=str, default="wikitext", help="HuggingFace dataset name for accuracy/PPL")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration string")
    parser.add_argument("--output-img", type=str, default="benchmark_results.png", help="Output plot filename")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Output JSON filename")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
        
    print("="*60)
    print("LLM QUANTIZATION BENCHMARK SUITE")
    print("="*60)
    print(f"Device:      {device.upper()}")
    print(f"Models:      {', '.join(args.models)}")
    print(f"Precisions:  {', '.join(args.precisions)}")
    print(f"Dataset:     {args.dataset} ({args.dataset_config})")
    print("="*60)
    
    all_res = []
    for model_id in args.models:
        for p in args.precisions:
            res = benchmark_model(model_id, p, device, args.dataset, args.dataset_config)
            if res: 
                all_res.append(res)
                
    if all_res: 
        save_results_json(all_res, args.output_json)
        plot_complex_results(all_res, args.output_img)
    else:
        print("No successful benchmarks completed. Check hardware compatibility or model names.")
