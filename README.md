# LLM Quantization Benchmark

This project benchmarks the `facebook/opt-125m` model at different precision levels (FP32, FP16, INT8, INT4) to evaluate performance trade-offs in inference latency, memory footprint, and perplexity.

## Features
- **Precision Levels:** FP32, FP16, INT8, and INT4 (INT8/INT4 require CUDA).
- **Hardware Support:** 
  - **CUDA:** Full support for all precisions via `bitsandbytes`.
  - **MPS (Apple Silicon):** Optimized support for FP32 and FP16.
- **Metrics:** Inference latency (ms/token), Peak Memory (MB), and Perplexity (WikiText-2).
- **Environment:** Managed with `uv` for fast, reproducible setups.

## Requirements

- Python 3.10+
- `uv` (recommended) or `pip`
- A CUDA-compatible GPU (for INT8/INT4) or Apple Silicon (for MPS acceleration).

## Getting Started

### Using uv (Recommended)

```bash
# Install dependencies and run the benchmark
uv run python benchmark.py
```

### Using pip

```bash
pip install -r requirements.txt
python benchmark.py
```

## Results

The benchmark generates a summary plot `benchmark_results.png` comparing the metrics across precisions.

![Benchmark Results](benchmark_results.png)

*Note: Results depend on your hardware. On Apple Silicon, FP16 typically offers the best balance of speed and memory usage.*
