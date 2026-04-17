# LLM Quantization Benchmark Suite

An advanced, highly configurable Command Line Interface (CLI) suite for evaluating Large Language Model (LLM) quantization strategies. This project goes beyond basic latency checks to analyze the structural and probabilistic impact of precision reduction on model weights and accuracy.

## Technical Analysis Features
- **Extensive Precision Support**: Evaluate models in `FP32`, `FP16`, `BF16` (Bfloat16), `INT8`, `INT4`, and `NF4` (NormalFloat4).
- **Rigorous Accuracy Metrics**: Evaluates **Top-1** and **Top-5 Accuracy** on a validation slice of the WikiText dataset.
- **Probabilistic Scoring**: Measures **Perplexity (PPL)** to quantify information loss across precisions.
- **Weight Distribution Analysis**: Utilizes **Kernel Density Estimation (KDE)** to visualize how quantization shifts global model tensors.
- **Data Export**: Saves raw benchmark metrics to a `.json` file for downstream programmatic analysis.
- **Multi-Hardware Optimization**:
  - **CUDA**: Full support for `bitsandbytes` (INT8/INT4/NF4).
  - **MPS (Apple Silicon)**: Native support for FP32 and FP16.

## Requirements
- Python 3.10+
- `uv` (recommended)
- Hardware: CUDA GPU (for INT8/INT4/NF4) or Apple Silicon (for MPS).

## Quickstart

```bash
uv run python benchmark.py
```

## CLI Usage

The benchmark is fully configurable via CLI arguments:

```bash
uv run python benchmark.py --help
```

### Options:
- `--models`: List of HuggingFace model IDs to benchmark (e.g., `facebook/opt-125m facebook/opt-350m`). Default: `facebook/opt-125m`.
- `--precisions`: List of precisions to test. Default tests all supported precisions for your device.
- `--device`: Force a specific compute device (`auto`, `cuda`, `mps`, `cpu`). Default: `auto`.
- `--dataset`: HuggingFace dataset name for accuracy/PPL. Default: `wikitext`.
- `--dataset-config`: Dataset configuration string. Default: `wikitext-2-raw-v1`.
- `--output-img`: Filename for the generated plot. Default: `benchmark_results.png`.
- `--output-json`: Filename for the exported JSON metrics. Default: `benchmark_results.json`.

### Example: Multi-Model Sweep
```bash
uv run python benchmark.py --models facebook/opt-125m facebook/opt-350m --precisions FP32 FP16 INT8 --output-img comparison.png
```

## Comparative Analysis

The script automatically generates a multi-dimensional visualization (`benchmark_results.png`):

![Benchmark Analysis](benchmark_results.png)

*Key Insights:*
- **Efficiency Frontier**: Visualizes the Pareto frontier between inference speed and Top-1 accuracy.
- **Memory Compression**: Quantifies the VRAM savings provided by aggressive quantization (NF4, INT4).
- **Weight Drift**: The KDE plot reveals how quantization alters the numerical distribution of the model's intelligence.

---
*Created as part of a deep-dive into LLM optimization and deployment efficiency.*