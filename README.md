# Sensitivity-Aware Hybrid Quantization (SAHQ) Research

This project evaluates the non-uniform impact of quantization across Transformer decoder layers. It introduces Sensitivity-Aware Hybrid Quantization (SAHQ), a method that identifies and preserves critical layers to minimize perplexity drift during 4-bit model compression.

## Methodology

The evaluation is divided into two phases: a baseline assessment of established quantization formats and a layer-wise sensitivity analysis for hybrid optimization.

### Baseline Evaluation (Public)
Established quantization precisions are benchmarked for inference latency (ms/token), memory footprint (MB), and perplexity (PPL) using the WikiText-2 dataset:
- FP32 and FP16: High-precision baselines.
- INT8 and INT4: Standard integer quantization.
- NF4 (NormalFloat4): 4-bit quantization via the QLoRA framework.

### Sensitivity-Aware Hybrid Quantization (Novel)
SAHQ identifies decoder layers with the highest structural sensitivity. The process involves:
1. Layer-Wise Noise Injection: Quantization noise is systematically injected into individual decoder layers.
2. Global Perplexity Measurement: The resulting drift in global model perplexity is measured for each layer.
3. Critical Layer Identification: Layers exceeding the 85th percentile of sensitivity are classified as critical.
4. Hybrid Allocation Strategy: A hybrid configuration is proposed where critical layers remain in FP16 while non-critical layers are compressed to 4-bit (NF4).

## Benchmark Results

The automated pipeline generates a multi-dimensional research report (`research_report.png`) detailing the performance frontier and the layer sensitivity map.

![Quantization Benchmark Results](research_report.png)

### Key Metrics and Observations
- Efficiency Frontier: Illustrates the trade-off between inference latency and perplexity.
- Sensitivity Heatmap: Visualizes the non-linear distribution of sensitivity across decoder layers. Layers 5 and 7 in the OPT-125M architecture consistently exhibit the highest vulnerability to quantization noise.
- Pareto Optimization: SAHQ provides a superior balance between memory reduction and language modeling accuracy compared to uniform INT4 quantization.

## Technical Implementation

The suite is built for reproducible research using the following stack:
- Framework: PyTorch with HuggingFace Transformers and Accelerate.
- Quantization: bitsandbytes (NF4/INT8/INT4).
- Environment: Managed via uv.

### Execution
Run the complete benchmarking and research pipeline:
```bash
uv run python benchmark.py
```

---
This research demonstrates that non-uniform bit allocation based on layer sensitivity is a viable path for optimizing large language model compression.
