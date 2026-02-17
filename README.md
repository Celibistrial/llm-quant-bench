# LLM Quantization Benchmark

This is a small benchmarking project where I evaluate the `facebook/opt-125m` model at different precision levels:
- FP32
- INT8
- INT4

The project uses `bitsandbytes` and the HuggingFace `transformers` library to load the model in reduced precision and measures:
1. **Inference Latency:** Time taken to generate new tokens.
2. **Memory Footprint:** Peak memory allocated on the GPU during model loading and inference.
3. **Perplexity:** Language modeling performance on a subset of the WikiText-2 dataset.

## Requirements

This project requires a CUDA-compatible GPU to run the INT8 and INT4 benchmarks (which rely on `bitsandbytes`).

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Benchmark

Execute the benchmarking script:

```bash
python benchmark.py
```

## Results

*Results will vary depending on the specific GPU and environment used.*
The script outputs metrics for model load time, peak memory usage in MB, average inference latency in seconds, and perplexity scores.
