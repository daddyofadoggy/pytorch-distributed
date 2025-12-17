# PyTorch Distributed Training

## Initial Setup

This project uses `uv` for Python package management.

### Install uv

First, install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup Project

```bash
# Install dependencies
uv sync

# Run Python scripts
uv run python main.py

# Run with environment variables
ENV_VARIABLE=value uv run python script.py
```

## Assignments

- [Assignment 0: Single GPU Training & Profiling](assignments/assignment0/README.md)
- [Assignment 1: Data Parallel Training (DDP/FSDP)](assignments/assignment1/README.md)

## Output Log 
### Experiment on a single GPU
- CUDA_VISIBLE_DEVICES=0 uv run python memory_analysis.py
```python
Model: GPT-2 Small
Batch size: 8
Sequence length: 1024
Device: CUDA
Model parameters: 124,439,808
Memory snapshot saved: outputs/task1_memory_snapshot.pickle
View at: https://pytorch.org/memory_viz

--- Memory Breakdown ---
Estimated Memory:
  Parameters:       474.70 MB
  Gradients:        474.70 MB
  Optimizer States: 949.40 MB
  Total Estimated:  1898.80 MB

Actual Memory:
  Peak Allocated:   3076.27 MB
  Reserved:         15890.00 MB

Difference:
  Allocated vs Estimated: 1177.47 MB
```
- CUDA_VISIBLE_DEVICES=0 uv run python throughput.py
```python
Task 2: Throughput Measurement & Scaling
==================================================
Device: cuda
Model: GPT-2 Small (124,439,808 parameters)
Batch size: 8
Sequence length: 1024
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 163,840
Time elapsed: 5.03 seconds
Tokens per second: 32582.4
Steps per second: 3.98

--- Modern LLM Training Extrapolation ---
GPT-2 Small: 32,582 tokens/sec
1T Model throughput: 4.055 tokens/sec

Training 10T tokens:
  Time: 28,545,911 days (78208.0 years)

--- Batch Size Analysis ---

Testing batch_size=1...
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 20,480
Time elapsed: 1.24 seconds
Tokens per second: 16520.7
Steps per second: 16.13

Testing batch_size=4...
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 81,920
Time elapsed: 2.66 seconds
Tokens per second: 30818.8
Steps per second: 7.52

Testing batch_size=8...
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 163,840
Time elapsed: 5.03 seconds
Tokens per second: 32584.9
Steps per second: 3.98

Testing batch_size=16...
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 327,680
Time elapsed: 9.67 seconds
Tokens per second: 33900.1
Steps per second: 2.07

Testing batch_size=32...
Warming up...
Measuring throughput over 20 steps...

--- Throughput Results ---
Total tokens processed: 655,360
Time elapsed: 18.96 seconds
Tokens per second: 34559.6
Steps per second: 1.05

Testing batch_size=64...
Warming up...
  OOM at batch_size=64

--- Batch Size Results ---
Batch Size   Tokens/sec   Memory (MB) 
----------------------------------------
1            16521        15619       
4            30819        15619       
8            32585        16104       
16           33900        30224       
32           34560        58438  
```

### Experiment on DDP
- CUDA_VISIBLE_DEVICES=0 uv run python train_baseline.py
```python
Tokens in shard: 100,000,000
step=0 | loss=11.0429 | lr=2.98e-04 | time=5.9s
step=10 | loss=7.9564 | lr=1.44e-04 | time=62.6s
Training completed in 112.3s
Baseline training completed!
```
- uv run torchrun --nproc_per_node=2 train_ddp.py
```python
Tokens in shard: 100,000,000
step=0 | loss=11.0901 | lr=2.98e-04 | time=3.2s
step=10 | loss=7.9705 | lr=1.44e-04 | time=33.1s
Training completed in 58.3s
DDP training completed!
```
### Experiment on FSDP
- uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy FULL_SHARD
```python
Tokens in shard: 100,000,000
step=0 | loss=11.0901 | lr=2.98e-04 | time=3.9s
step=10 | loss=7.9705 | lr=1.44e-04 | time=33.8s
Training completed in 58.8s
FSDP training with FULL_SHARD completed!
Traces saved to: outputs/traces/fsdp_full_shard/
```
- uv run torchrun --nproc_per_node=2 train_fsdp.py --strategy SHARD_GRAD_OP
```python
Tokens in shard: 100,000,000
step=0 | loss=11.0901 | lr=2.98e-04 | time=3.9s
step=10 | loss=7.9705 | lr=1.44e-04 | time=33.6s
Training completed in 58.6s
FSDP training with SHARD_GRAD_OP completed!
Traces saved to: outputs/traces/fsdp_shard_grad_op/
```