# Contributing to DISCO

## Development Setup

1. Clone and install:
   ```bash
   git clone https://github.com/kenza-ily/disco.git
   cd disco
   uv sync
   ```

2. Configure credentials:
   ```bash
   cp .env.example .env.local
   # Add your API keys
   ```




## Adding a New Benchmark

1. Create `benchmarks/dataset_specific/benchmark_yourdataset.py`
2. Follow the pattern in existing benchmarks
3. Save results to `results/1_raw/{dataset}/{phase}/`
4. Add entry to `scripts/run_benchmark.py` BENCHMARK_MODULES

## Running Benchmarks

Use the unified runner:
```bash
uv run python scripts/run_benchmark.py \
    --dataset <name> \
    --phases <phase1> <phase2> \
    --sample-limit <n> \
    --models <model1> <model2>
```

## Code Style

- Use type hints
- Follow PEP 8 (enforced by ruff)
- Add docstrings to public functions
