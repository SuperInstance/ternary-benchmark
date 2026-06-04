# ternary-benchmark

Standardized benchmarks for ternary agent systems — reproducible performance numbers.

## Overview

This crate provides a benchmarking framework for ternary strategy systems, where agents choose between three actions (**Cooperate**, **Defect**, **Withhold**) in iterated games. It measures performance of core operations like strategy enumeration, evolutionary dynamics, and ecological simulations.

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **ExhaustiveBenchmark** | Time to enumerate all 3^n strategies and evaluate pairwise matchups |
| **EvolutionBenchmark** | Time per generation of an evolving population (fitness → selection → crossover) |
| **EcologyBenchmark** | Time per step of a three-species Lotka-Volterra simulation (Euler & RK4) |
| **ScalingBenchmark** | Throughput at scales [100, 1000, 10000, 100000] for pairwise, ecology, and enumeration |
| **BenchmarkSuite** | Run all standard benchmarks in one call |

## Methodology

### Timing
- All timing uses `std::time::Instant` (wall-clock, nanosecond resolution)
- No external dependencies — pure `std`
- Results report: iterations, total time, mean µs/iter, throughput (iter/s)

### Ternary Strategy Model
- A strategy maps opponent's last move → next move (4 entries: response to C/D/W + opening move)
- 3^4 = 81 unique strategies (with 4 response slots)
- Payoff matrix extends Prisoner's Dilemma with a neutral "Withhold" option

### Payoff Matrix
| | **Cooperate** | **Defect** | **Withhold** |
|---|---|---|---|
| **Cooperate** | (3, 3) | (0, 5) | (1, 1) |
| **Defect** | (5, 0) | (1, 1) | (2, 0) |
| **Withhold** | (1, 1) | (0, 2) | (1, 1) |

### Lotka-Volterra Ecology
- Three species: prey (cooperators), predator (defectors), omnivore (withholders)
- Euler and RK4 integration available
- Clamped to [0, 10^6] to prevent numerical divergence

### Reproducibility
- All benchmarks use deterministic inputs (no randomness)
- Population initialization uses fixed seed derivation: `strategy[i] = decode((i * 7919) as u32)`
- Results are machine-dependent (CPU speed) but algorithmically consistent

## Quick Start

```rust
use ternary_benchmark::*;

// Run the full suite
let suite = BenchmarkSuite::run_all(100, 100, 50, 20);
for result in suite.results() {
    println!("{}", result.to_line());
}

// Run scaling benchmarks
let scaling = ScalingBenchmark::run_all(100);
let report = BenchmarkReport::new(scaling);
print!("{}", report);
```

## Running Benchmarks

```bash
# Run all tests
cargo test

# Run a quick benchmark (tests include timing assertions)
cargo test -- --nocapture
```

## Architecture

```
src/
├── lib.rs          # Core types: TernaryChoice, TernaryStrategy, payoff functions
├── benchmark.rs    # BenchmarkResult, BenchmarkSuite
├── exhaustive.rs   # ExhaustiveBenchmark
├── evolution.rs    # EvolutionBenchmark
├── ecology.rs      # EcologyBenchmark (Lotka-Volterra)
├── scaling.rs      # ScalingBenchmark
├── report.rs       # BenchmarkReport formatting
└── tests.rs        # 24 tests
```

## License

MIT
