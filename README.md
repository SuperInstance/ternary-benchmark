# Ternary Benchmark

**Ternary Benchmark** is a comprehensive benchmark harness comparing FP32 vs ternary-packed computation kernels — measuring memory density, throughput, accuracy tradeoffs, and evolutionary dynamics for real GPU workloads.

## Why It Matters

Replacing 32-bit floating-point with 2-bit ternary values yields **16x memory density improvement** — a model that needs 8GB in FP32 fits in 512MB ternary. But does the computation actually get faster? This crate provides the definitive benchmark suite: matrix multiplication, attention heads, and evolutionary strategy tournaments measured in both representations with precise timing, throughput ratios, and accuracy quantification.

## How It Works

### FP32 vs Ternary Kernel Model

For N×N matrix multiplication:

```
FP32:     N² elements × 4 bytes  = 4N² bytes
Ternary:  N² elements × 2 bits   = N²/4 bytes

Density ratio: 4N² / (N²/4) = 16×

FP32 ops:   2N per element (multiply + add)
Ternary ops: N/16 per element (XNOR + popcount packs 16)

Theoretical speedup: (2N) / (N/16) = 32×
```

Simulated throughput: FP32 at ~10 TFLOPS, ternary at ~50 TOPS (XNOR-popcount is simpler silicon).

### Benchmark Suite

Each `KernelBench` produces a `BenchmarkResult`:

```
BenchmarkResult {
    fp32_bytes, ternary_bytes,        // memory
    fp32_ops, ternary_ops,            // operation counts
    fp32_time_us, ternary_time_us,    // wall clock
    accuracy_loss_pct,                // quality tradeoff
}
```

Derived metrics:
- `density_ratio()` = fp32_bytes / ternary_bytes
- `speedup()` = fp32_time / ternary_time
- `throughput_ratio()` = ternary_throughput / fp32_throughput

### Evolution Benchmark

The `EvolutionBenchmark` simulates strategy evolution:

```
population_size agents
each round: play_iterated(strategy_i, strategy_j) for all pairs
compute_fitness → select top 50% → crossover → next generation
```

Fitness computation: **O(N² · R)** for N agents, R rounds per match. Selection: **O(N log N)** (sort by fitness). Crossover: **O(N/2)** (fill population).

### Exhaustive Benchmark

Enumerate all 3^n strategies and play every pair:

```
total_strategies = 3^n
pairwise_matches = (3^n)²  = 3^(2n)
```

For n=4: 81 strategies, 6,561 matches. Cost: **O(3^(2n) · R)**.

### Ecology Benchmark

Lotka-Volterra three-species simulation:

```
dx/dt = x(α - βy)     (prey/cooperators)
dy/dt = -y(δ - γx - εz) (predator/defectors)
dz/dt = -z(ζ - 0.02y)  (omnivore/explorers)
```

Each Euler step: **O(1)** (12 FLOPs). N steps: **O(N)**.

## Quick Start

```rust
use ternary_benchmark::KernelBench;

let bench = KernelBench::matmul(1024); // 1024×1024
let result = bench.run();

println!("Density: {:.1}×", result.density_ratio());
println!("Speedup: {:.1}×", result.speedup());
println!("Accuracy loss: {:.1}%", result.accuracy_loss_pct);
```

## API

| Module | Key Types |
|--------|-----------|
| `benchmark` | `BenchmarkResult`, `BenchmarkSuite`, `BenchmarkResult::measure()` |
| `ecology` | `EcologyBenchmark`, `LvParams` |
| `evolution` | `EvolutionBenchmark` with fitness, selection, crossover |
| `exhaustive` | `ExhaustiveBenchmark` — enumerate all 3^n strategies |
| `report` | Formatted output generation |
| `scaling` | Scalability benchmarks |

## Architecture Notes

Ternary Benchmark provides the performance validation for the ternary computation stack in SuperInstance. In γ + η = C, ternary packing maximizes γ (growth — more computation per watt/dollar) while the accuracy_loss_pct measures η (avoidance — quality degradation from quantization). The 16× density ratio directly implements the conservation principle: fewer bits for equivalent computation.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for GPU computation architecture.

## References

1. Courbariaux, M. et al. (2016). "Binarized Neural Networks." *NeurIPS*.
2. Rastegari, M. et al. (2016). "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks." *ECCV*.
3. Hubara, I. et al. (2017). "Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations." *JMLR*, 18, 1–30.

## License

Apache-2.0
