use std::time::Instant;

use crate::{play_iterated, TernaryChoice, TernaryStrategy, BenchmarkResult};

/// Benchmark: run at multiple scales and report throughput.
pub struct ScalingBenchmark;

impl ScalingBenchmark {
    /// Standard scales to test.
    pub const SCALES: &'static [usize] = &[100, 1_000, 10_000, 100_000];

    /// Run iterated play at different population scales.
    /// At each scale, create N strategies and do pairwise evaluation.
    pub fn run_pairwise_scaling(rounds: usize) -> Vec<BenchmarkResult> {
        Self::SCALES
            .iter()
            .map(|&scale| {
                let strategies: Vec<TernaryStrategy> = (0..scale)
                    .map(|i| TernaryStrategy::decode((i * 7919) as u32))
                    .collect();

                let start = Instant::now();
                let mut matchups = 0u64;
                // Sample pairwise: each strategy vs the next one (wrap around)
                for i in 0..scale {
                    let j = (i + 1) % scale;
                    let _ = play_iterated(&strategies[i], &strategies[j], rounds);
                    matchups += 1;
                }
                let elapsed = start.elapsed().as_micros() as u64;
                BenchmarkResult::new(
                    format!("scaling_pairwise_{}", scale),
                    matchups,
                    elapsed,
                )
            })
            .collect()
    }

    /// Run Lotka-Volterra simulation at different step counts.
    pub fn run_ecology_scaling() -> Vec<BenchmarkResult> {
        Self::SCALES
            .iter()
            .map(|&steps| {
                let mut eco = crate::EcologyBenchmark::new(100.0, 50.0, 25.0);
                let start = Instant::now();
                eco.simulate(steps);
                let elapsed = start.elapsed().as_micros() as u64;
                BenchmarkResult::new(
                    format!("scaling_ecology_{}steps", steps),
                    steps as u64,
                    elapsed,
                )
            })
            .collect()
    }

    /// Run strategy enumeration at increasing slot counts.
    pub fn run_enumeration_scaling() -> Vec<BenchmarkResult> {
        [2u32, 3, 4, 5, 6]
            .iter()
            .map(|&n_slots| {
                let bench = crate::ExhaustiveBenchmark::new(n_slots);
                let start = Instant::now();
                let strategies = bench.enumerate_decoded();
                let elapsed = start.elapsed().as_micros() as u64;
                BenchmarkResult::new(
                    format!("scaling_enumerate_3^{}", n_slots),
                    strategies.len() as u64,
                    elapsed,
                )
            })
            .collect()
    }

    /// Run all scaling benchmarks.
    pub fn run_all(rounds: usize) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        results.extend(Self::run_pairwise_scaling(rounds));
        results.extend(Self::run_ecology_scaling());
        results.extend(Self::run_enumeration_scaling());
        results
    }
}
