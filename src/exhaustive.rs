use std::time::Instant;

use crate::{play_iterated, TernaryChoice, TernaryStrategy, BenchmarkResult};

/// Benchmark: time how long it takes to enumerate all 3^n strategies.
pub struct ExhaustiveBenchmark {
    /// Number of response slots (4 means 3^4 = 81 strategies).
    pub n_slots: u32,
}

impl ExhaustiveBenchmark {
    /// Create a new exhaustive benchmark with the given number of slots.
    pub fn new(n_slots: u32) -> Self {
        Self { n_slots }
    }

    /// Default: 4 slots (81 strategies).
    pub fn default() -> Self {
        Self { n_slots: 4 }
    }

    /// Count total strategies: 3^n_slots.
    pub fn total_strategies(&self) -> u64 {
        3u64.pow(self.n_slots)
    }

    /// Enumerate all strategies as encoded u32 values.
    pub fn enumerate(&self) -> Vec<u32> {
        let total = self.total_strategies();
        let mut result = Vec::with_capacity(total as usize);
        for i in 0..total {
            result.push(i as u32);
        }
        result
    }

    /// Enumerate, decode, and collect all strategies.
    pub fn enumerate_decoded(&self) -> Vec<TernaryStrategy> {
        self.enumerate()
            .iter()
            .map(|&v| TernaryStrategy::decode(v))
            .collect()
    }

    /// Run the benchmark: enumerate all strategies `iterations` times.
    pub fn run(&self, iterations: u64) -> BenchmarkResult {
        let start = Instant::now();
        for _ in 0..iterations {
            let strategies = self.enumerate_decoded();
            std::hint::black_box(strategies);
        }
        let elapsed = start.elapsed().as_micros() as u64;
        BenchmarkResult::new(
            format!("exhaustive_3^{}", self.n_slots),
            iterations,
            elapsed,
        )
    }

    /// Run exhaustive pairwise evaluation: pit every strategy against every other.
    pub fn run_pairwise(&self, rounds: usize) -> BenchmarkResult {
        let strategies = self.enumerate_decoded();
        let total = strategies.len();
        let start = Instant::now();
        let mut matchups = 0u64;
        for i in 0..total {
            for j in 0..total {
                let (s1, s2) = play_iterated(&strategies[i], &strategies[j], rounds);
                std::hint::black_box((s1, s2));
                matchups += 1;
            }
        }
        let elapsed = start.elapsed().as_micros() as u64;
        BenchmarkResult::new(
            format!("exhaustive_pairwise_3^{}_{}r", self.n_slots, rounds),
            matchups,
            elapsed,
        )
    }
}
