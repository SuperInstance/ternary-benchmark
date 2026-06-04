use std::time::Instant;

use crate::{play_iterated, TernaryChoice, TernaryStrategy};

/// Benchmark result with timing statistics.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Number of iterations completed.
    pub iterations: u64,
    /// Total wall-clock time in microseconds.
    pub total_us: u64,
    /// Mean time per iteration in microseconds.
    pub mean_us: f64,
    /// Throughput: iterations per second.
    pub throughput: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result from raw measurements.
    pub fn new(name: impl Into<String>, iterations: u64, total_us: u64) -> Self {
        let mean_us = if iterations > 0 {
            total_us as f64 / iterations as f64
        } else {
            0.0
        };
        let throughput = if total_us > 0 {
            iterations as f64 / (total_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        Self {
            name: name.into(),
            iterations,
            total_us,
            mean_us,
            throughput,
        }
    }

    /// Run a closure `iterations` times and measure the elapsed time.
    pub fn measure(name: impl Into<String>, iterations: u64, f: impl Fn()) -> Self {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed = start.elapsed().as_micros() as u64;
        Self::new(name, iterations, elapsed)
    }

    /// Format as a single-line summary.
    pub fn to_line(&self) -> String {
        format!(
            "{:<30} {:>10} iters  {:>12.2} µs/iter  {:>15.0} iter/s",
            self.name, self.iterations, self.mean_us, self.throughput
        )
    }
}

/// A suite that runs all standard benchmarks and collects results.
pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new empty suite.
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Add a result to the suite.
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Run a benchmark and add its result.
    pub fn bench(&mut self, name: impl Into<String>, iterations: u64, f: impl Fn()) {
        let result = BenchmarkResult::measure(name, iterations, f);
        self.results.push(result);
    }

    /// Run all standard benchmarks with the given configuration.
    pub fn run_all(iterations: u64, rounds: usize, population: usize, generations: usize) -> Self {
        let mut suite = Self::new();

        // 1. Exhaustive enumeration benchmark
        suite.bench("exhaustive_enumeration", iterations, || {
            let n = 8u32; // 3^8 = 6561 strategies
            let mut count = 0u64;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            let val = i | (j << 2) | (k << 4) | (l << 6);
                            let _ = TernaryStrategy::decode(val);
                            count += 1;
                        }
                    }
                }
            }
            std::hint::black_box(count);
        });

        // 2. Iterated play benchmark
        let s1 = TernaryStrategy::new([TernaryChoice::Cooperate; 4]);
        let s2 = TernaryStrategy::new([TernaryChoice::Defect; 4]);
        suite.bench("iterated_play", iterations, || {
            let _ = play_iterated(&s1, &s2, rounds);
        });

        // 3. Evolution benchmark
        let mut pop: Vec<f64> = (0..population).map(|i| i as f64 / population as f64).collect();
        suite.bench("evolution_fitness", iterations, || {
            let total: f64 = pop.iter().sum();
            let _new_pop: Vec<f64> = pop.iter().map(|&f| f / total).collect();
            std::hint::black_box(&pop);
        });

        // 4. Ecology benchmark (Lotka-Volterra step)
        suite.bench("ecology_lv_step", iterations, || {
            let dt = 0.01;
            let a = 1.0;
            let b = 0.1;
            let c = 0.075;
            let d = 1.5;
            let mut pops = [population as f64, population as f64 / 2.0, population as f64 / 4.0];
            for _ in 0..generations {
                let dx = pops[0] * (a - b * pops[1]);
                let dy = -pops[1] * (d - c * pops[0] - 0.05 * pops[2]);
                let dz = -pops[2] * (0.5 - 0.02 * pops[1]);
                pops[0] += dx * dt;
                pops[1] += dy * dt;
                pops[2] += dz * dt;
                // Clamp to prevent divergence
                for p in pops.iter_mut() {
                    if *p < 0.0 { *p = 0.0; }
                    if *p > 1e6 { *p = 1e6; }
                }
            }
            std::hint::black_box(pops);
        });

        // 5. Strategy encode/decode
        suite.bench("strategy_encode_decode", iterations, || {
            let s = TernaryStrategy::new([
                TernaryChoice::Cooperate,
                TernaryChoice::Defect,
                TernaryChoice::Withhold,
                TernaryChoice::Cooperate,
            ]);
            let encoded = s.encode();
            let decoded = TernaryStrategy::decode(encoded);
            std::hint::black_box(decoded);
        });

        // 6. Payoff matrix lookup
        suite.bench("payoff_matrix", iterations * 100, || {
            let choices = TernaryChoice::all();
            for &a in &choices {
                for &b in &choices {
                    std::hint::black_box(crate::ternary_payoff(a, b));
                }
            }
        });

        suite
    }

    /// Get all results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Find the fastest benchmark by mean time.
    pub fn fastest(&self) -> Option<&BenchmarkResult> {
        self.results.iter().min_by(|a, b| a.mean_us.partial_cmp(&b.mean_us).unwrap())
    }

    /// Find the slowest benchmark by mean time.
    pub fn slowest(&self) -> Option<&BenchmarkResult> {
        self.results.iter().max_by(|a, b| a.mean_us.partial_cmp(&b.mean_us).unwrap())
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}
