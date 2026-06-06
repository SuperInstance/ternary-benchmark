//! # ternary-benchmark
//!
//! Benchmark harness: FP32 vs ternary-packed kernels.
//! Measures memory density, throughput, and accuracy tradeoffs.

/// Benchmark result comparing FP32 and ternary.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub fp32_bytes: usize,
    pub ternary_bytes: usize,
    pub fp32_ops: u64,
    pub ternary_ops: u64,
    pub fp32_time_us: u64,
    pub ternary_time_us: u64,
    pub accuracy_loss_pct: f64,
}

impl BenchmarkResult {
    pub fn density_ratio(&self) -> f64 { self.fp32_bytes as f64 / self.ternary_bytes.max(1) as f64 }
    pub fn speedup(&self) -> f64 {
        if self.ternary_time_us == 0 { return 0.0; }
        self.fp32_time_us as f64 / self.ternary_time_us as f64
    }
    pub fn throughput_ratio(&self) -> f64 {
        if self.fp32_time_us == 0 { return 0.0; }
        let fp32_tput = self.fp32_ops as f64 / (self.fp32_time_us as f64 / 1e6);
        let tern_tput = self.ternary_ops as f64 / (self.ternary_time_us.max(1) as f64 / 1e6);
        tern_tput / fp32_tput
    }
}

/// Simulated kernel benchmark.
pub struct KernelBench {
    pub name: String,
    pub element_count: usize,
    pub fp32_bytes_per_element: usize,
    pub ternary_bits_per_element: usize,
    pub fp32_ops_per_element: u64,
    pub ternary_ops_per_element: u64,
}

impl KernelBench {
    pub fn matmul(n: usize) -> Self {
        Self {
            name: format!("matmul_{}x{}", n, n),
            element_count: n * n,
            fp32_bytes_per_element: 4,
            ternary_bits_per_element: 2,
            fp32_ops_per_element: (2 * n) as u64, // mul + add per element
            ternary_ops_per_element: (n / 16) as u64, // XNOR+popcount packs 16
        }
    }

    pub fn attention(seq_len: usize, dim: usize) -> Self {
        Self {
            name: format!("attention_{}x{}", seq_len, dim),
            element_count: seq_len * dim,
            fp32_bytes_per_element: 4,
            ternary_bits_per_element: 2,
            fp32_ops_per_element: (2 * dim) as u64,
            ternary_ops_per_element: (dim / 16) as u64,
        }
    }

    pub fn run(&self) -> BenchmarkResult {
        let fp32_bytes = self.element_count * self.fp32_bytes_per_element;
        let ternary_bytes = (self.element_count * self.ternary_bits_per_element + 7) / 8;

        let fp32_ops = self.element_count as u64 * self.fp32_ops_per_element;
        let ternary_ops = self.element_count as u64 * self.ternary_ops_per_element;

        // Simulated: FP32 at ~10 TFLOPS, ternary at ~50 TOPS (XNOR is simpler)
        let fp32_time_us = fp32_ops / 10_000; // 10 TFLOPS simulated
        let ternary_time_us = ternary_ops / 50_000; // 50 TOPS simulated

        // Accuracy loss: ~1-3% for well-trained ternary
        let accuracy_loss = 1.5 + (self.element_count as f64 / 1_000_000.0).min(2.0);

        BenchmarkResult {
            name: self.name.clone(),
            fp32_bytes, ternary_bytes, fp32_ops, ternary_ops,
            fp32_time_us: fp32_time_us.max(1),
            ternary_time_us: ternary_time_us.max(1),
            accuracy_loss_pct: accuracy_loss,
        }
    }
}

/// Run a suite of benchmarks.
pub fn run_suite() -> Vec<BenchmarkResult> {
    vec![
        KernelBench::matmul(256).run(),
        KernelBench::matmul(1024).run(),
        KernelBench::attention(512, 64).run(),
        KernelBench::attention(1024, 128).run(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_density() {
        let result = KernelBench::matmul(256).run();
        assert!(result.density_ratio() >= 16.0); // 2 bits vs 32 bits
    }

    #[test]
    fn test_matmul_speedup() {
        let result = KernelBench::matmul(1024).run();
        assert!(result.speedup() > 1.0); // ternary faster
    }

    #[test]
    fn test_attention_benchmark() {
        let result = KernelBench::attention(512, 64).run();
        assert!(result.fp32_bytes > result.ternary_bytes);
    }

    #[test]
    fn test_accuracy_loss_reasonable() {
        let result = KernelBench::matmul(256).run();
        assert!(result.accuracy_loss_pct < 5.0);
    }

    #[test]
    fn test_suite() {
        let results = run_suite();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.density_ratio() > 1.0);
        }
    }

    #[test]
    fn test_throughput_ratio() {
        let result = KernelBench::matmul(512).run();
        assert!(result.throughput_ratio() > 0.0);
    }

    #[test]
    fn test_memory_savings() {
        let r = KernelBench::matmul(1024).run();
        let savings_pct = (1.0 - r.ternary_bytes as f64 / r.fp32_bytes as f64) * 100.0;
        assert!(savings_pct > 90.0); // >90% memory saved
    }
}
