use crate::BenchmarkResult;

/// Formatted report comparing all benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    results: Vec<BenchmarkResult>,
    title: String,
}

impl BenchmarkReport {
    /// Create a new report from results.
    pub fn new(results: Vec<BenchmarkResult>) -> Self {
        Self {
            results,
            title: "Ternary Benchmark Report".to_string(),
        }
    }

    /// Set a custom title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Format the full report as a string.
    pub fn format(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{}\n", self.title));
        out.push_str(&"=".repeat(self.title.len()));
        out.push_str("\n\n");

        if self.results.is_empty() {
            out.push_str("No benchmark results.\n");
            return out;
        }

        // Header
        out.push_str(&format!(
            "{:<30} {:>10}  {:>14}  {:>15}\n",
            "Benchmark", "Iterations", "Mean (µs)", "Throughput (iter/s)"
        ));
        out.push_str(&"-".repeat(75));
        out.push('\n');

        // Results sorted by name
        let mut sorted = self.results.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));

        for r in &sorted {
            out.push_str(&format!(
                "{:<30} {:>10}  {:>14.2}  {:>15.0}\n",
                r.name, r.iterations, r.mean_us, r.throughput
            ));
        }

        // Summary
        out.push('\n');
        let total_iters: u64 = self.results.iter().map(|r| r.iterations).sum();
        let total_us: u64 = self.results.iter().map(|r| r.total_us).sum();
        let total_s = total_us as f64 / 1_000_000.0;

        out.push_str(&format!("Total: {} benchmarks, {} iterations, {:.3}s\n",
            self.results.len(), total_iters, total_s));

        if let (Some(fast), Some(slow)) = (self.fastest(), self.slowest()) {
            out.push_str(&format!("Fastest: {} ({:.2} µs/iter)\n", fast.name, fast.mean_us));
            out.push_str(&format!("Slowest: {} ({:.2} µs/iter)\n", slow.name, slow.mean_us));
        }

        out
    }

    /// Find the fastest benchmark.
    pub fn fastest(&self) -> Option<&BenchmarkResult> {
        self.results.iter().min_by(|a, b| a.mean_us.partial_cmp(&b.mean_us).unwrap())
    }

    /// Find the slowest benchmark.
    pub fn slowest(&self) -> Option<&BenchmarkResult> {
        self.results.iter().max_by(|a, b| a.mean_us.partial_cmp(&b.mean_us).unwrap())
    }

    /// Get all results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Print the report to stdout.
    pub fn print(&self) {
        print!("{}", self.format());
    }
}

impl std::fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}
