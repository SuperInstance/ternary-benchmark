use std::time::Instant;

use crate::{play_iterated, TernaryChoice, TernaryStrategy, BenchmarkResult};

/// Benchmark: time N generations of evolution for a population of strategies.
pub struct EvolutionBenchmark {
    /// Population size.
    pub population_size: usize,
    /// Number of rounds per matchup.
    pub rounds: usize,
    /// Strategies in the population.
    pub strategies: Vec<TernaryStrategy>,
}

impl EvolutionBenchmark {
    /// Create a new evolution benchmark with a random-ish population.
    pub fn new(population_size: usize, rounds: usize) -> Self {
        let strategies = (0..population_size)
            .map(|i| {
                let val = (i * 7919) as u32; // deterministic pseudo-random
                TernaryStrategy::decode(val)
            })
            .collect();
        Self { population_size, rounds, strategies }
    }

    /// Compute fitness for each strategy by playing against all others.
    pub fn compute_fitness(&self) -> Vec<f64> {
        let n = self.strategies.len();
        let mut fitness = vec![0.0f64; n];
        for i in 0..n {
            let mut total = 0.0;
            for j in 0..n {
                if i != j {
                    let (score, _) = play_iterated(&self.strategies[i], &self.strategies[j], self.rounds);
                    total += score;
                }
            }
            fitness[i] = total;
        }
        fitness
    }

    /// Selection: pick top 50% by fitness.
    pub fn select(&self, fitness: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = fitness.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed[..indexed.len() / 2].iter().map(|(i, _)| *i).collect()
    }

    /// Crossover two strategies to produce offspring.
    pub fn crossover(a: &TernaryStrategy, b: &TernaryStrategy) -> TernaryStrategy {
        let mut responses = [TernaryChoice::Cooperate; 4];
        for i in 0..4 {
            responses[i] = if i < 2 { a.responses[i] } else { b.responses[i] };
        }
        TernaryStrategy::new(responses)
    }

    /// Run N generations and measure time.
    pub fn run(&mut self, generations: usize) -> BenchmarkResult {
        let start = Instant::now();
        let mut total_ops = 0u64;

        for _gen in 0..generations {
            let fitness = self.compute_fitness();
            let survivors = self.select(&fitness);

            let mut new_pop = Vec::with_capacity(self.population_size);
            for &i in &survivors {
                new_pop.push(self.strategies[i].clone());
            }

            // Fill rest with crossover offspring
            let s_len = survivors.len();
            while new_pop.len() < self.population_size {
                let p1 = survivors[total_ops as usize % s_len];
                let p2 = survivors[(total_ops as usize + 1) % s_len];
                new_pop.push(Self::crossover(&self.strategies[p1], &self.strategies[p2]));
                total_ops += 1;
            }

            self.strategies = new_pop;
        }

        let elapsed = start.elapsed().as_micros() as u64;
        BenchmarkResult::new(
            format!("evolution_{}pop_{}gen", self.population_size, generations),
            generations as u64,
            elapsed,
        )
    }
}
