//! # ternary-benchmark
//!
//! Standardized benchmarks for ternary agent systems — reproducible performance numbers.
//!
//! This crate provides benchmarking utilities for measuring performance of
//! ternary strategy systems, including exhaustive enumeration, evolutionary
//! optimization, and Lotka-Volterra ecology simulations.

mod benchmark;
mod ecology;
mod evolution;
mod exhaustive;
mod report;
mod scaling;
mod tests;

pub use benchmark::{BenchmarkResult, BenchmarkSuite};
pub use ecology::EcologyBenchmark;
pub use evolution::EvolutionBenchmark;
pub use exhaustive::ExhaustiveBenchmark;
pub use report::BenchmarkReport;
pub use scaling::ScalingBenchmark;

/// A ternary choice: Cooperate, Defect, or Withhold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryChoice {
    Cooperate,
    Defect,
    Withhold,
}

impl TernaryChoice {
    /// Returns all three ternary choices.
    pub fn all() -> [TernaryChoice; 3] {
        [TernaryChoice::Cooperate, TernaryChoice::Defect, TernaryChoice::Withhold]
    }

    /// Convert to a numeric index (0, 1, or 2).
    pub fn index(self) -> usize {
        match self {
            TernaryChoice::Cooperate => 0,
            TernaryChoice::Defect => 1,
            TernaryChoice::Withhold => 2,
        }
    }
}

/// A strategy for a single round of ternary interaction: a mapping from
/// opponent's last move to our next move.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TernaryStrategy {
    /// Response to each possible opponent move (or initial move at index 3).
    /// Index 0 = respond to Cooperate, 1 = Defect, 2 = Withhold, 3 = opening move.
    pub responses: [TernaryChoice; 4],
}

impl TernaryStrategy {
    /// Create a new strategy from response array.
    pub fn new(responses: [TernaryChoice; 4]) -> Self {
        Self { responses }
    }

    /// Get the opening move (response to no prior context).
    pub fn opening(&self) -> TernaryChoice {
        self.responses[3]
    }

    /// Get the response to an opponent's move.
    pub fn respond(&self, opponent_last: TernaryChoice) -> TernaryChoice {
        self.responses[opponent_last.index()]
    }

    /// Encode as a single u32 (each choice is 2 bits, 4 choices = 8 bits).
    pub fn encode(&self) -> u32 {
        let mut val = 0u32;
        for (i, &c) in self.responses.iter().enumerate() {
            val |= (c.index() as u32) << (i * 2);
        }
        val
    }

    /// Decode from a u32.
    pub fn decode(val: u32) -> Self {
        let choices = TernaryChoice::all();
        let mut responses = [TernaryChoice::Cooperate; 4];
        for i in 0..4 {
            responses[i] = choices[(((val >> (i * 2)) & 3) % 3) as usize];
        }
        Self { responses }
    }
}

/// Compute payoff for a single interaction between two ternary choices.
pub fn ternary_payoff(a: TernaryChoice, b: TernaryChoice) -> (f64, f64) {
    // Standard ternary payoff matrix:
    // Both cooperate: (3, 3)
    // One cooperates, other defects: (0, 5) / (5, 0)
    // Both defect: (1, 1)
    // Withhold always gives (1, 1) regardless of opponent
    // Cooperate vs Withhold: (1, 1)
    // Defect vs Withhold: (2, 0)
    match (a, b) {
        (TernaryChoice::Cooperate, TernaryChoice::Cooperate) => (3.0, 3.0),
        (TernaryChoice::Cooperate, TernaryChoice::Defect) => (0.0, 5.0),
        (TernaryChoice::Defect, TernaryChoice::Cooperate) => (5.0, 0.0),
        (TernaryChoice::Defect, TernaryChoice::Defect) => (1.0, 1.0),
        (TernaryChoice::Withhold, _) | (_, TernaryChoice::Withhold) => {
            match (a, b) {
                (TernaryChoice::Defect, TernaryChoice::Withhold) => (2.0, 0.0),
                (TernaryChoice::Withhold, TernaryChoice::Defect) => (0.0, 2.0),
                _ => (1.0, 1.0),
            }
        }
    }
}

/// Run an iterated ternary game between two strategies for N rounds.
pub fn play_iterated(s1: &TernaryStrategy, s2: &TernaryStrategy, rounds: usize) -> (f64, f64) {
    let mut score1 = 0.0f64;
    let mut score2 = 0.0f64;
    let mut last1 = s1.opening();
    let mut last2 = s2.opening();

    for _ in 0..rounds {
        let (p1, p2) = ternary_payoff(last1, last2);
        score1 += p1;
        score2 += p2;
        let next1 = s1.respond(last2);
        let next2 = s2.respond(last1);
        last1 = next1;
        last2 = next2;
    }

    (score1, score2)
}
