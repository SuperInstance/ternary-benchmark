use std::time::Instant;

use crate::BenchmarkResult;

/// Lotka-Volterra parameters for three-species competition.
#[derive(Debug, Clone)]
pub struct LvParams {
    /// Prey growth rate.
    pub alpha: f64,
    /// Predation rate (prey-predator).
    pub beta: f64,
    /// Predator death rate.
    pub delta: f64,
    /// Predator growth from prey.
    pub gamma: f64,
    /// Omnivore interaction with predator.
    pub epsilon: f64,
    /// Omnivore self-regulation.
    pub zeta: f64,
}

impl Default for LvParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.1,
            delta: 1.5,
            gamma: 0.075,
            epsilon: 0.05,
            zeta: 0.5,
        }
    }
}

/// Benchmark: time N steps of Lotka-Volterra simulation.
pub struct EcologyBenchmark {
    /// Population counts for three species: [prey, predator, omnivore].
    pub populations: [f64; 3],
    /// Model parameters.
    pub params: LvParams,
    /// Time step size.
    pub dt: f64,
}

impl EcologyBenchmark {
    /// Create a new ecology benchmark with default parameters.
    pub fn new(prey: f64, predator: f64, omnivore: f64) -> Self {
        Self {
            populations: [prey, predator, omnivore],
            params: LvParams::default(),
            dt: 0.01,
        }
    }

    /// Single Euler step of Lotka-Volterra dynamics.
    pub fn step(&mut self) {
        let [x, y, z] = self.populations;
        let p = &self.params;
        let dt = self.dt;

        let dx = x * (p.alpha - p.beta * y);
        let dy = -y * (p.delta - p.gamma * x - p.epsilon * z);
        let dz = -z * (p.zeta - 0.02 * y);

        self.populations[0] = (x + dx * dt).max(0.0).min(1e6);
        self.populations[1] = (y + dy * dt).max(0.0).min(1e6);
        self.populations[2] = (z + dz * dt).max(0.0).min(1e6);
    }

    /// Run N simulation steps and return the final populations.
    pub fn simulate(&mut self, steps: usize) -> [f64; 3] {
        for _ in 0..steps {
            self.step();
        }
        self.populations
    }

    /// Benchmark: run N steps `iterations` times.
    pub fn run(&mut self, steps: usize, iterations: u64) -> BenchmarkResult {
        let start = Instant::now();
        for _ in 0..iterations {
            // Reset populations each iteration for fairness
            self.populations = [100.0, 50.0, 25.0];
            let result = self.simulate(steps);
            std::hint::black_box(result);
        }
        let elapsed = start.elapsed().as_micros() as u64;
        BenchmarkResult::new(
            format!("ecology_lv_{}steps", steps),
            iterations * steps as u64,
            elapsed,
        )
    }

    /// Run Lotka-Volterra with RK4 integration (higher accuracy).
    pub fn step_rk4(&mut self) {
        let [x0, y0, z0] = self.populations;
        let p = &self.params;
        let dt = self.dt;

        let deriv = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
            let dx = x * (p.alpha - p.beta * y);
            let dy = -y * (p.delta - p.gamma * x - p.epsilon * z);
            let dz = -z * (p.zeta - 0.02 * y);
            (dx, dy, dz)
        };

        let (k1x, k1y, k1z) = deriv(x0, y0, z0);
        let (k2x, k2y, k2z) = deriv(x0 + k1x * dt / 2.0, y0 + k1y * dt / 2.0, z0 + k1z * dt / 2.0);
        let (k3x, k3y, k3z) = deriv(x0 + k2x * dt / 2.0, y0 + k2y * dt / 2.0, z0 + k2z * dt / 2.0);
        let (k4x, k4y, k4z) = deriv(x0 + k3x * dt, y0 + k3y * dt, z0 + k3z * dt);

        self.populations[0] = (x0 + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)).max(0.0).min(1e6);
        self.populations[1] = (y0 + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)).max(0.0).min(1e6);
        self.populations[2] = (z0 + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)).max(0.0).min(1e6);
    }
}
