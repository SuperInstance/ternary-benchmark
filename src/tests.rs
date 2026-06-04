use crate::*;

#[test]
fn test_ternary_choice_index() {
    assert_eq!(TernaryChoice::Cooperate.index(), 0);
    assert_eq!(TernaryChoice::Defect.index(), 1);
    assert_eq!(TernaryChoice::Withhold.index(), 2);
}

#[test]
fn test_ternary_choice_all() {
    let all = TernaryChoice::all();
    assert_eq!(all.len(), 3);
    assert!(all.contains(&TernaryChoice::Cooperate));
    assert!(all.contains(&TernaryChoice::Defect));
    assert!(all.contains(&TernaryChoice::Withhold));
}

#[test]
fn test_strategy_encode_decode_roundtrip() {
    let s = TernaryStrategy::new([
        TernaryChoice::Cooperate,
        TernaryChoice::Defect,
        TernaryChoice::Withhold,
        TernaryChoice::Cooperate,
    ]);
    let encoded = s.encode();
    let decoded = TernaryStrategy::decode(encoded);
    assert_eq!(s, decoded);
}

#[test]
fn test_strategy_encode_decode_all_zeros() {
    let s = TernaryStrategy::new([TernaryChoice::Cooperate; 4]);
    assert_eq!(s.encode(), 0);
    assert_eq!(TernaryStrategy::decode(0), s);
}

#[test]
fn test_payoff_cooperate_cooperate() {
    let (a, b) = ternary_payoff(TernaryChoice::Cooperate, TernaryChoice::Cooperate);
    assert_eq!(a, 3.0);
    assert_eq!(b, 3.0);
}

#[test]
fn test_payoff_cooperate_defect() {
    let (a, b) = ternary_payoff(TernaryChoice::Cooperate, TernaryChoice::Defect);
    assert_eq!(a, 0.0);
    assert_eq!(b, 5.0);
}

#[test]
fn test_payoff_withhold_neutral() {
    let (a, b) = ternary_payoff(TernaryChoice::Withhold, TernaryChoice::Withhold);
    assert_eq!(a, 1.0);
    assert_eq!(b, 1.0);
}

#[test]
fn test_payoff_defect_withhold() {
    let (a, b) = ternary_payoff(TernaryChoice::Defect, TernaryChoice::Withhold);
    assert_eq!(a, 2.0);
    assert_eq!(b, 0.0);
}

#[test]
fn test_play_iterated_all_cooperate() {
    let s = TernaryStrategy::new([TernaryChoice::Cooperate; 4]);
    let (a, b) = play_iterated(&s, &s, 100);
    assert_eq!(a, b);
    assert!(a > 0.0);
}

#[test]
fn test_play_iterated_tit_for_tat_vs_defect() {
    // Tit-for-tat: respond in kind, open with cooperate
    let tft = TernaryStrategy::new([
        TernaryChoice::Cooperate,
        TernaryChoice::Defect,
        TernaryChoice::Withhold,
        TernaryChoice::Cooperate,
    ]);
    let always_defect = TernaryStrategy::new([TernaryChoice::Defect; 4]);
    let (tft_score, ad_score) = play_iterated(&tft, &always_defect, 100);
    // TFT should score less against always-defect (only first round is cooperate)
    assert!(tft_score < ad_score);
}

#[test]
fn test_benchmark_result_creation() {
    let r = BenchmarkResult::new("test", 1000, 5000);
    assert_eq!(r.name, "test");
    assert_eq!(r.iterations, 1000);
    assert_eq!(r.total_us, 5000);
    assert!((r.mean_us - 5.0).abs() < 0.01);
}

#[test]
fn test_benchmark_result_measure() {
    let r = BenchmarkResult::measure("test_measure", 1000, || {
        let mut x = 0u64;
        for i in 0..100 {
            x += i;
        }
        std::hint::black_box(x);
    });
    assert_eq!(r.name, "test_measure");
    assert_eq!(r.iterations, 1000);
    assert!(r.total_us > 0);
    assert!(r.throughput > 0.0);
}

#[test]
fn test_exhaustive_benchmark_enumerate() {
    let bench = ExhaustiveBenchmark::new(4);
    let strategies = bench.enumerate_decoded();
    assert_eq!(strategies.len(), 81); // 3^4
}

#[test]
fn test_exhaustive_benchmark_total() {
    let bench = ExhaustiveBenchmark::new(3);
    assert_eq!(bench.total_strategies(), 27);
}

#[test]
fn test_evolution_benchmark_fitness() {
    let bench = EvolutionBenchmark::new(10, 50);
    let fitness = bench.compute_fitness();
    assert_eq!(fitness.len(), 10);
    // At least one should have nonzero fitness
    assert!(fitness.iter().any(|&f| f > 0.0));
}

#[test]
fn test_evolution_benchmark_crossover() {
    let a = TernaryStrategy::new([
        TernaryChoice::Cooperate,
        TernaryChoice::Cooperate,
        TernaryChoice::Cooperate,
        TernaryChoice::Cooperate,
    ]);
    let b = TernaryStrategy::new([
        TernaryChoice::Defect,
        TernaryChoice::Defect,
        TernaryChoice::Defect,
        TernaryChoice::Defect,
    ]);
    let child = EvolutionBenchmark::crossover(&a, &b);
    assert_eq!(child.responses[0], TernaryChoice::Cooperate);
    assert_eq!(child.responses[1], TernaryChoice::Cooperate);
    assert_eq!(child.responses[2], TernaryChoice::Defect);
    assert_eq!(child.responses[3], TernaryChoice::Defect);
}

#[test]
fn test_ecology_benchmark_step() {
    let mut eco = EcologyBenchmark::new(100.0, 50.0, 25.0);
    let initial = eco.populations;
    eco.step();
    // Populations should change after a step
    assert_ne!(eco.populations, initial);
}

#[test]
fn test_ecology_benchmark_simulate() {
    let mut eco = EcologyBenchmark::new(100.0, 50.0, 25.0);
    let result = eco.simulate(1000);
    // Populations should remain non-negative and finite
    for &p in &result {
        assert!(p >= 0.0);
        assert!(p.is_finite());
    }
}

#[test]
fn test_ecology_rk4_step() {
    let mut eco = EcologyBenchmark::new(100.0, 50.0, 25.0);
    let initial = eco.populations;
    eco.step_rk4();
    assert_ne!(eco.populations, initial);
}

#[test]
fn test_benchmark_suite_run_all() {
    let suite = BenchmarkSuite::run_all(10, 100, 20, 50);
    assert!(suite.results().len() >= 5);
    assert!(suite.fastest().is_some());
    assert!(suite.slowest().is_some());
}

#[test]
fn test_benchmark_report_format() {
    let results = vec![
        BenchmarkResult::new("test_a", 1000, 500),
        BenchmarkResult::new("test_b", 2000, 1000),
    ];
    let report = BenchmarkReport::new(results);
    let formatted = report.format();
    assert!(formatted.contains("test_a"));
    assert!(formatted.contains("test_b"));
    assert!(formatted.contains("Fastest"));
    assert!(formatted.contains("Slowest"));
}

#[test]
fn test_benchmark_report_display() {
    let results = vec![BenchmarkResult::new("display_test", 100, 200)];
    let report = BenchmarkReport::new(results);
    let display = format!("{}", report);
    assert!(display.contains("display_test"));
}

#[test]
fn test_scaling_benchmark_runs() {
    let results = ScalingBenchmark::run_ecology_scaling();
    assert_eq!(results.len(), 4); // 4 scales
    for r in &results {
        assert!(r.throughput > 0.0);
    }
}

#[test]
fn test_lv_params_default() {
    let p = ecology::LvParams::default();
    assert_eq!(p.alpha, 1.0);
    assert_eq!(p.beta, 0.1);
}
