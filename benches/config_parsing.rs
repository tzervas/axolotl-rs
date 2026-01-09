use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_config_parsing(c: &mut Criterion) {
    // Benchmarks for config parsing
    let group = c.benchmark_group("config_parsing");
    group.finish();
}

criterion_group!(benches, benchmark_config_parsing);
criterion_main!(benches);
