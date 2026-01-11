# Contributing to axolotl-rs

Thank you for your interest in contributing to axolotl-rs! This document provides guidelines for contributing to the project.

## Development Status

axolotl-rs is in early development. The configuration system, CLI interface, and dataset loaders are functional, but core training functionality is planned for future releases. See the [README](README.md) for details on what's implemented vs. planned.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Git
- (Optional) CUDA 12.0+ for GPU support

### Setup

```bash
git clone https://github.com/tzervas/axolotl-rs
cd axolotl-rs
cargo build
cargo test
```

## Development Workflow

### Branch Strategy

- `main` - stable releases (minimal activity currently)
- `dev` - active development branch
- Feature branches - `feat/description`, `test/description`, `fix/description`, `docs/description`

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: bug fix
test: add or modify tests
docs: documentation changes
chore: maintenance tasks
refactor: code restructuring
perf: performance improvements
ci: CI/CD changes
```

Examples:
```bash
git commit -m "feat: implement LoRA adapter loading"
git commit -m "test: add dataset loader validation tests"
git commit -m "fix: correct quantization bit handling"
```

### Pull Request Process

1. **Create a feature branch** from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Add tests** for new functionality

4. **Run the test suite**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

5. **Update documentation** if needed

6. **Rebase onto latest dev** before submitting:
   ```bash
   git fetch origin
   git rebase origin/dev
   ```

7. **Submit PR** targeting the `dev` branch with:
   - Clear description of changes
   - Reference to related issues
   - Test coverage summary

## Code Standards

### Style

- Follow Rust standard formatting (`rustfmt`)
- Maximum line length: 100 characters
- Use `cargo clippy` and address all warnings

### Testing

- Unit tests in module files (`#[cfg(test)] mod tests { ... }`)
- Integration tests in `tests/` directory
- Target: 80% code coverage
- See [TEST_COVERAGE_PLAN.md](TEST_COVERAGE_PLAN.md) for details

### Documentation

- Public APIs must have doc comments
- Include examples in doc comments where helpful
- Run `cargo doc --no-deps` to verify documentation builds

### Error Handling

- Use `Result<T>` for fallible operations (aliased to `Result<T, AxolotlError>`)
- Use `AxolotlError` variants for domain errors
- Provide context with error messages

Example:
```rust
pub fn load_model(path: &str) -> Result<Model> {
    if !Path::new(path).exists() {
        return Err(AxolotlError::Model(format!("Model not found: {}", path)));
    }
    // ...
}
```

## Project Structure

```
axolotl-rs/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── lib.rs           # Library exports
│   ├── config.rs        # YAML configuration parsing
│   ├── dataset.rs       # Dataset loading (4 formats)
│   ├── model.rs         # Model operations (stubs)
│   ├── trainer.rs       # Training loop (stubs)
│   ├── error.rs         # Error types
│   ├── cli.rs           # CLI command routing
│   └── mocks/           # Mock implementations
├── examples/            # Example configurations
├── benches/             # Benchmarks
├── tests/               # Integration tests
└── .github/workflows/   # CI/CD pipelines
```

## Testing Guidelines

### Unit Tests

Place tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Integration Tests

Place in `tests/` directory:

```rust
// tests/integration_test.rs
use axolotl_rs::config::AxolotlConfig;

#[test]
fn test_end_to_end_workflow() {
    // Test implementation
}
```

### Test Data

Use `tempfile` crate for temporary test files:

```rust
use tempfile::NamedTempFile;

let mut file = NamedTempFile::new().unwrap();
writeln!(file, "test data").unwrap();
```

## Areas for Contribution

### High Priority

1. **Core Training Loop** - Implement actual forward/backward passes
2. **Model Loading** - HuggingFace Hub integration
3. **Adapter Implementation** - LoRA, QLoRA with real backends
4. **Checkpoint Management** - Save/load model weights
5. **Test Coverage** - Expand to 80% target

### Medium Priority

6. **Multi-GPU Support** - Distributed training
7. **Optimizer Implementations** - AdamW, SGD, etc.
8. **Learning Rate Schedulers** - Cosine, linear, polynomial
9. **Mixed Precision Training** - FP16/BF16 support
10. **Gradient Accumulation** - Memory-efficient training

### Documentation

11. **Examples** - Real-world fine-tuning recipes
12. **API Documentation** - Comprehensive module docs
13. **Tutorials** - Step-by-step guides

## Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **PRs**: Reference related issues in PR descriptions

## Code Review

All submissions require review. Reviewers will check for:

- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Breaking changes

## License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE-MIT](LICENSE-MIT) for details.

## Questions?

Open an issue or discussion on GitHub if you have questions about contributing!
