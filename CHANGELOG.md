# Changelog

All notable changes to axolotl-rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold with Rust port of Axolotl
- YAML configuration parsing with 3 presets (LLaMA-2, Mistral, Phi-3)
- Dataset loaders for 4 formats: Alpaca, ShareGPT, Completion, Custom
- CLI interface with commands: `validate`, `train`, `merge`, `init`
- Error handling with comprehensive error types
- Mock implementations for PEFT, QLoRA, and Unsloth
- CI/CD pipeline with GitHub Actions
- GPU testing support (CUDA and ROCm)
- Codecov integration with 75% target coverage
- 9 comprehensive benchmarks for config parsing
- Extensive test suite:
  - 18 error handling tests
  - 28 config validation tests  
  - Tests for all dataset formats
  - Trainer lifecycle tests
- MIT license
- Documentation with early development status disclosure
- Contributing guidelines
- Test coverage plan targeting 80%

### Changed
- Updated from candle 0.4 to candle 0.8
- Fixed 54 clippy warnings
- Improved error messages and context
- Enhanced configuration validation

### Fixed
- Compilation issues in initial scaffold
- Workspace manifest configuration
- TemplateError handling in progress bars

## [0.1.0-dev] - 2026-01-10

### Status
**Early Development - Framework Scaffold**

This is an initial development release. The configuration system, CLI, and dataset loaders are functional. Core training functionality (model loading, actual training loops, adapter management, checkpoint handling) is planned for future releases.

**What Works:**
- ✅ YAML configuration parsing and validation
- ✅ Dataset loading (all 4 formats)
- ✅ CLI argument parsing
- ✅ Configuration presets

**What's Planned:**
- 🚧 Model loading from HuggingFace Hub
- 🚧 LoRA/QLoRA adapter implementation
- 🚧 Actual training loop with forward/backward passes
- 🚧 Checkpoint saving and loading
- 🚧 Adapter merging
- 🚧 Multi-GPU distributed training

### Project Metrics
- **Lines of Code**: ~1,500 Rust LOC
- **Test Coverage**: ~60-70% (48+ tests)
- **Dependencies**: Candle 0.8, Tokenizers 0.20, Serde 1.0
- **Platform Support**: Linux, macOS (Windows untested)
- **License**: MIT

### Development Team
- Tyler Zervas (@tzervas) - Primary author

---

## Release History

### Version 0.1.0-dev (January 10, 2026)
Initial development release establishing project structure and core scaffolding.

---

[Unreleased]: https://github.com/tzervas/axolotl-rs/compare/v0.1.0-dev...HEAD
[0.1.0-dev]: https://github.com/tzervas/axolotl-rs/releases/tag/v0.1.0-dev
