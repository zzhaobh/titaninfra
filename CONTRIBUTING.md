# Contributing to TitanInfra

Thank you for improving TitanInfra.

## Development workflow

1. Fork the repository and create a focused branch.
2. Follow the [English development guide](docs/en/development.md) or [中文开发指南](docs/zh-CN/development.md) to install the pinned TorchTitan runtime and test dependencies.
3. Keep public API, recipes, and both documentation languages synchronized.
4. Add or update behavioral tests for every code change. Do not weaken tests to accommodate a regression.
5. Run `git diff --check`, the relevant pytest suites, package build, and Markdown-link checks before opening a pull request.
6. Explain the motivation, user-visible behavior, validation evidence, and any remaining hardware/model limitations in the pull request.

## Documentation and compatibility

- `README.md` is English; `README.zh-CN.md` is its Chinese counterpart.
- Every guide in `docs/en/` has a same-named guide in `docs/zh-CN/`.
- Use neutral paths such as `/data/...` and `/models/...`; never commit private endpoints, local credentials, generated checkpoints, datasets, or agent-specific files.
- Python comments and docstrings are written in English.
- Pretrain, SFT, and CPT are the supported training modes. Do not present the future self-developed RL mode as an available API.
- Preserve BSD 3-Clause and applicable third-party notices in derived files.

## Questions

Use [GitHub Issues](https://github.com/zzhaobh/titaninfra/issues) for bug reports, feature proposals, and compatibility questions. Include a minimal reproduction and the exact Python, PyTorch, TorchTitan, and TitanInfra revisions.
