# 开发与测试

[English](../en/development.md) · [README](../../README.zh-CN.md)

## 环境

使用 Python 3.12 和已验证的 PyTorch 2.15 基线。不要通过隐式依赖解析替换固定的 TorchTitan 引擎。

```bash
git clone https://github.com/zzhaobh/titaninfra.git
cd titaninfra
git checkout open-source
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 核心与开发依赖。
python -m pip install -r requirements-dev.txt

# 兼容的公开 TorchTitan 运行时。
git clone https://github.com/pytorch/torchtitan.git ../torchtitan
git -C ../torchtitan checkout 72e5566782bbdafaa9bafd422733a307319213c5
python -m pip install --no-deps -e ../torchtitan

# 从源码安装 MDS streaming，避开旧版 Transformers<5 元数据限制。
git clone --depth 1 https://github.com/mosaicml/streaming.git ../streaming
(cd ../streaming && git fetch --depth 1 origin 1381ca1213fc95b6573efb44e5ec9f85c765ce1c \
  && git checkout --detach FETCH_HEAD \
  && python -m pip install --no-build-isolation --no-deps .)

# 安装 TitanInfra 本身。
python -m pip install --no-deps -e ".[data,dev]"
```

如果只是运行 CPU 静态/release 检查，安装 `pytest` 的轻量虚拟环境即可。运行时测试需要上述完整依赖。

## 测试夹具

全量测试前下载 tokenizer 素材：

```bash
python tests/fixtures/tokenizers/download_all.py
```

这些文件是生成的测试输入，已被 Git 忽略。部分上游资源需要访问授权或可能不可用。FSB、MATH500、AIME/basic 和多轮测试所需的小型 fixture 已保存在 `tests/fixtures/datasets/`。pytest 启动时，`tests/_fixture_mds.py` 会在运行时依赖可用的环境中，通过公开转换 API 将这些原料的 MDS 形式生成到被忽略的 `data/` 目录。

## CPU 测试

```bash
pytest tests/ -v -n auto -k "not roundtrip" \
    --ignore=tests/alignment \
    --basetemp=$TMPDIR/titaninfra-pytest
```

开发时可先运行聚焦测试：

```bash
pytest tests/test_public_snapshot.py tests/test_public_examples.py -v
pytest tests/test_basic.py tests/test_trainer.py tests/test_config.py -v
python -m py_compile examples/pretrain.py examples/sft.py
```

公开 example 测试只校验配置，不启动训练。Roundtrip、多 GPU、外部权重和部分 tokenizer 测试需要额外硬件、资源或网络，不能将它们表述为自包含的 CPU 覆盖。

## 静态与 release 检查

```bash
git diff --check
ruff check titaninfra/ tests/ scripts/ examples/
python -m build
bash scripts/verify_install.sh
```

`tests/test_public_snapshot.py` 检查必需文档、本地 Markdown 链接、内部目录清理、私有信息残留、BSD 元数据和旧 RL 公共 API 的移除。`tests/test_public_examples.py` 使用真实配置模型和调度校验器验证完整 Pretrain/SFT 配置。

## 文档修改

在同一个改动中同步两种语言：

- `README.md` ↔ `README.zh-CN.md`
- `docs/en/api.md` ↔ `docs/zh-CN/api.md`
- `docs/en/data.md` ↔ `docs/zh-CN/data.md`
- `docs/en/training.md` ↔ `docs/zh-CN/training.md`
- `docs/en/development.md` ↔ `docs/zh-CN/development.md`

示例必须可执行且与当前源码一致。Python 注释和 docstring 使用英文。

## 提交前

1. 检查最终 diff，确认不包含私有路径、凭据、生成数据集、checkpoint 或 agent 专属文件。
2. 运行相关行为测试与 release 检查。
3. 构建 wheel/sdist 并验证公开导入。
4. 记录 Python/PyTorch/TorchTitan 版本以及未验证的硬件/模型组合。
5. 保留 BSD 3-Clause 和第三方声明。

Pull request 流程见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。
