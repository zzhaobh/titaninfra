"""Unit tests for hf roundtrip."""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def run_roundtrip(
    hf_path: str,
    model_name: str,
    work_dir: str | None = None,
    export_dtype: str = "float32",
    check_numerical: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """Run roundtrip."""
    import torch

    errors = []
    result = {
        "success": False,
        "max_abs_diff": float("nan"),
        "num_params": 0,
        "num_mismatched": 0,
        "errors": errors,
    }

    try:
        import torchtitan.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import (
            HuggingFaceStorageReader,
            HuggingFaceStorageWriter,
        )
        from torchtitan.components.checkpointer import ModelWrapper
    except ImportError as e:
        errors.append(f"torchtitan 未安装：{e}")
        return result

    if "/" not in model_name:
        errors.append(f"model_name 格式错误：{model_name!r}，期望 'family/flavor'")
        return result

    family, flavor = model_name.split("/", 1)

    import importlib
    try:
        model_module = importlib.import_module(f"torchtitan.models.{family}")
        model_spec = model_module.model_registry(flavor)
    except (ImportError, KeyError) as e:
        errors.append(f"加载模型 spec 失败：{e}")
        return result

    try:
        sd_adapter = model_spec.state_dict_adapter(model_spec.model, hf_path)
    except Exception as e:
        errors.append(f"创建 state_dict_adapter 失败：{e}")
        return result

    if sd_adapter is None:
        errors.append(
            f"{family}/{flavor} 没有 state_dict_adapter，无法做 HF↔DCP 转换。"
            "请先实现 model_spec.state_dict_adapter。"
        )
        return result

    tmp_dir = None
    if work_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="titaninfra_roundtrip_")
        work_dir = tmp_dir
    else:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    dcp_dir = str(Path(work_dir) / "dcp")
    hf_out_dir = str(Path(work_dir) / "hf_out")

    try:
        logger.info(f"[1/3] 加载 HF 权重：{hf_path}")
        model_config = model_spec.model

        with torch.device("cpu"):
            model = model_config.build()
        model = ModelWrapper(model)
        state_dict = model._get_state_dict()

        hf_state_dict = sd_adapter.to_hf(state_dict)

        import torch.distributed.checkpoint as dcp
        dcp.load(
            hf_state_dict,
            storage_reader=HuggingFaceStorageReader(path=hf_path),
        )

        state_dict_tt = sd_adapter.from_hf(hf_state_dict)

        result["num_params"] = sum(
            t.numel() for t in state_dict_tt.values() if isinstance(t, torch.Tensor)
        )
        logger.info(f"   参数总数：{result['num_params']:,}")

        logger.info(f"[2/3] 保存 DCP checkpoint：{dcp_dir}")
        Path(dcp_dir).mkdir(parents=True, exist_ok=True)
        dcp.save(state_dict_tt, checkpoint_id=dcp_dir)

        logger.info(f"[3/3] 转换回 HF 格式：{hf_out_dir}")
        Path(hf_out_dir).mkdir(parents=True, exist_ok=True)

        with torch.device("cpu"):
            model2 = model_config.build()
        model2 = ModelWrapper(model2)
        state_dict2 = model2._get_state_dict()

        dcp.load(state_dict2, checkpoint_id=dcp_dir)

        sd_adapter2 = model_spec.state_dict_adapter(model_config, hf_path)
        hf_state_dict_out = sd_adapter2.to_hf(state_dict2)

        from torchtitan.config import TORCH_DTYPE_MAP
        TORCH_DTYPE_MAP.get(export_dtype, torch.float32)

        storage_writer = HuggingFaceStorageWriter(
            path=hf_out_dir,
            save_distributed=True,
            fqn_to_index_mapping=sd_adapter2.fqn_to_index_mapping,
            enable_consolidation=True,
        )
        dcp.save(hf_state_dict_out, storage_writer=storage_writer)

        if check_numerical:
            logger.info("[4/4] 数值比较...")
            max_diff, num_mismatched = _compare_state_dicts(
                state_dict_tt, state_dict2, atol=atol, rtol=rtol
            )
            result["max_abs_diff"] = max_diff
            result["num_mismatched"] = num_mismatched

            if num_mismatched > 0:
                errors.append(
                    f"数值不匹配：{num_mismatched} 个 tensor 超出容忍度 "
                    f"(atol={atol}, rtol={rtol})，最大绝对误差={max_diff:.2e}"
                )
            else:
                logger.info(f"   数值验证通过，最大绝对误差：{max_diff:.2e}")

        result["success"] = len(errors) == 0

    except Exception as e:
        errors.append(f"转换过程中出现异常：{type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def run_roundtrip_random_init(model_name: str, work_dir: str | None = None) -> dict:
    """Run roundtrip random init."""
    import torch

    errors = []
    result = {
        "success": False,
        "max_abs_diff": float("nan"),
        "num_params": 0,
        "num_mismatched": 0,
        "errors": errors,
    }

    if "/" not in model_name:
        errors.append(f"model_name 格式错误：{model_name!r}")
        return result

    family, flavor = model_name.split("/", 1)

    import importlib
    try:
        model_module = importlib.import_module(f"torchtitan.models.{family}")
        model_spec = model_module.model_registry(flavor)
    except (ImportError, KeyError) as e:
        errors.append(f"加载模型 spec 失败：{e}")
        return result

    tmp_dir = None
    if work_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="titaninfra_roundtrip_rand_")
        work_dir = tmp_dir
    else:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    dcp_dir = str(Path(work_dir) / "dcp")

    try:
        import torch.distributed.checkpoint as dcp
        from torchtitan.components.checkpointer import ModelWrapper

        model_config = model_spec.model
        logger.info(f"[随机初始化] 构建模型 {model_name}...")

        with torch.device("cpu"):
            model = model_config.build()
        model = ModelWrapper(model)
        state_dict = model._get_state_dict()

        result["num_params"] = sum(
            t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)
        )
        logger.info(f"   参数总数：{result['num_params']:,}")

        logger.info(f"[随机初始化] 保存 DCP：{dcp_dir}")
        Path(dcp_dir).mkdir(parents=True, exist_ok=True)
        dcp.save(state_dict, checkpoint_id=dcp_dir)

        with torch.device("cpu"):
            model2 = model_config.build()
        model2 = ModelWrapper(model2)
        state_dict2 = model2._get_state_dict()
        dcp.load(state_dict2, checkpoint_id=dcp_dir)

        max_diff, num_mismatched = _compare_state_dicts(
            state_dict, state_dict2, atol=1e-6, rtol=1e-6
        )
        result["max_abs_diff"] = max_diff
        result["num_mismatched"] = num_mismatched

        if num_mismatched > 0:
            errors.append(
                f"DCP 保存/加载后数值不一致：{num_mismatched} 个 tensor 不匹配"
            )
        else:
            logger.info("   DCP 保存/加载数值验证通过")

        sd_adapter = model_spec.state_dict_adapter(model_config, None)
        if sd_adapter is not None:
            logger.info("[随机初始化] 测试 state_dict_adapter HF↔DCP 转换...")
            hf_sd = sd_adapter.to_hf(state_dict)
            tt_sd = sd_adapter.from_hf(hf_sd)

            for key in state_dict:
                if key in tt_sd:
                    orig_shape = state_dict[key].shape if isinstance(state_dict[key], torch.Tensor) else None
                    new_shape = tt_sd[key].shape if isinstance(tt_sd[key], torch.Tensor) else None
                    if orig_shape != new_shape:
                        errors.append(f"shape 不匹配 [{key}]：{orig_shape} vs {new_shape}")

            if not errors:
                logger.info("   state_dict_adapter 形状检查通过")
        else:
            logger.info("   该模型无 state_dict_adapter，跳过 HF↔DCP 转换测试")

        result["success"] = len(errors) == 0

    except Exception as e:
        errors.append(f"测试过程中出现异常：{type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def _compare_state_dicts(
    sd1: dict,
    sd2: dict,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[float, int]:
    """Compare state dicts."""
    import torch

    max_diff = 0.0
    num_mismatched = 0

    common_keys = set(sd1.keys()) & set(sd2.keys())
    if not common_keys:
        return max_diff, 0

    for key in common_keys:
        t1, t2 = sd1[key], sd2[key]
        if not (isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor)):
            continue

        if t1.shape != t2.shape:
            num_mismatched += 1
            logger.debug(f"  [{key}] 形状不匹配：{t1.shape} vs {t2.shape}")
            continue

        t1_fp32 = t1.float()
        t2_fp32 = t2.float()
        diff = (t1_fp32 - t2_fp32).abs()
        max_diff = max(max_diff, diff.max().item())

        if not torch.allclose(t1_fp32, t2_fp32, atol=atol, rtol=rtol):
            num_mismatched += 1
            logger.debug(
                f"  [{key}] 不匹配：max_diff={diff.max().item():.2e}, "
                f"mean_diff={diff.mean().item():.2e}"
            )

    return max_diff, num_mismatched


# ============================================================
# ============================================================


def pytest_addoption(parser):
    """Pytest addoption."""
    parser.addoption("--hf-path", default=None, help="HF 权重目录路径")
    parser.addoption("--model-name", default="qwen3/0.6B", help="模型名称，如 qwen3/8B")
    parser.addoption("--work-dir", default=None, help="中间文件目录（None=使用临时目录）")
    parser.addoption("--random-init", action="store_true", default=False, help="使用随机初始化（不需要 HF 权重）")


def pytest_configure(config):
    config.addinivalue_line("markers", "roundtrip: HF roundtrip 验证测试")


@pytest.mark.roundtrip
def test_random_init_roundtrip(request):
    """Verify random init roundtrip."""
    pytest.importorskip("torch")
    pytest.importorskip("torchtitan")
    model_name = request.config.getoption("--model-name", default="qwen3/0.6B")
    work_dir = request.config.getoption("--work-dir", default=None)

    result = run_roundtrip_random_init(model_name=model_name, work_dir=work_dir)

    assert result["success"], (
        "随机初始化 roundtrip 失败：\n" + "\n".join(result["errors"])
    )
    assert result["num_mismatched"] == 0, (
        f"DCP 保存/加载后 {result['num_mismatched']} 个参数数值不一致"
    )


@pytest.mark.roundtrip
def test_hf_roundtrip(request):
    """Verify hf roundtrip."""
    hf_path = request.config.getoption("--hf-path", default=None)
    model_name = request.config.getoption("--model-name", default="qwen3/0.6B")
    work_dir = request.config.getoption("--work-dir", default=None)

    if hf_path is None:
        pytest.skip("需要通过 --hf-path 指定 HF 权重目录")

    result = run_roundtrip(
        hf_path=hf_path,
        model_name=model_name,
        work_dir=work_dir,
        export_dtype="float32",
        check_numerical=True,
    )

    assert result["success"], (
        "HF roundtrip 失败：\n" + "\n".join(result["errors"])
    )
    assert result["num_mismatched"] == 0, (
        f"转换后 {result['num_mismatched']} 个参数数值不一致，"
        f"最大绝对误差：{result['max_abs_diff']:.2e}"
    )


# ============================================================
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="HuggingFace ↔ DCP Roundtrip 验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python tests/test_hf_roundtrip.py --model-name qwen3/0.6B --random-init

  python tests/test_hf_roundtrip.py \\
    --model-name qwen3/8B \\
    --hf-path /path/to/Qwen3-8B \\
    --work-dir /tmp/roundtrip
        """,
    )
    parser.add_argument("--model-name", default="qwen3/0.6B", help="模型名称，如 qwen3/8B")
    parser.add_argument("--hf-path", default=None, help="HF 权重目录路径")
    parser.add_argument("--work-dir", default=None, help="中间文件目录（None=临时目录）")
    parser.add_argument("--random-init", action="store_true", help="使用随机初始化")
    parser.add_argument(
        "--export-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="导出精度",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.random_init or args.hf_path is None:
        print(f"模式：随机初始化 roundtrip  |  模型：{args.model_name}")
        result = run_roundtrip_random_init(
            model_name=args.model_name,
            work_dir=args.work_dir,
        )
    else:
        print(f"模式：完整 HF roundtrip  |  模型：{args.model_name}  |  权重：{args.hf_path}")
        result = run_roundtrip(
            hf_path=args.hf_path,
            model_name=args.model_name,
            work_dir=args.work_dir,
            export_dtype=args.export_dtype,
        )

    print("\n" + "=" * 60)
    print(f"状态：{'✓ 通过' if result['success'] else '✗ 失败'}")
    print(f"参数总数：{result['num_params']:,}")
    if not result['success']:
        print(f"不匹配参数数：{result['num_mismatched']}")
        print(f"最大绝对误差：{result['max_abs_diff']:.2e}")
        print("\n错误信息：")
        for err in result["errors"]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"最大绝对误差：{result['max_abs_diff']:.2e}")
        print("=" * 60)
