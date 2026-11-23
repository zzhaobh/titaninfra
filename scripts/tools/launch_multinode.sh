#!/bin/bash
# 多节点 torchrun 启动脚本（JoyBuilder 集群）
#
# 用法：
#   bash scripts/tools/launch_multinode.sh <train_script.py> [其他参数]
#   bash scripts/tools/launch_multinode.sh scripts/train/pretrain_qwen3_30b_2node16gpu.py
#
# 容错开关：
#   ELASTIC=1 (默认)  启用容错：c10d rendezvous + 自动重启
#   ELASTIC=0          禁用容错：static rendezvous，快速启动，无自动重启
#
# 调试开关：
#   DEBUG=0 (默认)  生产模式：CUDA 异步、无分布式日志、NCCL WARN、最高吞吐
#   DEBUG=1          调试模式：CUDA 同步（栈精确但 TPS 大幅下降）、分布式 INFO、NCCL INFO
#
# 容错参数（仅 ELASTIC=1 时生效）：
#   MAX_RESTARTS=5    bash scripts/tools/launch_multinode.sh ...   # 覆盖默认 3 次
#   RDZV_PORT=29400   bash scripts/tools/launch_multinode.sh ...   # 覆盖 rendezvous 端口
#   JOB_ID=my_job_123 bash scripts/tools/launch_multinode.sh ...   # 自定义 job ID
#
# JoyBuilder 会自动注入以下环境变量：
#   PET_NNODES          - 总节点数
#   PET_NODE_RANK       - 当前节点 rank（0 = master）
#   PET_MASTER_ADDR     - master 节点 IP 地址
#   PET_MASTER_PORT     - master 节点端口
#   PET_NPROC_PER_NODE  - 每节点进程数（通常等于 GPU 数）
#   CHOGORI_VGPU        - 当前节点 GPU 数（备用）

set -euo pipefail

# ─────────────────────────────────────────────
# 0. 解析训练脚本参数
# ─────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "用法: bash $0 <train_script.py> [其他参数]"
    echo "例:   bash $0 scripts/train/pretrain_qwen3_30b_2node16gpu.py"
    exit 1
fi

TRAIN_SCRIPT="$1"
shift  # 剩余 "$@" 传给 Python 脚本

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[error] 训练脚本不存在: ${TRAIN_SCRIPT}"
    exit 1
fi

# ─────────────────────────────────────────────
# 1. 读取 JoyBuilder 平台环境变量
# ─────────────────────────────────────────────
NNODES="${PET_NNODES:-1}"
NODE_RANK="${PET_NODE_RANK:-0}"
MASTER_ADDR="${PET_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"           # 优先使用平台注入的端口
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${CHOGORI_VGPU:-${NPROC_PER_NODE:-8}}}"  # 优先使用平台注入的 GPU 数

echo "[launch] nnodes=${NNODES}  node_rank=${NODE_RANK}  master=${MASTER_ADDR}:${MASTER_PORT}"

# ─────────────────────────────────────────────
# 2. 容错配置
#
#    ELASTIC=1（默认）：c10d rendezvous + 自动重启
#      - Worker 失败检测（心跳 + monitor_interval 轮询）
#      - 全员自动重启（max_restarts 次机会）
#      - 环境变量 TORCHELASTIC_RESTART_COUNT 供训练脚本
#        判断是否在重启中（> 0 → 从 checkpoint 恢复）
#      - 与 torchft（per-step quorum）互补：torchrun 负责
#        进程级 cold restart，torchft 负责 replica 级热替换。
#
#    ELASTIC=0：static rendezvous，无自动重启
#      - 启动更快（无心跳/rendezvous 开销）
#      - 适合调试、短任务、或外部容错（如 JoyBuilder 重调度）
# ─────────────────────────────────────────────
ELASTIC="${ELASTIC:-1}"
DEBUG="${DEBUG:-0}"

if [[ "$ELASTIC" == "1" ]]; then
    MAX_RESTARTS="${MAX_RESTARTS:-3}"

    # Rendezvous 端口（独立于 MASTER_PORT，避免 c10d store 和 NCCL init 端口冲突）
    RDZV_PORT="${RDZV_PORT:-29400}"

    # Job ID：唯一标识本次训练任务，防止多 job 混用同一 rendezvous
    # 从 MASTER_ADDR 提取 job 名（所有节点一致），不能用 date（各节点启动时间不同 → ID 不同 → 组网失败）
    if [[ -z "${JOB_ID:-}" ]]; then
        SCRIPT_BASENAME=$(basename "${TRAIN_SCRIPT}" .py)
        JOB_ID="${SCRIPT_BASENAME}_${MASTER_ADDR%%:*}"
    fi

    # Rendezvous 超时配置（通过 --rdzv-conf 传递给 c10d 后端）
    RDZV_JOIN_TIMEOUT="${RDZV_JOIN_TIMEOUT:-900}"
    RDZV_LAST_CALL_TIMEOUT="${RDZV_LAST_CALL_TIMEOUT:-30}"
    RDZV_KEEP_ALIVE_INTERVAL="${RDZV_KEEP_ALIVE_INTERVAL:-5}"
    RDZV_CONF="join_timeout=${RDZV_JOIN_TIMEOUT},last_call_timeout=${RDZV_LAST_CALL_TIMEOUT},keep_alive_interval=${RDZV_KEEP_ALIVE_INTERVAL}"

    MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

    echo "[launch] 容错: ELASTIC=1, max_restarts=${MAX_RESTARTS}, rdzv_backend=c10d, rdzv_port=${RDZV_PORT}, job_id=${JOB_ID}"
else
    echo "[launch] 容错: ELASTIC=0 (禁用), rdzv_backend=static"
fi

# ─────────────────────────────────────────────
# 3. InfiniBand GID 索引（RoCE v2）
#    运维确认值：GID_INDEX=1
#    show_gids 自动检测作为保险，回退值改为运维确认的 1
# ─────────────────────────────────────────────
detect_gid_index() {
    if command -v show_gids &>/dev/null; then
        local output line ipv4 idx
        output=$(show_gids 2>/dev/null | grep -i "v2")
        while IFS= read -r line; do
            ipv4=$(echo "$line" | awk '{print $5}')
            if [[ -n "$ipv4" && \
                  "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && \
                  "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
                idx=$(echo "$line" | awk '{print $3}')
                if [[ -n "$idx" && "$idx" =~ ^[0-9]+$ ]]; then
                    echo "$idx"
                    return
                fi
            fi
        done <<< "$output"
    fi
    echo "1"  # 运维确认回退值（原为 3）
}

NCCL_IB_GID_INDEX=$(detect_gid_index)
echo "[launch] NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"

# ─────────────────────────────────────────────
# 4. NCCL 环境变量（基于 JoyBuilder 运维文档 + Megatron 参考配置）
# ─────────────────────────────────────────────

# ── IB 核心配置 ──────────────────────────────
# 运维指定：8块 GDR 专用 HCA（每 GPU 对应一块，充分利用多 HCA 带宽）
export NCCL_IB_HCA=mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"   # 运维确认值：1
export NCCL_IB_TC=136                              # QoS 流量分类（与交换机配置匹配）
export NCCL_IB_SL=0                                # 服务级别
export NCCL_IB_TIMEOUT=22                          # IB 超时（2^22 × 4.096μs ≈ 17s）
export NCCL_IB_RETRY_CNT=7                         # 重试次数
export NCCL_IB_QPS_PER_CONNECTION=8                # 每连接 8 个 QP，提升多 HCA 并发带宽
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"     # 诊断开关：设为 1 强制走 TCP

# ── 多 HCA 路由 ──────────────────────────────
export NCCL_CROSS_NIC=1                            # 允许跨 NIC 路由（8 HCA 场景必须）
export NCCL_PXN_DISABLE=0                          # 启用 PXN（GPUDirect RDMA via NVLink+IB）
export NCCL_NET_GDR_LEVEL=2                        # GPU Direct RDMA 读+写

# ── 超时配置 ─────────────────────────────────
# NCCL_TIMEOUT：NCCL 层集合通信超时（30 min），上次训练 4 分钟超时的根因
export NCCL_TIMEOUT=1800
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC：PyTorch watchdog 心跳超时
# 容错模式下从 1h 缩短到 30min，让 watchdog 更快触发 worker 退出 → torchrun 检测失败 → 全员重启
# 非容错模式保持 1h 默认值，减少误杀
if [[ "$ELASTIC" == "1" ]]; then
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
else
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
fi

# ── Socket/TCP 回退配置 ───────────────────────
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_NTHREADS=8                      # socket 线程数（提升 TCP 回退性能）

# ── 其他 ─────────────────────────────────────
export NCCL_NVLS_ENABLE=0                          # 禁用 NVLS（RoCE 环境不可用）
# NCCL_DEBUG: DEBUG=1 → INFO（打印连接建立+路由），DEBUG=0 → WARN（只报错误）
if [[ "$DEBUG" == "1" ]]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
else
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# 存储分两类（按数据特性，不可混）：
#   LOCAL_CACHE_ROOT（/dev/shm 节点本地 tmpfs）：Triton/inductor 编译缓存 + TMPDIR
#     （MosaicML filelock）+ HF。这些要「本地文件系统原子性 + 每节点独立」——放 NFS
#     会因写读传播延迟触发 .so/.source 找不到的竞态崩溃（实测 rank3/rank7 崩在此）。
#     可重建、几百 MB，tmpfs 装得下；submit 回收即销无所谓。
#   训练输出（checkpoint/wandb/torchrun 日志）由各 recipe 自行配置 DUMP_FOLDER /
#     TORCHRUN_LOG_DIR，指向 NFS 持久盘。launch 脚本不管输出路径。
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/dev/shm/refmoe_pretrain_cache}"

# ── 调试模式（CUDA/分布式日志）──────────────────
# DEBUG 变量在上方与 ELASTIC 一起声明（NCCL_DEBUG 条件需要提前引用）。
# ⚠ TORCH_DISTRIBUTED_DEBUG 绝不能用 DETAIL——ProcessGroupWrapper 包裹 torchtitan
#   DataLoader 的 fake backend → 启动即崩（见 Known Issues）。

if [[ "$DEBUG" == "1" ]]; then
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
    export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
    echo "[launch] 调试模式: DEBUG=1 (CUDA_LAUNCH_BLOCKING=1, DISTRIBUTED_DEBUG=INFO, CPP_STACKTRACES=1)"
else
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
    export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
fi

# ── CUDA 配置 ─────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=32              # MoE 多 stream 并发（Megatron 推荐值）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 缓存重定向（shell 层，torchrun/torchelastic/NCCL/MosaicML 全继承）──────
# GPU 节点 /tmp 在 ~50G root overlay 上：Triton/inductor 编译产物、HF 下载、
# MosaicML shard filelock（streaming 用 tempfile.gettempdir() 定位）、torchelastic
# error_file 都默认写 /tmp → [Errno 28] No space left on device 撑爆。
# 重定向到 LOCAL_CACHE_ROOT（/dev/shm 节点本地 tmpfs）——这些缓存要本地原子性 +
# 每节点独立，绝不能放 NFS（NFS 写读传播延迟会让 .so/.source 找不到而崩）。
# shell 层 export 比训练脚本 Python 顶层更早生效——torchrun 本身和 rendezvous
# 在 import 训练脚本前就用 /tmp，必须在这里就改掉。
export TMPDIR="${LOCAL_CACHE_ROOT}/tmp"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_ROOT}/inductor"
export TRITON_CACHE_DIR="${LOCAL_CACHE_ROOT}/triton"
export HF_HOME="${LOCAL_CACHE_ROOT}/hf"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"
echo "[launch] 本地缓存（编译/filelock/HF）→ ${LOCAL_CACHE_ROOT}（/dev/shm 节点本地，避 NFS 竞态）"

# PyTorch 2.11 回归（pytorch#179597）：DeviceMesh 在 2.11 改为 opaque 类型后，flattened
# mesh（HSDP dp_replicate×dp_shard + EP）的 root↔child 双向引用形成循环，FxGraphCache
# 用 fast-mode pickle 序列化 cache key 时报 "can't pickle cyclic objects" → 每次编译刷屏
# 8×warning + cache miss。反正 pickle 失败=本来就没缓存，直接关掉 FX graph 磁盘缓存：
# 消除 log spam + 省去无用 pickle 尝试，缓存状态与现在等价（都没缓存）。
# 注：只关 FX graph 缓存，不影响 Triton kernel 缓存（TRITON_CACHE_DIR 仍工作）。
# 根治需 cherry-pick pytorch#180422 或升级到含该 fix 的版本。
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"

# 清理上次任务残留（submit 模式无法手动 ssh 清理，必须自动化）：
# /tmp 里历史 Triton/MosaicML filelock/torchelastic 垃圾 + 本次 TMPDIR 旧 filelock。
rm -rf /tmp/torchinductor_* /tmp/triton_* /tmp/*_cache_filelock \
       /tmp/torchelastic_* 2>/dev/null || true
rm -rf "${TMPDIR:?}"/*_cache_filelock 2>/dev/null || true

echo "[launch] NCCL_IB_HCA=${NCCL_IB_HCA}"

# ─────────────────────────────────────────────
# 5. 预检查（各节点均执行）
# ─────────────────────────────────────────────

# 5a. 清理 MosaicML StreamingDataset 残留共享内存
#     上次训练异常退出时 /dev/shm/ 中的 segment 不会自动清理，
#     导致下次启动 FileExistsError 或 barrier 超时（跨节点尤甚）。
shm_stale=$(find /dev/shm -maxdepth 1 -name '*_locals' -o -name '*_cache_usage' 2>/dev/null | head -20)
if [[ -n "$shm_stale" ]]; then
    echo "[clean] 清理 MosaicML 残留共享内存:"
    echo "$shm_stale" | head -5
    find /dev/shm -maxdepth 1 \( -name '*_locals' -o -name '*_cache_usage' \) -delete 2>/dev/null || true
    echo "[clean] 已清理"
else
    echo "[check] /dev/shm 无残留共享内存"
fi

# 5b. 清理残留 torchrun 进程（重启场景下尤其重要）
ZOMBIE_PROCS=$(ps aux | grep -E '[t]orchrun|[t]orch.distributed.run' | grep -v $$ || true)
if [[ -n "$ZOMBIE_PROCS" ]]; then
    echo "[clean] 发现残留 torchrun 进程:"
    echo "$ZOMBIE_PROCS" | head -5
    pkill -f 'torchrun|torch.distributed.run' 2>/dev/null || true
    sleep 2
    echo "[clean] 已清理"
else
    echo "[check] 无残留 torchrun 进程"
fi

# 5c. IB 端口状态检查
if command -v ibstat &>/dev/null; then
    if ! ibstat 2>/dev/null | grep -q "State: Active"; then
        echo "[warn]  未检测到 Active 状态的 IB 端口，跨节点通信可能降速或失败"
    else
        echo "[check] IB 端口状态正常"
    fi
fi

# 5d. Worker 节点检查 master 可达性
if [[ "$NODE_RANK" != "0" ]]; then
    if command -v nc &>/dev/null; then
        if nc -z -w5 "${MASTER_ADDR}" "${MASTER_PORT}" 2>/dev/null; then
            echo "[check] master ${MASTER_ADDR}:${MASTER_PORT} 可达"
        else
            echo "[warn]  master ${MASTER_ADDR}:${MASTER_PORT} 暂不可达（可能还未就绪，torchrun 会重试）"
        fi
    fi
fi

# 5e. Master 节点检查端口是否被占用
if [[ "$NODE_RANK" == "0" ]]; then
    CHECK_PORTS=("${MASTER_PORT}")
    [[ "$ELASTIC" == "1" ]] && CHECK_PORTS+=("${RDZV_PORT}")
    for check_port in "${CHECK_PORTS[@]}"; do
        if command -v ss &>/dev/null; then
            if ss -tlnp 2>/dev/null | grep -q ":${check_port} "; then
                echo "[error] 端口 ${check_port} 已被占用，请修改后重试"
                exit 1
            fi
        fi
    done
    if [[ "$ELASTIC" == "1" ]]; then
        echo "[check] 端口 ${MASTER_PORT}（master）和 ${RDZV_PORT}（rendezvous）可用"
    else
        echo "[check] 端口 ${MASTER_PORT}（master）可用"
    fi
fi

# ─────────────────────────────────────────────
# 6. 启动 torchrun
# ─────────────────────────────────────────────
# 日志策略：
#   --local-ranks-filter=0 → 只让 rank0 刷屏幕，避免 16 rank 刷屏（始终启用）。
#   --tee=3 + --log-dir    → 全 rank stdout+stderr 落盘 NFS（仅 DEBUG=1 启用）。
#     生产模式不落盘：省 NFS I/O + 避免磁盘占用累积；出问题时 DEBUG=1 重跑即可。
#     落盘目录必须在 NFS（持久共享盘），不能落 /dev/shm 或 $TMPDIR（节点本地 tmpfs，
#     job 结束即清空，崩溃后捞不到真凶栈）。
if [[ "$DEBUG" == "1" ]]; then
    if [[ -z "${TORCHRUN_LOG_DIR:-}" ]]; then
        echo "[error] DEBUG=1 需要设置 TORCHRUN_LOG_DIR（NFS 持久盘路径）用于日志落盘"
        exit 1
    fi
    LOG_DIR="${TORCHRUN_LOG_DIR}"
    mkdir -p "${LOG_DIR}"
    echo "[launch] torchrun 日志落盘（DEBUG=1，全 rank stdout+stderr）: ${LOG_DIR}"
    LOG_ARGS="--local-ranks-filter=0 --tee=3 --log-dir=${LOG_DIR}"
else
    LOG_ARGS="--local-ranks-filter=0"
fi

echo "[launch] 启动训练: ${TRAIN_SCRIPT}"

if [[ "$ELASTIC" == "1" ]]; then
    echo "[launch] torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE}"
    echo "[launch]   --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${RDZV_PORT}"
    echo "[launch]   --rdzv-id=${JOB_ID} --max-restarts=${MAX_RESTARTS}"
    echo "[launch]   --rdzv-conf=${RDZV_CONF}"
    echo "[launch]   --monitor-interval=${MONITOR_INTERVAL}"

    exec torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
        --rdzv-id="${JOB_ID}" \
        --rdzv-conf="${RDZV_CONF}" \
        --max-restarts="${MAX_RESTARTS}" \
        --monitor-interval="${MONITOR_INTERVAL}" \
        --role=rank \
        ${LOG_ARGS} \
        "${TRAIN_SCRIPT}" \
        "$@"
else
    echo "[launch] torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE}"
    echo "[launch]   --rdzv-backend=static --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

    exec torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv-backend=static \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        ${LOG_ARGS} \
        "${TRAIN_SCRIPT}" \
        "$@"
fi
