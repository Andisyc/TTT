# 独立官方 FADA–Oracle 链路

此目录是单独的 Python 项目：有自己的 `pyproject.toml`、`uv.lock`、`.venv`、官方源码 checkout 和输出目录。运行时不导入 `unilab`，不调用 `scripts/train_distill.py` 或旧的 `scripts/train_fada_upstream.py`；两条现有 UniLab 链路保持原样。

## 模块边界

|模块|职责|
|---|---|
|`upstream.py`、`upstream_patches/`|取得固定官方版本，检查源码身份，应用可检查的接口补丁。|
|`oracle/`|独立 Torch Oracle 网络、原 checkpoint 与归一化器读取、推理及中间权重切换。|
|`environment/`|完整原始场景、观测、控制、命令、随机化及影子环境物理状态适配。|
|`integration.py`|组装官方环境工厂、影子进程、ReplayBuffer、Planner–IDM 和官方训练循环。|
|`cli.py`|参数入口；训练参数继续交给官方解析器。|

官方来源为 [LeCAR-Lab/FADA](https://github.com/LeCAR-Lab/FADA)，固定提交 `290b3a9c5c0ebe2e9851b61350617544be05e9bc`。MuJoCo Warp 固定提交为 `09ec1da65ac44bd68703f9507c9c9e8f22ae7c71`。私有源码位于 `.deps/FADA`，不要在该目录内直接修改源码。

接口补丁仅覆盖影子环境工厂注入、场景工厂扩展点、MuJoCo 代理视图的取值/写回，以及快照携带原场景的质量、摩擦、求解器 warm-start 和时钟。官方网络、loss、采样规则、优化步骤和训练循环没有另写一份。

## 安装

所有命令均从本目录执行，不能使用仓库根目录的虚拟环境。

```bash
cd /path/to/TTT/standalone/fada_oracle

# Linux + NVIDIA：正式并行后端。
uv sync --frozen --no-sources --extra warp
uv run --frozen --no-sync fada-oracle setup
```

本地 CPU 开发使用 `uv sync --frozen --no-sources --group dev`，随后执行同一条 `setup`。Python 固定为 3.12。`--no-sources` 必须保留：被固定的 mujoco-warp 自身声明了 nightly MuJoCo 源，本项目使用锁定的标准 MuJoCo 3.4.0，而不是 nightly 或 `mujoco-uni`。首次安装会下载独立依赖和官方源码。

`warp` 后端需要 Linux、CUDA 与适配的 NVIDIA 驱动；`classic` 只接受 `--device cpu --num-envs 1`。不会把 CUDA 故障静默降级到 CPU。不要照搬旧链路的 `PYTHONPATH`、`LD_PRELOAD` 或 `FADA_ROOT`。

## 输入契约

1. **原 Oracle checkpoint**：直接读取 `unilab_motion_quality_oracle_v1`，保留原 actor 和 privileged normalizer，不导出或伪造新权重。
2. **同一次 teacher 训练的中间 checkpoint**：官方默认的 suboptimal 配方会从最终权重同目录的 `model_*.pt` 中抽取历史 teacher。只有一个最终文件不满足该配方；入口会在创建环境前报错，不会自动关闭 suboptimal 训练。所有被选中的权重必须有一致的网络、观测、机器人和控制契约。
3. **完整原始场景目录**：传入原 `scene_flat.xml`；保留 include、所有 XML、引用的 mesh 和传感器。输入 XML 指纹必须等于 checkpoint 的 `asset_sha256`。指纹按原格式覆盖整个目录内的 XML，包括未引用的历史 XML；不能随意删掉它们。被引用的空 mesh 不是有效资产。

资产只在启动时检查、复制到本次输出目录。复制品将 position actuator 转换为 motor，保留原力矩限制，避免 XML position controller 与官方 PD 重复控制；不会改写输入资产。运行时没有 XML 解析或 UniLab backend 依赖。

基准质量、摩擦和质心数据在初始化时按对应物理缓冲区的 dtype/device 保存独立快照：标准 MuJoCo 保留 float64，Warp 使用 float32；随机化路径不再临时转换精度。

整局与局部 reset 共用官方 root/DOF 写回、接触历史清理和派生观测刷新顺序；完成后清除刷新标志，空集合不刷新，整局 reset 不额外执行物理步。根状态中的角速度按 MuJoCo 本地 qvel 保存，不再多做一次旋转。Warp 保留其原生 CUDA 流，Torch 通过官方 `stream_to_torch` 接入，并先等待旧 Torch 流完成；不把 Warp 切换到外部流。模型字段扩展后重新捕获 step 和 forward 图，reset 与影子快照恢复重放 forward 图，仍完整计算约束但不推进物理时间。

标准 MuJoCo 的静态参数重算使用初始化时分配的独立 `MjData`，不把活动仿真状态传给会重置 qpos 的 `mj_setConst`，因此随机化和快照恢复不会覆盖刚写入的根状态。

Teacher 使用原 98 维观测和 205 维特权信息，学生使用官方 64 维 compact observation 和 7 维 command。关节顺序、原 PD 参数、150 Hz 仿真 / 50 Hz 控制、世界坐标 torso upvector、原始弧度 gait phase、上一拍原始 action 和实际物理随机化数据由适配层提供。源任务启用但此适配层未支持的模式会明确拒绝，而不是填零冒充数据。Reward 继续使用官方配置，不声称与旧 UniLab reward 等价。

本次本地准备的完整输入位于 `.deps/oracle_source/g1/scene_flat.xml`，匹配当前 `model/mirror/model_50000.pt` 的 XML 指纹。该目录和 checkpoint 都是忽略的本地数据，不随 Git 自动同步。其空的可视 mesh 由机器人 XML 完全相同的本地原始资产补齐，原 XML 未改。

## 启动示例

以下是未来同步并准备服务器输入后的命令，本次没有在服务器上执行。`--scene-model-file` 必须指向已复制的完整原始场景；teacher 文件夹必须包含配套中间权重。

```bash
cd /ssd1/cyx/TTT/standalone/fada_oracle
set -o pipefail
RUN="outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs

env MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  uv run --frozen --no-sync fada-oracle train \
  --oracle-checkpoint /ssd1/cyx/TTT/logs/oracle/mirror_yaw025_50k_20261108-054124/model_50000.pt \
  --scene-model-file .deps/oracle_source/g1/scene_flat.xml \
  --output-dir "$RUN" \
  --backend warp \
  --device cuda:0 \
  --num-envs 64 \
  --batch-size 4096 \
  --replay-capacity 524288 \
  --warmup-train-steps 2000 \
  --dagger-iters 30 \
  --train-steps-per-iter 1000 \
  --wandb-mode disabled \
  2>&1 | tee "${RUN}.log"
STATUS=$?
echo "训练退出码：$STATUS" | tee -a "${RUN}.log"
```

不要提前创建 `$RUN` 本身；入口要求新的运行目录。缓存、场景副本、配置快照、训练日志、阶段权重和 `model_final.pt` 都保存在该目录，终端输出另存为 `${RUN}.log`。此入口只启动新 run，不提供旧 UniLab run 或其 checkpoint 的续训兼容层。官方学生 checkpoint 的读取和优化器恢复由官方实现负责。

## 验证范围

```bash
OMP_NUM_THREADS=1 uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync fada-oracle --help
```

本次只执行本地 CPU 离线验证：90 项针对性测试通过；真实 Oracle 非零输入推理对照、真实场景冷初始化、官方完整快照恢复、影子进程创建/退出、人工轨迹更新两个网络，以及重新加载归一化器和两个优化器后的下一次更新对照均通过。验证期间禁止 `mj_step`，不把人工轨迹更新当作真实环境训练效果。CUDA 并行运行和长时间训练稳定性不在本次已验证范围内。

本目录沿用仓库 Apache-2.0 许可证。官方 checkout 保留其原始许可证；`oracle/` 和 `environment/features.py` 标注所提取的 UniLab 数值实现来源。
