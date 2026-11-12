# 仓库背景与 FADA 启动命令交接

本文介绍仓库背景、常用入口和当前使用的 checkpoint，不包含具体任务的执行进度。

## 环境与定位

- 本地：`/Users/chengyuxuan/ArtiIntComVis/TTT`
- 服务器：`/ssd1/cyx/TTT`
- 仓库是基于 UniLab 的 ICE-Cal 工程，包含 G1 行走、FADA Planner–IDM、在线适应及可视化。
- 配置使用 Hydra；切换后端必须通过 `task=<任务>/<后端>`，不能只改 `training.sim_backend`。
- Python 命令统一使用 `uv run`。
- 先读根目录 `AGENTS.md`、`note/README.md`；服务器运行规范见 `docs/runbooks/server-training.md`。

主要入口：

- FADA 训练：`scripts/train_distill.py`
- Oracle 路线配置：`conf/distill/oracle_fada.yaml`
- 录制 / 本地查看：`scripts/play_fada_online.py`
- 浏览器查看：`scripts/play_fada_viser.py`
- 播放配置：`conf/distill/config.yaml`
- 录制预设：`conf/distill/recording_profile/`
- 播放会话：`src/unilab/visualization/playback_distill_sessions.py`

## 当前使用的 checkpoint

服务器路径：

```text
/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt
```

这是已完成 31 次迭代的 **Oracle 路线 FADA 学生模型**：

- obs=66，action=29，command=3。
- history=30，prediction horizon=6。
- 已使用 MuJoCo、CUDA 播放。
- 仿真步长 `1/150 s`，控制步长 `0.02 s`。

不要与 official 路线的 64/29/7 checkpoint 混用，也不要把学生 checkpoint 当成训练用 Oracle 教师。

上述路径指交接时使用的模型，不代表自动选择仓库中最新或最优的模型。

## 服务器公共准备

以下命令在同一个 Bash 会话中按需使用，不要同时启动所有任务。日志和视频名称若已存在，应更换版本号，避免覆盖。

```bash
cd /ssd1/cyx/TTT

export LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1
export MUJOCO_GL=egl
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

CKPT="$PWD/logs/fada/oracle/20261111_102618/student_iter_0031.pt"

PLAY=(
  task=g1_walk_flat/mujoco
  training.device=cuda
  training.play_checkpoint_path="$CKPT"
  interactive.action_mode=policy
  env.runtime_payload.enabled=false
)

mkdir -p logs/fada_online
```

### 1. 内置八字预设：固定模型录制五分钟

```bash
RUN=v1_g8_fixed_5m

nohup uv run --frozen --no-sync python scripts/play_fada_online.py \
  "${PLAY[@]}" \
  recording_profile=figure_eight \
  online_adaptation.enabled=false \
  video_recording.enabled=true \
  video_recording.duration_s=300 \
  video_recording.progress_interval_s=1.0 \
  video_recording.output_path="$PWD/logs/fada_online/${RUN}.mp4" \
  > "logs/fada_online/${RUN}.log" 2>&1 < /dev/null &

echo "PID=$!  日志=logs/fada_online/${RUN}.log"
```

该命令使用仓库内置的 `figure_eight` 预设，不表示其已满足特定轨迹精度或多圈稳定性要求。

### 2. FADA 在线适应录制五分钟

```bash
RUN=v2_g8_ttt_5m

nohup uv run --frozen --no-sync python scripts/play_fada_online.py \
  "${PLAY[@]}" \
  recording_profile=figure_eight \
  online_adaptation.enabled=true \
  online_adaptation.mode=fada_ttt \
  online_adaptation.sigma=0 \
  video_recording.enabled=true \
  video_recording.duration_s=300 \
  video_recording.output_path="$PWD/logs/fada_online/${RUN}.mp4" \
  > "logs/fada_online/${RUN}.log" 2>&1 < /dev/null &

echo "PID=$!  日志=logs/fada_online/${RUN}.log"
```

`fada_ttt` 必须显式设置 `sigma=0`；默认在线模式并不是 `fada_ttt`。这是一条显式启用在线更新的命令，不是固定模型播放。

### 3. 浏览器可视化

```bash
uv run --frozen --no-sync python scripts/play_fada_viser.py \
  "${PLAY[@]}" \
  recording_profile=figure_eight \
  online_adaptation.enabled=false
```

浏览器地址和端口以启动输出为准；远程访问需要对应 SSH 端口转发。

### 4. Oracle 路线训练入口

这是训练模板，不是上述 checkpoint 原训练命令的完整复现。需要先设置真实的 Oracle 教师路径；不得用上面的学生 checkpoint 代替。

```bash
uv run --frozen --no-sync python scripts/train_distill.py \
  --config-name oracle_fada \
  training.device=cuda \
  teacher.checkpoint_path="${ORACLE_CKPT:?请先设置Oracle教师checkpoint绝对路径}" \
  training.fada.checkpoint_path="$PWD/logs/fada/oracle/v1_train/student.pt" \
  training.fada.async_artifact_dir="$PWD/logs/fada/oracle/v1_train/source_batches"
```

## 操作偏好与边界

- 日志、视频用 `v1_描述简写`、`v2_描述简写` 命名，不用时间戳；避免覆盖已有结果。
- 已有服务器 SSH 曾位于本地 tmux pane `0:1.1`。该编号不是稳定配置：先确认会话，再用 `tmux capture-pane` / `send-keys` 操作；不要使用 GUI 或 cua-driver。
- 不擅自训练、部署、新建分支或修改依赖。
- 用户偏好小范围修改、最必要验证、直接报告实际结果。
- 启动入口和配置已在交接时核对；生成本文没有启动训练或仿真，本文不构成在线适应效果或训练复现的验证报告。
