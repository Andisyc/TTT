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

## 五柱 S 形绕行：名义场景与单趟录制

`recording_profile=pole_slalom` 使用 5 根真实可碰撞、可倾倒的自由柱体，中心距 1 米，
直径 15 厘米、高 80 厘米、质量 1 千克、摩擦系数 0.6。柱心在起步坐标系中依次为
`(1.5, 0.5)` 至 `(5.5, 0.5)`；机器人从 `(0, 0)` 朝 +X 起步，左右交替穿绕。
参考线横向范围为 0–1 米，终点约 `(7, 0)`。四柱版本用 `demo.slalom.pole_count=4`。
这些道具参数是仿真初值，不代表已与真机现场柱子标定。

指令由 `src/unilab/visualization/slalom.py` 的圆缓 S 形程序生成，名义前进速度
0.2 m/s，控制周期 0.02 秒、每段保持 0.8 秒。当前五柱离散路线行走约 52.8 秒，
随后停止给行走指令，并留 2 秒检查柱子与任务状态；不会在终点循环回起点。
指令只依赖配置和控制步，不使用位姿反馈，不修改策略，也不影响 Gerono v14。

**必须用专用入口取得任务结果。** 普通 `play_fada_online.py` 可以播放该场景，但
不负责输出逐柱通过／倾倒判定。下面的输出必须使用新的版本号；已有 MP4、NPZ 或 JSON
不会被专用入口覆盖，shell 重定向的 `.log` 则需要由操作者避免复用同名文件。

```bash
cd /ssd1/cyx/TTT
mkdir -p logs/fada_online
OUTPUT="$PWD/logs/fada_online/v1_g1_slalom_5p_1m"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --no-sync python scripts/record_fada_slalom.py \
  task=g1_walk_flat/mujoco \
  recording_profile=pole_slalom \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt \
  video_recording.output_path="${OUTPUT}.mp4" \
  > "${OUTPUT}.log" 2>&1 < /dev/null &

printf 'PID：%s\n日志：%s.log\n视频：%s.mp4\n' "$!" "$OUTPUT" "$OUTPUT" >&2
```

预期输出为同名前缀的 `.mp4` 视频、`.npz` 实际轨迹／指令／柱子姿态及 `.json` 任务报告。
报告的 `success=true` 才表示实际按顺序从五柱交替两侧通过、到达终点且稳定观察期结束；
指令播放完毕、视频写完或没摔倒都不单独算成功。

柱子倾斜超过 30° 持续 0.2 秒、水平移动超过 0.1 米、错误一侧通过、遗漏柱子或机器人终止，
都会记录失败。接触单独统计，不把短暂接触等同倾倒；接触是物理步末采样，不能据此证明
所有子步都绝对零接触。失败会保留已执行的轨迹与结果，不自动 reset 再试。

机器人 29 个动作和机器人专属观测维度保持不变，道具自由关节只存在于完整物理状态中。
道具场景在私有临时目录生成，机器人原始 XML 指纹和 checkpoint 校验保留，正常关闭会清理。
当前只支持 MuJoCo 平地名义场景，不叠加 terrain、scene fragments、负重或在线适应。

本地已通过真实 G1 的柱体碰撞／倾倒／reset、机器人契约兼容和实际回放工厂接入检查，
并渲染确认五柱与参考路线可见。**指定 checkpoint 只在服务器，尚未实测策略能否穿绕一趟。**
先由用户更新服务器代码，再运行以上一次单趟录制；不要把几何可行或本地检查当作策略已成功。

## 操作偏好与边界

- 日志、视频用 `v1_描述简写`、`v2_描述简写` 命名，不用时间戳；避免覆盖已有结果。
- 已有服务器 SSH 曾位于本地 tmux pane `0:1.1`。该编号不是稳定配置：先确认会话，再用 `tmux capture-pane` / `send-keys` 操作；不要使用 GUI 或 cua-driver。
- 不擅自训练、部署、新建分支或修改依赖。
- 用户偏好小范围修改、最必要验证、直接报告实际结果。
- 本地模块／物理检查和配置核对不等于指定 checkpoint 已完成任务；各 Demo 的实际运行证据以上文对应章节为准。
