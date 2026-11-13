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

## S 形绕柱：三根细管与五柱历史版本

`recording_profile=pole_slalom` 现在默认使用 **3 根细管**，中心距仍为 1 米。
管外径先按 4 厘米、高 80 厘米、质量 0.2 千克、摩擦系数 0.6 建模；外径和质量都是
可调的仿真占位值，不是用户实物的测量值。相邻管面净空为 96 厘米，柱心位于
`(1.5, 0.5)`、`(2.5, 0.5)`、`(3.5, 0.5)`；机器人从 `(0, 0)` 朝 +X 起步。
柱列长度由 4 米缩短为 **2 米**，基座参考路线纵向由 7 米缩短为 **5 米**，横向仍为
0–1 米。建议按约 **8×4 米**净空区初选场地，包含身体和粗略外围余量，不是真机安全认证。
成功三柱录像的基座中心实测范围约 **5.19×1.16 米**；身体占地估计约5.60×1.77米，完整说明见 [三根细管 v3 文档](fada-slalom-3p-v3.md)。

名义指令由 `src/unilab/visualization/slalom.py` 的圆缓 S 形程序生成，名义前进速度
0.2 m/s，控制周期 0.02 秒、每段保持 0.8 秒。三柱名义路线行走约 36.8 秒；
三柱已调优表行走 54.4 秒，再观察 2 秒；五柱原始几何表为 52.8 秒。

### 五柱历史成功版本（保留，不覆盖）

已跑通的五柱 v2 使用单独保存的固定指令表，不是原始几何表。其原柱径为15厘米、质量为1千克。
调整包括起步直行多保持
1.6 秒、针对实际左右转向响应调整持转和回正时机、复用有效的两柱穿绕单元，以及
离开第四根后才补第五根的左弯。没有调高前进速度、没有使用在线位姿反馈、没有训练策略。
新表共 100 段、每段 0.8 秒，行走 80 秒后给零速度指令，再观察 2 秒。

| 成功版本的服务器文件 | 路径 |
|---|---|
| 最终指令表与复跑轨迹 | `/ssd1/cyx/TTT/logs/fada_online/v2_g1_slalom_timing/v2_best_5p_1m.npz` |
| 成功录像 | `/ssd1/cyx/TTT/logs/fada_online/v2_g1_slalom_timing/v2_best_5p_1m.mp4` |
| 任务结果 | `/ssd1/cyx/TTT/logs/fada_online/v2_g1_slalom_timing/v2_best_5p_1m.json` |
| 录制日志 | `/ssd1/cyx/TTT/logs/fada_online/v2_g1_slalom_timing/v2_best_5p_1m.log` |

同一 checkpoint、相同随机种子 1 下，无视频试跑和录制复跑均为 `success=true`、
`passed_poles=5`，全部柱子未倾倒、未超过 0.1 米位移。第二根有 1 次接触采样，
因此是“不碰倒且不明显推移”，不是零接触。五根通过时基座至柱心的横向距离依次约为
36.5、54.1、35.6、55.7、54.9 厘米；这不是机器人外壳到柱面的距离。
该结果仅验证当前固定场景和种子，未证明换种子、柱径、柱距或真机后仍可复现。

### 三柱细管 v3 的启动方式（已实测通过）

服务器固定指令表为 `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin/input_commands.npz`。
它复用五柱成功表里通过前三根的有效前缀，并重新连接出场弯和直行，不是播到第三根就截断。
输入共 68 段、行走 54.4 秒；完整录像实际56.4秒，报告为3/3通过、无倾倒、无移位、未采样到接触。
详细文件路径、checkpoint、带版本号和防覆盖的启动命令见 [独立复现文档](fada-slalom-3p-v3.md)。

服务器已经支持 `pole_count=3`。复现时仍必须显式加载固定指令表；直接运行普通入口只会生成
未调优的名义表，仅修改文件名也不会切换指令。下述进程内加载方式不改仓库源代码。

```bash
cd /ssd1/cyx/TTT
mkdir -p logs/fada_online
OUTPUT="$PWD/logs/fada_online/v3_g1_slalom_3p_thin"
export SLALOM_COMMAND_TABLE="$PWD/logs/fada_online/v3_g1_slalom_3p_thin/input_commands.npz"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --no-sync python -c '
import runpy
import os
import numpy as np
import torch
from unilab.visualization.slalom import SlalomCommandProgram

np.random.seed(1)
torch.manual_seed(1)
torch.set_num_threads(1)
original = SlalomCommandProgram.from_config.__func__

def replay(cls, *args, **kwargs):
    program = original(cls, *args, **kwargs)
    path = os.environ["SLALOM_COMMAND_TABLE"]
    with np.load(path, allow_pickle=False) as data:
        if (
            program.segment_steps != int(data["segment_steps"])
            or not np.isclose(program.ctrl_dt, float(data["ctrl_dt"]))
            or not np.allclose(program.pole_centers, data["pole_centers"])
            or not np.allclose(program.reference_points, data["reference_points"])
        ):
            raise ValueError("Current course does not match the saved command table")
        table = data["commands"].copy()
        if table.ndim != 2 or table.shape[1] != 3 or not np.isfinite(table).all():
            raise ValueError("Invalid command table")
        program.commands = table
        program.commands.setflags(write=False)
        program.total_control_steps = len(table) * program.segment_steps
        program.duration_s = program.total_control_steps * program.ctrl_dt
    return program

SlalomCommandProgram.from_config = classmethod(replay)
runpy.run_path("scripts/record_fada_slalom.py", run_name="__main__")
' \
  task=g1_walk_flat/mujoco \
  recording_profile=pole_slalom \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt \
  video_recording.output_path="${OUTPUT}.mp4" \
  > "${OUTPUT}.log" 2>&1 < /dev/null &

printf 'PID：%s\n日志：%s.log\n视频：%s.mp4\n' "$!" "$OUTPUT" "$OUTPUT" >&2
```

历史五柱复现：将 `SLALOM_COMMAND_TABLE` 改为上表的五柱 v2 NPZ、`OUTPUT` 换一个新版本号，
并在命令中显式添加 `demo.slalom.pole_count=5 demo.slalom.pole_radius_m=0.075 demo.slalom.pole_mass_kg=1.0`，
以免新版三柱默认值改变旧场景。4、5 柱能力仍保留，Gerono v14 未改动。

预期输出为同名前缀的 `.mp4` 视频、`.npz` 实际轨迹／指令／柱子姿态及 `.json` 任务报告。
报告的 `success=true` 才表示实际按顺序从配置指定的全部柱子交替两侧通过、到达终点且稳定观察期结束；
指令播放完毕、视频写完或没摔倒都不单独算成功。

柱子倾斜超过 30° 持续 0.2 秒、水平移动超过 0.1 米、错误一侧通过、遗漏柱子或机器人终止，
都会记录失败。接触单独统计，不把短暂接触等同倾倒；接触是物理步末采样，不能据此证明
所有子步都绝对零接触。失败会保留已执行的轨迹与结果，不自动 reset 再试。

机器人 29 个动作和机器人专属观测维度保持不变，道具自由关节只存在于完整物理状态中。
道具场景在私有临时目录生成，机器人原始 XML 指纹和 checkpoint 校验保留，正常关闭会清理。
当前只支持 MuJoCo 平地名义场景，不叠加 terrain、scene fragments、负重或在线适应。

本地检查已验证真实柱体碰撞／倾倒／reset、机器人契约兼容和回放工厂接入；
服务器已完成五柱 v2 实测及复跑，以及当前三柱细管 v3 的单趟成功录像。
三柱实测与文件信息以 [三根细管 v3 文档](fada-slalom-3p-v3.md) 为准，不把五柱数据混作三柱结果。

## 操作偏好与边界

- 日志、视频用 `v1_描述简写`、`v2_描述简写` 命名，不用时间戳；避免覆盖已有结果。
- 已有服务器 SSH 曾位于本地 tmux pane `0:1.1`。该编号不是稳定配置：先确认会话，再用 `tmux capture-pane` / `send-keys` 操作；不要使用 GUI 或 cua-driver。
- 不擅自训练、部署、新建分支或修改依赖。
- 用户偏好小范围修改、最必要验证、直接报告实际结果。
- 本地模块／物理检查和配置核对不等于指定 checkpoint 已完成任务；各 Demo 的实际运行证据以上文对应章节为准。
