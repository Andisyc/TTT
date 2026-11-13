# G1 Gerono 八字 Demo：v14 复现与场地说明

## 当前结论

**v14 是当前保留用于展示的开环 Gerono 版本。** 已完成五分钟录像，用户观察到前五圈几乎闭环，约 3 分 30 秒后开始轻微偏移／右倾。用户接受当前展示效果，结束本轮精调，推进后续 Demo。它不是“无限圈零漂移”或“通过五分钟严格轨迹误差验收”的版本。

正式播放周期重复同一套指令，不逐圈另设指令、不按位置纠偏、不训练、不启用在线适应。仿真位姿曾用于离线评价和选择参数，但不参与最终指令生成。

## 1. 必须保留的文件

以下路径均在服务器，仓库根目录为 `/ssd1/cyx/TTT`。

| 用途 | 路径 |
|---|---|
| **最终 v14 速度指令表** | `/ssd1/cyx/TTT/logs/fada_online/v14_g8_return_trim/right_less05.npz` |
| **已完成的五分钟视频** | `/ssd1/cyx/TTT/logs/fada_online/v14_g8_return_trim_5min_matched.mp4` |
| 对应成功录制日志 | `/ssd1/cyx/TTT/logs/fada_online/v14_g8_return_trim_5min_matched.log` |
| 学生 checkpoint | `/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt` |
| 原始录制预设 | `conf/distill/recording_profile/figure_eight.yaml` |
| 录制入口 | `scripts/play_fada_online.py` |

NPZ 和 checkpoint 是复现输入，不是可以随意清理的临时日志。迁移服务器或重新克隆仓库时应单独保留，不要假定 Git 会携带日志目录中的这些文件。视频用于回顾展示效果，不能代替指令表。

**仅设置 `RUN_ID="v14"` 或使用 `recording_profile=figure_eight`，不会自动加载最终 v14 表。** YAML 内的全局倍率不包含所有局部修正；复现必须读取上述 NPZ。

## 2. 指令表与运行节拍

| 字段／参数 | 含义或当前值 |
|---|---|
| `commands` | `(52, 3)`，每行依次为机体系前进速度 `vx`、横向速度 `vy`、转向角速度 `wz`；单位 m/s、m/s、rad/s |
| `ctrl_dt` | 0.02 秒 |
| `segment_steps` | 每条指令保持 40 个控制步，即 0.8 秒 |
| 一整套八字周期 | 52 × 0.8 = **41.6 秒** |
| 300 秒录制 | 7 个完整周期，再加第 8 圈的一部分；不是每圈一套不同指令 |
| 保存表的前进速度范围 | 约 0.3193–0.3348 m/s；不要用标称的 0.3 m/s 覆盖它 |
| 横向速度 | 全部为 0 |
| 转向角速度范围 | 约 −0.7940 至 +0.7743 rad/s |
| `reference_points`、`reference_headings` | 目标参考轨迹和切线朝向，用于绘图／离线评估，不是补偿指令的理想积分 |
| `poses`、`local_poses`、`report_json` | 三圈调参试验记录，不是本次五分钟录像的同步位姿记录 |

该版本保留了 v10 的第一瓣局部进弯延后 **0.2 秒**，又在第 **47–48 段**合计少给 **0.5° 的右转指令量**。这是固定时间表的转向修正，不是运行时位置或航向反馈。最终数值以 NPZ 的 `commands` 为准，不建议手工逐项重建。

## 3. Gerono 尺寸与真机场地预选

| 项目 | 尺寸／说明 |
|---|---|
| 生成器标称尺寸 | **宽 4 米 × 高 2 米**，对应 `width_m=4`、`height_m=2` |
| 标称几何 | `x=2 sin(t), y=2 sin(t) cos(t)`；坐标系再按起步切线对齐、镜像到第一瓣左转方向 |
| 保存参考点的外接范围 | 起步坐标系 X、Y 均约 **−1.828 至 +1.828 米**，即约 **3.66 × 3.66 米** |
| 场地预选时的轨迹占地 | 按约 **3.7 × 3.7 米**考虑；参考点之间是弧段，采样点外接框不是绝对硬边界 |
| 建议先找的平整净空区 | **至少约 6 × 6 米**，在目标轨迹外各侧粗留约 1 米以上；人员活动区另算 |

**4×2 米描述的是 Gerono 自身的两条几何轴，不等于机器人初始前进方向的长宽。** 当前两瓣沿斜方向展开，所以起步坐标系的 X/Y 占地更接近正方形。改变机器人初始朝向会旋转这个外接框，但不会改变路线本身的尺寸。

上述均为基座中心的目标路线，不包含机器人身体／双脚外廓、实际漂移、起步落脚、意外失稳范围或紧急停机距离。6×6 米只是找场地时的初选建议，不是已验证的真机安全边界。地面、围障、人机间距和现场急停仍需单独确认；仿真指令也尚未证明可在真机获得相同路线。不要把柱子或场地边缘紧贴标称外接框布置。

## 4. 五分钟启动命令

在服务器 Bash 中执行。命令读取现有 v14 NPZ，无需上传新的 Python 代码；依赖现有仓库、uv 环境、checkpoint 和场景资源。

- 使用版本号加描述性简写命名输出，不附加时间戳。
- 下例新输出为 `v15_g8_v14_replay_5min`；这里的 `v15` 只是新录像编号，指令仍是 **v14**。
- 输出存在就拒绝覆盖；重复录制时递增 `RUN_ID`。
- 原录制第一次启动曾报 `asset_sha256` 不匹配。下述命令沿用成功录制的做法：复制现有场景到私有临时目录，验证与 checkpoint 相同的资源指纹，再显式指定该场景。**不跳过校验、不修改仓库场景，正常退出自动清理临时副本。**

```bash
cd /ssd1/cyx/TTT
mkdir -p logs/fada_online
RUN_ID="v15_g8_v14_replay_5min"
OUTPUT="$PWD/logs/fada_online/${RUN_ID}"

if [ -e "${OUTPUT}.mp4" ] || [ -e "${OUTPUT}.log" ]; then
  printf '输出已存在，请递增 RUN_ID 版本号：%s\n' "$OUTPUT" >&2
else
  nohup env \
    LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH="$PWD/src" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uv run --frozen --no-sync python -c '
import runpy
import sys
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import numpy as np
import torch
from unilab.visualization.demo_commands import FigureEightCommandProgram
from unilab.envs.locomotion.g1.fada_privileged import build_g1_fada_checkpoint_layout_identity

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(1)
original = FigureEightCommandProgram.from_config.__func__

def replay(cls, *args, **kwargs):
    program = original(cls, *args, **kwargs)
    path = "logs/fada_online/v14_g8_return_trim/right_less05.npz"
    with np.load(path, allow_pickle=False) as data:
        if (
            program.segment_steps != int(data["segment_steps"])
            or not np.isclose(program.ctrl_dt, float(data["ctrl_dt"]))
            or program.commands.shape != data["commands"].shape
            or not np.allclose(program.reference_points, data["reference_points"])
        ):
            raise ValueError("Current profile does not match the saved v14 commands")
        program.commands = data["commands"].copy()
        program.commands.setflags(write=False)
    return program

FigureEightCommandProgram.from_config = classmethod(replay)

with TemporaryDirectory(prefix="v14_g1_scene_") as temp:
    source = Path("src/unilab/assets/robots/g1").resolve()
    scene = copytree(source, Path(temp) / "g1") / "scene_flat.xml"
    digest = build_g1_fada_checkpoint_layout_identity(
        body_names=["probe"], actuated_joint_names=["probe"], model_file=scene,
    ).asset_sha256
    if digest != "6339734dada64e025b04d4d264cbca899b312705485178840ff1d9f414c28402":
        raise ValueError("The private scene copy does not match the checkpoint asset hash")
    print("V14_SCENE_MATCH", digest, flush=True)
    sys.argv.append("++training.fada.scene_model_file=" + str(scene))
    runpy.run_path(str(Path("scripts/play_fada_online.py").resolve()), run_name="__main__")
' \
    task=g1_walk_flat/mujoco \
    recording_profile=figure_eight \
    training.device=cuda \
    training.play_checkpoint_path=/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt \
    env.runtime_payload.enabled=false \
    online_adaptation.enabled=false \
    video_recording.enabled=true \
    video_recording.duration_s=300 \
    video_recording.progress_interval_s=1.0 \
    video_recording.output_path="${OUTPUT}.mp4" \
    > "${OUTPUT}.log" 2>&1 < /dev/null &

  printf 'PID：%s\n日志：%s.log\n视频：%s.mp4\n' "$!" "$OUTPUT" "$OUTPUT" >&2
fi
```

日志中出现 `Video saved: ... (9000 frames, 15000 control steps)` 才表示 300 秒录制和 MP4 编码完成。仅有 PID 不代表成功；出现 `asset_sha256` 等契约错误时不要跳过校验或修改 checkpoint。

需要查看录制进度时，在上面启动命令所在的同一个 Bash 中执行：

```bash
tail -f "${OUTPUT}.log" >&2
```

Ctrl-C 退出这个日志查看命令，不会停止已经通过 nohup 启动的录制进程。

## 5. 证据边界与后续使用

- 已有视频保存了 **9000 帧、15000 控制步**，对应五分钟；成功路径包含 `_matched`。
- 同目录不含 `_matched` 的 `v14_g8_return_trim_5min.log` 是第一次启动失败的日志，不是最终成功录像。
- 三圈实测：最大分段位置误差约 36.5 厘米、最大交叉点／圈末位置误差约 17.6 厘米，第三圈整体倾角约右倾 0.2°。这些数值不能外推为五分钟都满足相同门槛。
- 用户认可现有五分钟展示效果；约第五圈结束／第六圈开始（3:28–3:30）后可见轻微漂移。五分钟录像没有同步保存完整位姿序列，尚不能据此给出精确的长期定位误差。
- 后续从本版本继续时，优先保留原表另存版本；不要用未验证的 v11–v13 候选替换，不需继续追求无限圈零误差。
- Table 2 的冻结八字脚本 `scripts/run_table2_figure_eight_frozen.sh` 默认不绘制顶部状态栏，避免遮住机器人。`run.log` 和 `raw/` 仍照常记录。
