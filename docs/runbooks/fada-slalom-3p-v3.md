# G1 三根细管绕行 Demo：v3 复现与场地说明

## 当前结论

**v3 是当前保留的三根细管绕行展示版本，用户已观看录像并认可效果。** 对应实测报告为
`success=true`、`passed_poles=3`：依次从三根柱子交替两侧通过，到达终点，并完成 2 秒稳定观察。
三根柱子均未倾倒、未明显移位，接触采样均为 0。

这是固定场景、指定 checkpoint、随机种子 1 下的一次成功录制，不代表任意初始条件或真机都能复现。
接触传感器按物理步末采样；“未采样到接触”不等于严格证明所有子步都绝对没有瞬时接触。

运行时只播放固定速度指令，不读取位姿来纠偏，不训练、不启用在线适应。位置、姿态和碰撞数据仅用于
离线调试和任务结果评价。路线只走一趟，不在终点循环，不用每根柱子重新 reset。

## 1. 服务器文件与 checkpoint

服务器仓库根目录：`/ssd1/cyx/TTT`。

| 用途 | 服务器绝对路径 |
|---|---|
| **复现输入：固定速度指令表** | `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin/input_commands.npz` |
| **已认可的视频** | `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin.mp4` |
| 录制日志 | `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin.log` |
| 实测任务结果 | `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin.json` |
| 实际轨迹、执行指令、柱子姿态 | `/ssd1/cyx/TTT/logs/fada_online/v3_g1_slalom_3p_thin.npz` |
| **学生 checkpoint** | `/ssd1/cyx/TTT/logs/fada/oracle/20261111_102618/student_iter_0031.pt` |

checkpoint 是已完成 31 次迭代的原生 Oracle 路线 FADA 学生：obs=66、action=29、command=3，
history=30、prediction horizon=6，MuJoCo 仿真步长 `1/150 s`，控制步长 `0.02 s`。
不要替换成 Oracle 教师、official 路线的 64/29/7 学生或其他 checkpoint 后仍声称复现了本结果。

相关仓库入口：

- 专用录制与评分：`scripts/record_fada_slalom.py`。
- 场景及录像预设：`conf/distill/recording_profile/pole_slalom.yaml`。
- 名义指令和任务判定：`src/unilab/visualization/slalom.py`。
- 单趟运行与输出保存：`src/unilab/visualization/slalom_recording.py`。

**必须单独备份指令表和 checkpoint。** 它们位于日志目录，不应假定会随 Git 克隆、代码同步或仓库清理保留。
视频不能代替指令表；建议连同 JSON 结果、实际轨迹和日志一起保留。

## 2. 指令表：不要混淆两个 NPZ

| 文件 | `commands` 的含义 | 当前形状 |
|---|---|---:|
| `v3_g1_slalom_3p_thin/input_commands.npz` | 每段播放一次的固定输入表 | **68×3** |
| `v3_g1_slalom_3p_thin.npz` | 每个控制步实际发送的指令，包含重复保持与停止观察 | **2820×3** |

**启动必须读取第一行的 `input_commands.npz`，不能用第二行的实际轨迹文件替换。**

输入表每列依次为机体系前进速度 `vx`、横向速度 `vy`、转向角速度 `wz`，单位为 m/s、m/s、rad/s。

| 项目 | 当前值 |
|---|---:|
| `ctrl_dt` | 0.02 秒 |
| `segment_steps` | 40 个控制步 |
| 每段保持 | 0.8 秒 |
| 输入表总行走时长 | **54.4 秒** |
| 停止观察 | 2 秒 |
| 本次完整实测 | **56.4 秒，2820 个控制步** |
| 前进速度范围 | 约 0.1845–0.2083 m/s |
| 横向速度 | 全部为 0 |
| 转向角速度范围 | −0.50 至 +0.55 rad/s |

这张表复用五柱成功版本中通过前三根的有效前缀，再连接出场弯和直行，**不是播放到第三根就截断**。
它保留了起步直行、持转和回正时机的实际调优；直接调用名义几何生成器不会得到同一组指令。
输入 NPZ 中的 `report_json` 是准备阶段的来源说明；运行是否成功以外部实测 `.json` 为准。

## 3. 三根细管的尺寸与摆放

| 参数 | 本次仿真设置 |
|---|---|
| 数量 | **3 根** |
| 相邻柱中心距 | **1 米** |
| 柱子外径 | **4 厘米**，`pole_radius_m=0.02` |
| 柱面之间净空 | **96 厘米** |
| 第一根至第三根中心的距离 | **2 米** |
| 柱高 | 80 厘米 |
| 单根质量 | 0.2 千克 |
| 摩擦系数 | 0.6 |
| 柱心 XY | `(1.5, 0.5)`、`(2.5, 0.5)`、`(3.5, 0.5)`，单位米 |
| 初始基座 XY 与朝向 | `(0, 0)`，朝 +X |
| 通过顺序 | 第一根左侧、第二根右侧、第三根左侧 |
| 参考离场终点 | 约 `(5, 0)`；成功要求实际越过终点门 |

外径、质量和摩擦系数是仿真占位设置，不是用户实物塑料管的测量值；实物可能更细或更轻，应按实物调整。
柱子有真实碰撞和自由关节，可以移动、倾倒，不是焊死的装饰物，也没有用超重底座让它“碰不倒”。

## 4. 机器人实际走过的长宽与场地初选

以下实际范围来自本次成功录制的 `base_positions`，包含起步、穿绕、离场及停止观察：

| 范围 | 长×宽 | 依据 |
|---|---|---|
| **基座中心实际轨迹** | **约 5.19×1.16 米** | 全部 56.4 秒实测轨迹 |
| 身体占地估计 | 约 5.60×1.77 米 | 站立模型外接框随实测机身姿态旋转的估算 |
| 建议场地初选 | **约 8×4 米平整净空区** | 在估算身体占地之外粗留外围余量 |

长方向沿柱列，宽方向垂直柱列。基座 X 范围约 0–5.191 米，Y 范围约 −0.182–0.983 米。
柱列本身只有 2 米，但需要加上起步入弯和出场空间，不能按 2 米长找场地。

身体占地不是完整摆臂、迈步扫掠的实测结果；8×4 米也不是已验证的真机安全边界。
人员活动区、意外失稳范围、急停距离和围障应另行考虑。本文不证明仿真指令可直接用于真机。

## 5. 单趟录制启动命令

前提：服务器已同步支持三柱的代码，以上 checkpoint、指令表及原始场景资源均存在。

**仅修改 `RUN_ID` 或使用 `recording_profile=pole_slalom` 不会自动加载调优表。**
下面通过进程内包装加载既有 NPZ，再调用专用录制入口，不修改仓库源码。
固定随机种子为 1；更换种子、checkpoint、柱位或步态配置后，不能直接沿用本次成功结论。

新录像示例命名为 `v4_g1_slalom_3p_thin_replay`：`v4` 是新的输出编号，读取的仍是 **v3 指令表**。
不用时间戳；输出已存在时拒绝覆盖，复跑请递增版本号。

```bash
cd /ssd1/cyx/TTT
mkdir -p logs/fada_online
RUN_ID="v4_g1_slalom_3p_thin_replay"
OUTPUT="$PWD/logs/fada_online/${RUN_ID}"

if [ -e "${OUTPUT}.log" ] || [ -e "${OUTPUT}.mp4" ] || \
   [ -e "${OUTPUT}.npz" ] || [ -e "${OUTPUT}.json" ]; then
  printf '输出已存在，请递增 RUN_ID 版本号：%s\n' "$OUTPUT" >&2
else
  nohup env \
    LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH="$PWD/src" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    nice -n 10 uv run --frozen --no-sync python -c '
import runpy
import numpy as np
import torch
from unilab.visualization.slalom import SlalomCommandProgram

np.random.seed(1)
torch.manual_seed(1)
torch.set_num_threads(1)
original = SlalomCommandProgram.from_config.__func__

def replay(cls, *args, **kwargs):
    program = original(cls, *args, **kwargs)
    path = "logs/fada_online/v3_g1_slalom_3p_thin/input_commands.npz"
    with np.load(path, allow_pickle=False) as data:
        if (
            program.segment_steps != int(data["segment_steps"])
            or not np.isclose(program.ctrl_dt, float(data["ctrl_dt"]))
            or not np.allclose(program.pole_centers, data["pole_centers"])
            or not np.allclose(program.reference_points, data["reference_points"])
        ):
            raise ValueError("Current course does not match the three-pipe input table")
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
    demo.slalom.pole_count=3 \
    demo.slalom.spacing_m=1.0 \
    demo.slalom.pole_radius_m=0.02 \
    demo.slalom.pole_height_m=0.8 \
    demo.slalom.pole_mass_kg=0.2 \
    demo.slalom.pole_friction=0.6 \
    env.runtime_payload.enabled=false \
    online_adaptation.enabled=false \
    video_recording.enabled=true \
    video_recording.duration_s=null \
    video_recording.output_path="${OUTPUT}.mp4" \
    > "${OUTPUT}.log" 2>&1 < /dev/null &

  printf 'PID：%s\n日志：%s.log\n视频：%s.mp4\n结果：%s.json\n轨迹：%s.npz\n' \
    "$!" "$OUTPUT" "$OUTPUT" "$OUTPUT" "$OUTPUT" >&2
fi
```

`video_recording.duration_s=null` 在这个专用入口中表示一趟输入指令加停止观察，不是无限录制。
进程输出 PID 只证明已提交后台启动；日志出现 `[slalom] SUCCESS` 且结果文件的 `success=true`，
才表示该次任务完成。启动异常或任务失败看同名前缀 `.log` 与 `.json`，不要跳过 checkpoint／资源契约校验。

需要看进度时，在同一个 Bash 中执行：

```bash
tail -f "${OUTPUT}.log" >&2
```

Ctrl-C 退出日志查看，不会停止已由 nohup 启动的录制。不要同时启动多份相同输出名的录制。

## 6. 成功标准与本次证据

实测报告记录：

- `success=true`、`failed=false`、`passed_poles=3`、`failure_reasons=[]`。
- `elapsed_s=56.4`，`control_steps=2820`，停止观察 `settle_elapsed_s=2.0`。
- `pole_fallen=[false,false,false]`，`pole_displaced=[false,false,false]`。
- `pole_contact=[false,false,false]`，`contact_samples=[0,0,0]`。

判定需要实际按顺序从交替两侧穿绕，基座相对柱心的侧向通过距离至少 0.34 米，实际越过终点，
且停止观察期完成。0.34 米不是机器人外壳与管面的距离；细管版本也没有为通过而放宽这个门槛。
柱子倾斜超过 30° 连续 0.2 秒、水平位移超过 0.1 米、错误一侧／遗漏柱子或机器人终止均算失败。
视频编码完成、指令播放完、没有摔倒都不能单独代替任务成功。

原五柱 v2 和 Gerono v14 的文件均保留，不能拿它们的录像、接触统计或占地替代本三柱版本。
用户认可的是当前三柱仿真展示；实物管径、重量、地面和硬件控制时序仍需真机阶段单独核对。
