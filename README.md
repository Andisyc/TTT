# ICE-Cal

ICE-Cal is a research repository for **In-Context Execution Calibration**: adapting a frozen
planner/execution stack from observable Support trajectories without online parameter updates.
The runtime is derived from UniLab, but this repository documents only the ICE-Cal research
question, its contracts, implementation evidence, and current design transition.

## 维护仓库前必须读取

1. [`AGENTS.md`](AGENTS.md)：环境、配置、Backend、资产、Sim2Sim 和 Git 的硬性契约。
2. [`note/governance/README.md`](note/governance/README.md)：当前三个研究轨道的治理游标与权限边界。
3. [`note/README.md`](note/README.md)：研究文档总索引和权威读取顺序。
4. [`note/fada/contracts/README.md`](note/fada/contracts/README.md)：当前有效及历史 FADA Contract。
5. [`note/architecture/08_in_context_execution_calibration.html`](note/architecture/08_in_context_execution_calibration.html)
   与 [`09_in_context_execution_calibration_design_inspector.html`](note/architecture/09_in_context_execution_calibration_design_inspector.html)：
   当前 Concept Figure 与 Design Inspector。
6. [`CONTRIBUTING.md`](CONTRIBUTING.md) 与 [`docs/README.md`](docs/README.md)：开发环境、提交规范、稳定工程说明和 Runbook 入口。

维护时必须使用 `uv run`；`scripts/` 只组装流程；任务、Reward 和 Backend 优先由 Hydra
配置及 owner 模块表达。训练、仿真、部署、Git 发布和研究语义激活都需要各自明确授权。
测试、静态检查或文档状态不能单独证明策略质量。

## Streaming TTT 在线实验

本仓库的在线 LoRA 实验入口为 `scripts/play_fada_online.py`，使用
`task=g1_walk_flat/mujoco recording_profile=straight_payload`，并通过
`training.play_checkpoint_path` 指定真实 Planner-IDM checkpoint。它与下述冻结权重的
ICE-Cal 路径不同：Planner 和 IDM 基础权重冻结，只有 IDM LoRA 接受在线更新。

- `online_adaptation.enabled=false` 保持无在线更新的 baseline；`true` 启用冻结基座轨迹差分标签。
- 在 t 保存实际推理的历史输入、完整 Planner 标称未来 `Y_nom` 和在线首动作快照；
  到 t+K 收齐同一时间段的真实状态轨迹 `Y_exec`。K 来自 checkpoint 的预测长度
  (当前模型 K=6)，不是训练批量 T；终止/reset 或命令切换不能拼成跨边界轨迹。
- 在更新批次形成时，冻结基座对同一历史分别输入 `Y_nom` 与 `Y_exec`，取首动作作
  `nom-exec` 差分。固定目标为 `U_snapshot + gamma * delta`，不使用单步关节误差、
  不除 `action_scale`，不随在线参数重算；环境执行端的动作缩放保持不变。
- 基座在在线 LoRA 注入前独立复制，只复制 IDM，eval/关闭梯度，不进入优化器。
- 默认 `window_size=10`、`warmup_steps=40`、`update_interval=10`，
  `gamma=0.05`、`sigma=1e-4`。控制周期 0.02s、K=6 时，第 40 步 (0.8s) 首次更新，
  使用最后 10 条成熟轨迹；此后第 50、60……步 (每 0.2s) 消费 10 条新成熟轨迹。
  成功只清成熟训练批，不清在途轨迹或推理历史；失败保留批次并抛出异常。
  非默认配置须同时满足 warmup、更新周期和满批条件，不等待负重挂载。
  这些数值是试验初值，不代表已证实的稳定性或适应效果。
- 损失为窗口内首动作误差平方范数的 `1/(2T)` 均值，加相对部署初始 LoRA 参数的
  显式 L2 锚定；不对动作维再平均。更新时关闭 dropout 但开启梯度，使用 Adam，
  不通过 AdamW 的默认权重衰减替代非零参数锚点。
- reset 清空推理历史、在途轨迹和成熟窗口，并重新计算 warmup，但保留已学 LoRA、
  冻结基座、锚点与优化器状态。
  关闭 autoreset 时，终止后仍可继续物理步进，但暂停采集/更新，直到显式 reset。
- 旧在线 `lambda_forget` 配置和 `scripts/test_online_adaptation.py` 手动入口已移除。
  使用 `uv run --frozen --no-sync python scripts/play_fada_online.py`，启动日志应包含
  `objective=expert_trajectory_difference`。视频录制、负重事件和离线 FADA 路径保持原接口。

### Linux/CUDA 服务器后台录制

以下沿用此前服务器的仓库和 checkpoint 示例路径，本轮未连接服务器核验。
如实际路径不同，只调整 `cd` 和 `training.play_checkpoint_path`。
命令使用已经配置好的 uv 环境，不创建环境、不安装依赖。
`LD_PRELOAD` 沿用此前 Linux x86_64 服务器的进程级 libgcc 设置，不是跨平台通用配置。

两组均从第 0 秒向左肩挂载 3kg，固定前进命令为 0.4m/s，侧后方跟随视角，
录制 120 秒仿真时间。命令显式覆盖 profile 默认的 30 秒挂载延迟，不修改 profile。
分别运行两组；除在线开关与产物命名外，checkpoint、物理和录制条件保持一致。
视频和日志名带时间戳；重复启动前确认名称不同，避免覆盖已有结果。

#### 开启基座轨迹差分在线更新

```bash
cd /ssd1/cyx/TTT
mkdir -p video logs
TTT_RUN_ID="$(date +%Y%m%d-%H%M%S)"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD/src" \
  uv run --frozen --no-sync python scripts/play_fada_online.py \
  task=g1_walk_flat/mujoco \
  recording_profile=straight_payload \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/FADA_Unofficial/model/v024_checkpoint/planner_idm_close_some_dr_v001.pt \
  env.runtime_payload.enabled=true \
  env.runtime_payload.added_mass_kg=3.0 \
  env.runtime_payload.attach_time_s=0.0 \
  online_adaptation.enabled=true \
  online_adaptation.window_size=10 \
  online_adaptation.warmup_steps=40 \
  online_adaptation.update_interval=10 \
  online_adaptation.gamma=0.05 \
  online_adaptation.sigma=1e-4 \
  online_adaptation.learning_rate=1e-4 \
  video_recording.enabled=true \
  video_recording.duration_s=120.0 \
  video_recording.progress_interval_s=1.0 \
  video_recording.output_path="$PWD/video/left_shoulder_expert_${TTT_RUN_ID}.mp4" \
  > "$PWD/logs/left_shoulder_expert_${TTT_RUN_ID}.log" 2>&1 &
```

在启动所在 shell 查看本次日志：

```bash
tail -f "/ssd1/cyx/TTT/logs/left_shoulder_expert_${TTT_RUN_ID}.log"
```

#### 相同负重、关闭在线更新的 baseline

这里的 baseline 仍然挂载 3kg，不是无负重 baseline。

```bash
cd /ssd1/cyx/TTT
mkdir -p video logs
BASELINE_RUN_ID="$(date +%Y%m%d-%H%M%S)"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD/src" \
  uv run --frozen --no-sync python scripts/play_fada_online.py \
  task=g1_walk_flat/mujoco \
  recording_profile=straight_payload \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/FADA_Unofficial/model/v024_checkpoint/planner_idm_close_some_dr_v001.pt \
  env.runtime_payload.enabled=true \
  env.runtime_payload.added_mass_kg=3.0 \
  env.runtime_payload.attach_time_s=0.0 \
  online_adaptation.enabled=false \
  online_adaptation.window_size=10 \
  online_adaptation.warmup_steps=40 \
  online_adaptation.update_interval=10 \
  online_adaptation.gamma=0.05 \
  online_adaptation.sigma=1e-4 \
  online_adaptation.learning_rate=1e-4 \
  video_recording.enabled=true \
  video_recording.duration_s=120.0 \
  video_recording.progress_interval_s=1.0 \
  video_recording.output_path="$PWD/video/left_shoulder_baseline_${BASELINE_RUN_ID}.mp4" \
  > "$PWD/logs/left_shoulder_baseline_${BASELINE_RUN_ID}.log" 2>&1 &
```

在该启动 shell 查看本次日志：

```bash
tail -f "/ssd1/cyx/TTT/logs/left_shoulder_baseline_${BASELINE_RUN_ID}.log"
```

`Ctrl-C` 退出 `tail` 不会停止后台录制。产物分别位于仓库的 `video/` 和 `logs/`，
以命令中的时间戳对应同一次运行。

#### 启动、进度与完成标准

- 新方法启动日志应包含 `objective=expert_trajectory_difference`，以及实际
  `prediction_horizon`、`window_size`、`warmup_steps`、`update_interval`。
  K 以加载 checkpoint 后的日志为准；在 K=6、0.02s 控制周期、没有提前终止的
  episode 中，默认在第 40/50/60……步更新。
- `Progress: step=当前步/总步` 同时显示仿真时间与帧数；`Update` 表示一次优化完成。
  进度按墙钟间隔在控制步完成后报告，不保证长耗时步骤内部每秒打印。
- baseline 应显示 `Online adaptation disabled`，视频更新计数保持 0，
  但同样应显示 `PAYLOAD +3kg`。
- 最后出现 `Video saved:` 才表示录制流程完成；文件已经存在或
  `Finalizing MP4 encoding...` 都不代表编码已经结束。
- 视频中的 `resets` 增长表示轨迹被终止/重置，不要把重置后的站立当成在线恢复。
  更新次数增加或 loss 较低也不能单独证明负重适应成功。

当前证据是第三步的真实小型网络与正式 factory/session 离线 smoke。
本节命令尚未在服务器实际启动，真实 checkpoint 下的录制与负重效果仍未验证。


## Current status

ICE-Cal has three separately governed research tracks; there is no single project-wide runtime
readiness label:

| Track | Current authority | Status |
|---|---|---|
| Source Oracle | `FADA-METHOD-v024` / `FADA-TRAIN-v024` | implemented and offline module-tested; bounded reset/step runtime smoke passed; full formal audit pending; not trained |
| Target adaptation | `FADA-ADAPT-METHOD-v003` / `FADA-ADAPT-TRAIN-v003` | active Contract; runtime and policy claims remain receipt-specific |
| In-context calibration | `FADA-CONTEXT-METHOD-v009` / `FADA-CONTEXT-TRAIN-v008` | data-driven basis design active; engineering proposal only |

The v024 Oracle replaces the unrun v023 phase-only route. It normalizes small commands to exact
zero, uses a deterministic null/non-null phase state machine, and schedules bounded contact
mismatch instead of a phase-height target. Translation and yaw jointly control a piecewise
`0`/`[f_min,f_max]` clock; Reward coefficients remain engineering parameters constrained by the
confirmed six-relation ordering card. The production selector is
`task=sac/g1_walk_flat/mujoco_fada_phase_contact`; offline tests do not establish simulator
reachability or policy quality. Targeted actuator faults are excluded from Source Oracle training;
left-knee gain attenuation is reserved for downstream failed-rollout collection after the Oracle is
frozen. Generic all-joint Kp/Kd and the remaining physical domain randomization stay enabled.

No design or documentation status authorizes training, simulation, deployment, or policy-quality
claims. See the [project governance index](note/governance/README.md) for the current cursor and
track boundaries.

## 仓库地图

```text
ICE-Cal/
│
├── conf/                                      Hydra 配置：决定“运行什么”
│   ├── offpolicy/                             SAC / TD3 / FlashSAC 配置根
│   │   ├── config.yaml                        离策略训练的顶层默认配置
│   │   ├── algo/                              算法参数
│   │   │   ├── sac.yaml                       SAC 网络、batch、更新频率、学习率
│   │   │   ├── td3.yaml                       TD3 参数
│   │   │   └── flashsac.yaml                  FlashSAC 参数
│   │   ├── task/                              任务与后端的配置所有者
│   │   │   └── sac/g1_walk_flat/
│   │   │       ├── mujoco.yaml                通用 MuJoCo G1WalkFlat
│   │   │       ├── motrix.yaml                通用 Motrix G1WalkFlat
│   │   │       ├── mujoco_clean_baseline.yaml 无 Gait、无特权输入、无 DR 基线
│   │   │       ├── mujoco_fada_source.yaml    v022 phase-neutral Source Oracle
│   │   │       ├── mujoco_fada_phase.yaml     v023 phase-locomotion 兼容配置
│   │   │       ├── mujoco_fada_phase_contact.yaml
│   │   │       │                               v024 command-gated phase/contact Source Oracle
│   │   │       ├── mujoco_fada_target*.yaml   FADA 目标域采集/评估任务
│   │   │       └── mujoco_context_teacher*.yaml
│   │   │                                       Context full-action/residual teacher
│   │   ├── fault/                             显式执行器故障
│   │   │   ├── left_knee_090.yaml             左膝 0.9 增益条件
│   │   │   └── right_knee_090.yaml            右膝 0.9 增益条件
│   │   ├── target_domain/                     目标域环境条件
│   │   │   ├── slope_10.yaml                  10° 斜坡
│   │   │   └── slope_15.yaml                  15° 斜坡
│   │   ├── collection/                        FADA 目标轨迹采集配置
│   │   ├── adaptation/                        IDM LoRA 适配配置
│   │   └── evaluation/                        FADA 评估配置
│   │
│   ├── distill/                               Planner–IDM 蒸馏配置
│   │   ├── task/g1_walk_flat/                 Student 环境和 Source 行为身份
│   │   ├── workflow/                          蒸馏阶段与采集流程
│   │   ├── calibration_playback/              校准策略回放覆盖
│   │   └── context_playback/                  Context 策略回放覆盖
│   ├── fada_context/                          In-Context 校准配置
│   │   ├── calibration_axes/                  Gain / Delay / Offset 等校准轴
│   │   └── calibration_collection/            校准数据采集事务
│   ├── appo/                                  异步 PPO 任务配置
│   ├── ppo/                                   RSL-RL PPO 任务配置
│   ├── ppo_him/                               HIM-PPO 任务配置
│   └── hora_distill/                          HORA 蒸馏配置
│
├── scripts/                                   面向用户的命令入口；只组装流程
│   ├── train_offpolicy.py                     SAC / TD3 / FlashSAC 训练入口
│   ├── train_distill.py                       FADA Planner–IDM 蒸馏入口
│   ├── play_interactive.py                    原生 MuJoCo 交互回放
│   ├── collect_fada_target.py                 FADA 目标域轨迹采集
│   ├── adapt_fada_target.py                   FADA 目标域 LoRA 适配
│   ├── evaluate_fada_slope.py                 斜坡适配前后评估
│   ├── train_fada_calibration*.py             Context 校准 Stage 1/2/3
│   ├── train_appo.py                          APPO 训练入口
│   ├── train_rsl_rl.py                        PPO 训练入口
│   ├── train_mlx_ppo.py                       MLX PPO 训练入口
│   ├── deploy/                                部署入口和适配器
│   ├── motion/                                动作数据处理工具
│   └── manip_loco/                            操作与移动复合任务工具
│
├── src/unilab/                                生产代码
│   │
│   ├── base/                                  框架核心契约
│   │   ├── np_env.py                          NpEnvState、reset/step 环境契约
│   │   ├── registry.py                        Env/EnvCfg 注册与实例化
│   │   ├── observations.py                    观测组公共结构
│   │   ├── curriculum.py                      通用课程机制
│   │   └── backend/
│   │       ├── base.py                        SimBackend 公共抽象接口
│   │       ├── mujoco/                        MuJoCo 实现、XML 与回放
│   │       └── motrix/                        Motrix 实现、场景与回放
│   │
│   ├── envs/                                  任务环境
│   │   ├── common/                            环境共享数学/旋转工具
│   │   ├── locomotion/
│   │   │   ├── common/                        各机器人共用移动逻辑
│   │   │   ├── go1/ go2/ go2w/ go2_arm/      其他腿式机器人任务
│   │   │   └── g1/                            G1 移动任务核心
│   │   │       ├── joystick.py                G1WalkEnv 组合入口与注册
│   │   │       ├── base.py                    G1 资产和基础状态
│   │   │       ├── walk_config.py             Env/Control/Reward/DR 配置结构
│   │   │       ├── walk_commands.py           速度指令采样和重采样
│   │   │       ├── walk_observation_bindings.py
│   │   │       │                               Actor/Critic/FADA 观测组装
│   │   │       ├── walk_control_bindings.py   Action 门控与实际执行控制
│   │   │       ├── walk_reward_bindings.py    Reward 汇总和终止相关代价
│   │   │       ├── walk_runtime_bindings.py   每步状态、课程日志、快照
│   │   │       ├── walk_domain_randomization.py
│   │   │       │                               G1 专属 DR 计划与课程
│   │   │       ├── walk_actuator_randomization.py
│   │   │       │                               执行器强度采样
│   │   │       ├── walk_reset_randomization.py
│   │   │       │                               Reset 状态和物理参数随机化
│   │   │       ├── fada_privileged.py         FADA 特权向量和布局身份
│   │   │       ├── calibration_fault.py       校准用执行故障注入
│   │   │       ├── action_trace.py             Action 权限与调试轨迹
│   │   │       └── walk_{control,reward,observation,math}.py
│   │   │                                       可复用的纯计算函数
│   │   ├── manipulation/                      Allegro/Sharpa/Stewart 操作任务
│   │   └── motion_tracking/g1/                G1 参考动作跟踪
│   │
│   ├── training/                              应用层训练流程
│   │   ├── offpolicy/
│   │   │   ├── application.py                 开始、训练、清理、结束生命周期
│   │   │   ├── factory.py                     Env/Learner/Runner 构建
│   │   │   └── playback.py                    Checkpoint 加载与回放
│   │   ├── backend_adapter.py                 Hydra 配置到 Backend/Env 的适配
│   │   ├── sim2sim.py                         跨后端配置兼容性检查
│   │   ├── experiment.py                      实验身份、摘要和日志生命周期
│   │   └── run.py                             通用运行辅助逻辑
│   │
│   ├── algos/                                 Learner 与研究方法
│   │   ├── torch/
│   │   │   ├── fast_sac/
│   │   │   │   └── learner.py                 SAC Actor/Critic、Loss、优化器
│   │   │   ├── fast_td3/                      TD3 Learner
│   │   │   ├── flash_sac/                     FlashSAC Runtime
│   │   │   ├── appo/                          APPO 实现
│   │   │   ├── offpolicy/                     离策略采集与学习调度
│   │   │   │   ├── worker.py                  Collector 子进程环境交互
│   │   │   │   ├── collector_session.py       Collector 会话生命周期
│   │   │   │   ├── double_buffer_runner.py    双缓冲训练调度与保存
│   │   │   │   ├── runner.py                  基础离策略 Runner
│   │   │   │   └── checkpoint_adapter.py      通用 Checkpoint 适配边界
│   │   │   ├── distill/
│   │   │   │   └── fada/                      FADA 主实现
│   │   │   │       ├── source/                公共阶段 1：Source Oracle
│   │   │   │       │   ├── runtime.py          特权 SAC Runtime 解析
│   │   │   │       │   └── diagnostics.py      Source 诊断入口
│   │   │   │       ├── planner_idm/           公共阶段 2：Planner–IDM
│   │   │   │       │   ├── __init__.py         模型与 Checkpoint 公共 API
│   │   │   │       │   └── training.py         Source 蒸馏流程入口
│   │   │   │       ├── target/                公共阶段 3：目标域
│   │   │   │       │   ├── collection.py       目标 rollout 采集入口
│   │   │   │       │   ├── adaptation.py       IDM LoRA 适配入口
│   │   │   │       │   └── evaluation.py       目标域评估入口
│   │   │   │       ├── model.py               Planner、IDM、组合策略
│   │   │   │       ├── checkpoint.py          Source Checkpoint 契约
│   │   │   │       ├── collector.py/oracle.py Source 轨迹与 Oracle 标签
│   │   │   │       ├── windows.py/replay.py   H/K 窗口和训练样本
│   │   │   │       ├── collection_*.py        采集身份、状态、事务、IO
│   │   │   │       ├── target_*.py            目标采集/适配/评估内部实现
│   │   │   │       ├── adaptation*.py         LoRA 适配与持久化
│   │   │   │       └── legacy_workflow.py     旧 Schema 隔离兼容层
│   │   │   └── fada_context/                  In-Context 校准
│   │   │       ├── calibration_{data,collection}.py
│   │   │       │                               校准样本与采集
│   │   │       ├── calibration_{models,policy}.py
│   │   │       │                               校准模型和冻结策略组合
│   │   │       ├── calibration_training/
│   │   │       │   ├── stage1.py               修正算子
│   │   │       │   ├── stage2.py               系数 Encoder
│   │   │       │   ├── stage3.py               尺度映射
│   │   │       │   ├── pipeline.py             串联三个冻结阶段
│   │   │       │   └── io.py/types.py          阶段产物 IO 与类型
│   │   │       ├── calibration_{runtime,readout}.py
│   │   │       │                               推理读数与修正组合
│   │   │       ├── calibration_evaluation.py  校准前后评估
│   │   │       └── formal_protocol.py         官方路径审计入口
│   │   └── mlx/                               MLX 算法实现
│   │
│   ├── ipc/                                   进程间通信与 Replay
│   │   ├── replay_buffer.py                   CPU 共享 Replay 存储
│   │   ├── async_runner.py                    通用异步 Runner 基础设施
│   │   ├── weight_sync.py                     Collector/Learner 权重同步
│   │   ├── shared_buffer.py                   共享内存基础结构
│   │   └── replay_pipelines/
│   │       ├── cpu_pinned_double_buffer.py    Batch 准备与 H2D 重叠
│   │       └── transfer/                      CUDA/XPU/Torch 传输实现
│   │
│   ├── dr/                                    与后端无关的域随机化契约
│   ├── assets/                                Robot XML、网格、场景、动作
│   │   ├── robots/g1/                         G1 模型和 task-level 场景
│   │   ├── robots/{go1,go2,go2w,...}/         其他机器人资产
│   │   └── motions/g1/                        G1 参考动作
│   ├── terrains/                              地形生成与表示
│   ├── logging/                               训练指标与 Trace
│   ├── visualization/                         渲染和可视化
│   ├── tools/                                 仓库/运行时工具
│   └── utils/                                 通用工具
│
├── tests/                                     与生产边界对应的测试
│   ├── architecture/                          所有权和仓库形状检查
│   ├── config/                                Hydra 组合契约
│   ├── envs/locomotion/g1/                    G1 Reward/Obs/Control/DR
│   ├── algos/                                 SAC、FADA、Context
│   ├── ipc/                                   Replay 与异步生命周期
│   ├── training/                              Training 应用层
│   ├── base/backend/                          Env/Backend 公共契约
│   ├── integration/                           跨模块集成
│   └── scripts/                               CLI 接线
│
├── note/                                      研究权威与证据
│   ├── governance/                            当前项目/轨道治理游标
│   ├── architecture/                          Concept Figure、Design Inspector
│   └── fada/
│       ├── contracts/
│       │   ├── active/                        当前 Method/Training Contract
│       │   └── history/                       已被替代、不可授权的旧 Contract
│       ├── plans/                             带日期的设计和工程方案
│       ├── governance/                        阶段转换与执行 Receipt
│       ├── testing/                           Module Test Card 和测试记录
│       ├── evidence/                          失败、运行与评估证据
│       └── reviews/                           方案/代码可维护性审查
│
├── docs/                                      稳定工程文档
│   ├── engineering/                           实现说明
│   ├── runbooks/                              可重复执行的操作手册
│   └── superpowers/                           历史开发过程记录，非当前权威
│
├── benchmark/                                 性能基准和测试数据
├── notebook/                                  探索性 Notebook，非生产 Owner
├── model/                                     本地 Checkpoint，使用前核对身份
├── logs/                                      训练与运行日志
├── artifacts/                                 FADA 数据集与评估产物
├── work/                                      生成的工作材料
├── pyproject.toml                             Python 依赖与工具配置
├── Makefile                                   仓库验证入口
├── CONTRIBUTING.md                            开发环境与贡献流程
└── AGENTS.md                                  Agent 必须遵守的仓库契约
```

## Oracle Policy 训练链路

独立的 `task=sac/g1_walk_flat/mujoco_motion_quality_oracle` 默认启用左右镜像增强与
`symmetry_mirror_loss_coeff=0.1`，转向跟踪误差分母 `tracking_ang_sigma=0.25` 与官方奖励宽度一致。
站立、指令分布、弱域随机化及其余奖励保持原设置；已有 checkpoint 仍按其保存配置读取。

Oracle 蒸馏源适配器位于 `src/unilab/algos/torch/distill/fada/oracle_source.py`：
`build_oracle_fada_source()` 接收 checkpoint、环境数、设备和可选场景路径，
返回 `(env, frozen_teacher, source_profile)`。它保留源环境和随机化，不应用播放 profile；
教师权重与归一化冻结，资产迁移仍校验内容和关节顺序。
返回的环境尚未初始化、重置或步进；调用方应先准备需要的隔离分支，再初始化，
并在 `finally` 中关闭环境。构建失败由适配器关闭已创建的环境。
`source_profile` 为可序列化的 `unilab_oracle` v1 数据，包含教师身份、权重哈希、
源环境及控制契约和 `g1_fada_state_v2` 学生观测契约，由采集产物携带，不单独保存为 profile 文件。

Oracle 批量采集入口是 `src/unilab/algos/torch/distill/fada/oracle_runtime.py` 的
`build_oracle_fada_runtime()`，接收蒸馏配置、66/29/3 架构、最终 Oracle、初始学生、
可选中间 Oracle 窗口分配及场景路径。当前 owner 为 `G1WalkFlat/mujoco`，
要求 `training.fada.oracle_shadow_enabled=true`。
先调用 `prepare(DaggerPrepareRequest)` 获取 `metadata.source_profile`；
准备进程只构建环境和隔离分支，随后关闭，不初始化或步进物理。
再用 `activate_checkpoint(path, state_dict=student.state_dict())` 发布学生权重；
该路径是版本身份，不要求预先存在学生 checkpoint。
`collect(DaggerCollectRequest)` 使用原生窗口采集器和请求级进程，最终 Oracle 提供影子参考，
中间 Oracle 或学生可控制实际轨迹；教师权重和两类归一化统计始终冻结。
产物及结果均携带源 profile，窗口和旧原生源格式保持一致。
父进程读取产物后先调用 `identity_fields()` 校验行字段，再通过
`FADAReplayBuffer.add_artifact()` 入库；最后关闭运行时。
关闭路径会显式 join/close 采集进程并等待自有队列 feeder 退出，不依赖垃圾回收释放句柄。
共享权重由创建者负责 unlink；导出视图阻止句柄关闭时，`cleanup()` 仍移除共享内存名称并传播错误，
调用方释放视图后可重试关闭句柄。附加进程的 `close()` 不移除创建者的名称。
原生训练入口为 `scripts/train_distill.py --config-name oracle_fada`，
`teacher.checkpoint_path` 指定最终 Oracle，`training.fada.scene_model_file` 可重定位同内容场景。
默认使用最终 Oracle；设置 `training.fada.paper_source_enabled=true` 时沿用现有
20 个中间 checkpoint、2:1 次优窗口预算契约。学生保存完整的
`runtime_config.source_profile` 及两个优化器状态；不改变官方 PPO 蒸馏入口。

学生回放和在线适应复用 `scripts/play_fada_online.py --config-name oracle_fada`。
回放在建环境前恢复内嵌源配置，不读取原 Oracle 文件，也不要求官方学生的外置 profile；
原有旧学生和官方学生分支保持不变。录制的指令、噪声、DR、负重、时长和 reset 设置仍可覆盖，
源物理步长、控制周期、PD、动作缩放、相位时钟及观测/关节顺序必须匹配。
跨机器时用 `training.fada.scene_model_file` 指向同内容场景文件及其配套资产。

下面以 `model/mirror/model_50000.pt` 及现有本地配套资产为例，输出到一个新的训练目录：

```bash
uv run --frozen --no-sync python scripts/train_distill.py --config-name oracle_fada \
  teacher.checkpoint_path=model/mirror/model_50000.pt \
  training.fada.scene_model_file=model/straight_mix_50k_assets/g1/scene_flat.xml \
  training.fada.checkpoint_path=logs/fada/oracle_demo/student.pt \
  training.fada.async_artifact_dir=logs/fada/oracle_demo/source_batches
```

Linux 服务器录制该次训练得到的学生，立即挂载 3 kg 负重并启用 FADA-TTT：

```bash
env LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
  uv run --frozen --no-sync python scripts/play_fada_online.py --config-name oracle_fada \
  recording_profile=straight_payload \
  training.play_checkpoint_path=logs/fada/oracle_demo/student.pt \
  training.fada.scene_model_file=model/straight_mix_50k_assets/g1/scene_flat.xml \
  env.runtime_payload.attach_time_s=0.0 \
  online_adaptation.mode=fada_ttt online_adaptation.sigma=0 \
  video_recording.duration_s=30 video_recording.output_path=video/oracle_student_ttt.mp4
```

默认训练/录制设备为 CUDA；本地 CPU 使用 `training.device=cpu`，macOS 不使用上述 Linux 环境前缀。
原来的 `trajectory_base`（OAT）参数未改；FADA-TTT 明确指定 `mode=fada_ttt`、`sigma=0`。
关闭在线适应可设 `online_adaptation.enabled=false`；八字录制改为 `recording_profile=figure_eight`。


Oracle 双圆八字回放会话由 `src/unilab/visualization/oracle_recording.py` 的
`create_oracle_recording_session()` 创建：恢复 checkpoint 的控制接口和冻结归一化，
调用方依次使用 `reset()`、`step_once()`，并在 `finally` 中 `close()`。
会话在首次终止或超时后停止；显式 reset 会清空双圆进度和足迹。
会话已通过 `record_fada_playback(..., policy_kind="oracle")` 接入共享录制器：
首次终止会输出末帧并结束，日志报告实际步数、仿真时长和帧数，画面显示 Oracle 身份而非 LoRA 信息。
默认 `policy_kind="fada"` 保持原行为。独立录制入口是 `scripts/record_oracle.py`，
配置入口是 `conf/record_oracle.yaml`：只借用共享八字 profile 的镜头和显示默认值，
不注入 FADA 的任务配置或在线适应设置，轨迹来自独立的 `demo.double_circle` 固定指令。

Oracle 默认先右转一整圈，再左转一整圈，不使用路径跟踪或在线调参。
针对本地 `model/mirror/model_50000.pt` 的无负重 MuJoCo 测量，前进指令固定为
`0.3 m/s`，右转 `-0.3 rad/s` 保持 `27.18 s`，左转 `+0.3 rad/s` 保持 `26.38 s`，
一组双圆共 `53.56 s`。时长已计入起转和右转切左转的过渡，并对齐 `ctrl_dt=0.02 s`。
这些是该 checkpoint 的固定时长，不是其他策略或真机的通用标定值。
灰色参考线为半径 `1 m` 的两个完整圆，不再显示 Gerono 曲线；
FADA 的原 Gerono 发生器保持不变。录制仍按固定时间推进，没有位置反馈纠偏。

服务器上录制新 Oracle checkpoint 的一组完整双圆八字：

```bash
cd /ssd1/cyx/TTT
REC_DIR="$PWD/logs/oracle/figure_eight_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$REC_DIR"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  uv run --frozen --no-sync python scripts/record_oracle.py \
  training.play_checkpoint_path=/ssd1/cyx/TTT/logs/oracle/mirror_yaw025_50k_20261108-054124/model_50000.pt \
  training.device=cpu \
  video_recording.duration_s=null \
  video_recording.output_path="$REC_DIR/figure_eight.mp4" \
  > "$REC_DIR/record.log" 2>&1 &

echo "录制进程 PID: $!"
echo "视频：$REC_DIR/figure_eight.mp4"
echo "日志：$REC_DIR/record.log"
```

省略 `video_recording.duration_s` 或设为 `null` 时默认只录一组完整双圆；首次终止或超时仍会提前结束。
推理默认使用 CPU，Linux 离屏渲染默认使用 EGL。跨机器搬运 checkpoint 时，
可用 `env.scene.model_file=/绝对路径/scene_flat.xml` 显式指定配套场景资源，
资产身份校验仍然保留。

以下链路对应当前仓库中可直接执行的 v022 Source Oracle：配置入口为
`mujoco_fada_source.yaml`，实现是 privileged SAC，Checkpoint 使用 sealed lineage 契约。

```text
Hydra 配置
│
├── conf/offpolicy/config.yaml
│     Off-policy 顶层训练配置
│
├── conf/offpolicy/algo/sac.yaml
│     SAC 网络、Replay、batch、学习率和更新频率
│
└── conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_source.yaml
      Oracle 行为、特权输入、Reward、Grouped DR 和 Checkpoint 谱系
        ↓
scripts/train_offpolicy.py
      Hydra 命令入口，只负责组合配置并转交训练应用
        ↓
src/unilab/training/offpolicy/application.py
::run_offpolicy()
  ├── ensure_registries()                         注册环境与算法
  ├── 设置随机种子
  ├── 创建 ExperimentTracker
  ├── build_runner()
  └── runner.learn()
        ↓
src/unilab/training/offpolicy/factory.py
::build_runner()
  ├── BackendAdapter                              生成 EnvCfg override
  ├── 创建一次 G1WalkEnv                          读取 obs/action 维度
  ├── 解析 algo.runtime_resolver
  └── 构造 Learner + DoubleBufferOffPolicyRunner
        ↓
algo.runtime_resolver
src/unilab/algos/torch/distill/fada/source/runtime.py
::resolve_privileged_locomotion_sac_runtime()
        ↓
FADAPrivilegedSACRuntime
  ├── validate_training_config()
  ├── 读取 G1 privileged observation 布局
  ├── 构造 FADAOracleCheckpointContract
  ├── 提供 FADAPrivilegedSACLearner
  └── 提供 sealed Checkpoint saver
        ↓
G1WalkEnv
  ├── 普通任务观测
  │     → Actor 基础输入
  │
  ├── g1_fada_privileged_v1
  │     → 追加到 Critic observation 尾部
  │
  ├── Command
  │     → 速度跟踪目标
  │
  ├── Reward
  │     → phase-neutral，feet_phase = 0
  │
  └── Physical DR
        ├── 定向执行器故障（包括左膝增益衰减）：仅用于 Oracle 冻结后的故障轨迹采集
        ├── 全关节 Kp / Kd：保留
        ├── 摩擦
        ├── base / body mass
        ├── COM
        └── DoF position bias
        ↓
FADAPrivilegedSACLearner
  └── 继承 HoraSACLearner
        └── 继承 FastSACLearner
              ↓
Actor
  ├── 普通任务观测
  ├── 从 Critic observation 尾部提取 privileged vector
  ├── 对 privileged vector 做归一化
  ├── Privileged MLP：256 → 128 → 32
  └── 普通观测 + 32-D privileged embedding → Action
              ↓
Critic
  └── 完整 Critic observation + Action → Distributional Q
              ↓
DoubleBufferOffPolicyRunner
  ├── Collector worker 与 G1WalkEnv 交互
  ├── transition 写入 ReplayBuffer
  ├── CPU-pinned DoubleBuffer 准备 batch
  ├── update_critic()
  ├── update_actor()
  ├── 更新温度参数
  └── 软更新 Target Q
              ↓
FADAOracleCheckpointGateway
  └── 保存 model_<iteration>.pt
        ├── Actor / Critic 参数
        ├── privileged normalization
        ├── oracle_lineage_id
        ├── behavior_profile
        ├── Env / Reward / Algo 配置哈希
        ├── observation / action 布局
        └── robot asset 身份
```

## 官方 FADA Locomotion 源训练

正式入口为 `scripts/train_distill.py --config-name official_fada`，配置位于 `conf/distill/official_fada.yaml`。官方入口现在只走共享轨迹收发、有限学习流水线、内存权重发布及单任务配对存盘，无加速开关或磁盘 fallback。非 FADA 蒸馏、旧 checkpoint 播放和现有在线 LoRA 方法保持独立。

在仓库根目录启动；先将 `UPSTREAM_ROOT` 和 `EXPERT_CHECKPOINT` 设置为真实官方 checkout 和 PPO 产物的绝对路径。以下命令开启已有性能计时，便于服务器对照：

```bash
: "${UPSTREAM_ROOT:?请设置固定版本官方 checkout 的绝对路径}"
: "${EXPERT_CHECKPOINT:?请设置官方 PPO checkpoint 的绝对路径}"
PYTHONPATH="$PWD/src" uv run --frozen --no-sync python scripts/train_distill.py \
  --config-name official_fada \
  training.fada.upstream_root="$UPSTREAM_ROOT" \
  training.fada.config.expert_checkpoint="$EXPERT_CHECKPOINT" \
  training.fada.performance.enabled=true
```

官方源码固定为 `LeCAR-Lab/FADA@290b3a9c5c0ebe2e9851b61350617544be05e9bc`。需要官方可读取的 PPO 配套保存配置、弱 checkpoint 历史，以及 Linux/CUDA/官方原生仿真环境；现有 MuJoCo SAC/APPO 和 66/3 学生不能替代。命令不会下载权重或安装依赖。

### 执行预算与恢复（第 6.5 步）

执行预算在 `training.fada.execution`，与官方 `training.fada.config` 算法参数隔离，不写入学习 checkpoint、缓存或来源身份。配置使用 bytes，默认如下；这是声明的 buffer 上限，不是进程总 RAM/VRAM 需求。

| 配置键 | 默认值 | 覆盖范围 |
|---|---:|---|
| `transport_budget_bytes` | 128 MiB | 全部共享槽的 raw payload；分配还需 metadata/header 空间 |
| `prefetch_budget_bytes` | 1 GiB | host/device 槽位副本及有界采样/拼接临时峰值 |
| `pending_raw_budget_bytes` | 2 GiB | 一次已采集、待入 Replay 的自有 raw 暂存 |
| `slots` | 2 | 原始轨迹传输槽位数 |
| `prefetch_batches` | 2 | 学习槽位总数，含正在消费及 in-flight batch |

解析会在建模型/worker 前拒绝未知键、bool/非正整数和无法容纳配置工作量的预算。Linux 还会检查 `/dev/shm` 可用容量，容器必须提供足够共享内存；pinned RAM、Replay、模型、原生环境和 shadow 的 RAM/VRAM 需求仍需分别留足。容量不足就调整对应 execution 预算或运行环境，不会自动减少环境数、样本、更新或切换 device；例如只改暂存上限可追加 `training.fada.execution.pending_raw_budget_bytes=4294967296`。

恢复时，在上述命令追加 `training.fada.config.resume_checkpoint="$CHECKPOINT"`，其中 `CHECKPOINT` 必须指向实际学习 checkpoint；保留它引用的 online train/val NPZ 文件及同身份的 offline caches。不要把裸学生权重当作完整学习恢复，也不要用额外 `resume_online_buffer` 覆盖正确的配对引用。新 run 默认仍可能重新执行 warmup（`force_warmup_train=true`）；已有 cache 不等于已经跳过学习。上传代码不会热切换已启动进程，新路径在使用更新配置启动或恢复后生效。


### 采集、Replay 与外循环

- `fada.async_runtime.build_official_fada_runtime` 接收真实 `OfficialFADACollectionSpec`、`initial_student` 和共享端点。先由同一 Runner 的 `prepare` 核准 Oracle 身份、预处理、配置和弱策略列表，再通过 `activate_checkpoint(..., state_dict=...)` 从当前内存发布权重；不为发布重建学生模型或调用推理 loader。
- `OfficialFADACollectRequest` 明确指定 `optimal` / `suboptimal` / `online`；弱策略请求另带 `weak_checkpoint`，可选 `collect_steps` 以跨全部 env 的原始 transition 总数计。一个常驻 worker 复用官方 env、Trainer 和固定 Oracle shadow；不调用官方完整训练主循环。
- 原始 13 字段（obs、command、executed/expert action、两类 expert chunk 及有效标志、strict-label 标志、reward、done、env/episode ID）直接经 typed 共享槽传输。接收方持有自有 raw 暂存，核准尾块/来源/版本后才入 Replay；正式训练不创建 `source_batches`/raw shard 交接文件。原 schema v8 artifact API 仍用于明确的文件数据用途，旧窗口 v7 不能冒充 raw Replay，历史 v5 reader 不变。
- 父进程的 `OfficialFADAReplayBuffer` 复用固定上游纯 NumPy ReplayBuffer，按轨迹抽取、train/val 划分和 random/fifo 淘汰；容量单位为 transition。离线集可增长，online 跨轮积累；更新时重新 `sample_chunk`，而不是反复训练 worker 预抽的静态窗口。跨 shard 的完整 episode 统一判断。
- 唯一外循环保留 offline 采集/缓存和 warmup；optimal 按固定 horizon 保留完整 episode，suboptimal 轮换弱 checkpoint 至目标量。学习阶段用有限 CPU 预取与设备搬运流水线，采样配额、增强、dropout 和两阶段更新顺序不变。每轮结束从内存发布新权重，再让配对存盘与下一轮采集重叠；存盘完成前不修改 learner/Replay。
- 第一轮 online 前发布 warmup 后权重，每轮更新后发布新权重；`save_every_iter` 仅控制保留的周期快照，不降低权重发布频率。恢复包含模型、两组 optimizer、更新/轮次/采集计数及对应 online train/val Replay；不承诺物理环境或 RNG 的精确断点重放。
- non-daemon collector 支持官方 shadow 子进程，错误通道、版本屏障和进程组回收保持不变。重叠的是 batch 搬运/学习及存盘/采集，不是用旧策略跨轮采集；原生仿真和 strict-shadow 算法没有被替换。

### 学生与学习接口

- `fada.official_model.build_official_fada_student` 构建纯 Torch 官方结构，默认 compact/action/command 为 64/29/7，H=30、K=6；保留官方位置编码、pre-Norm、delta 重建与原生权重布局，不借用旧 66/3 模型。
- `OfficialFADALearner.update(idm_batches_by_source, planner_batch)` 每批先更新 IDM，再固定其参数、经 IDM 反传更新 Planner。保留官方增强、mask、AdamW、梯度裁剪与来源配额；输出完整 K 步，但默认 `idm_action_loss_horizon_gamma=0` 仅监督首动作，Planner 状态损失系数默认 0。`sample_update(mode="offline" 或 "mixed")` 先采 Planner，再采五种 IDM 用途。
- 预处理来自 worker 实际解析的 Oracle run；artifact 必须与其身份一致，不猜测缺失的 scale/noise，也不额外启用 IO normalization。
- `save_official_fada_checkpoint` 保存独立 `unilab_official_fada_student` schema v2、模型/训练/预处理配置、原生权重、两组 optimizer、计数和 Replay 引用；`load_official_fada_checkpoint` 严格恢复学习状态，`load_official_fada_student_checkpoint` 仅恢复 eval 模型。旧学生、旧官方诊断 schema 和 Oracle PPO/APPO 不混用。

已完成两类离线证据：固定官方源码的 CPU 数值对照（网络、采样、loss、权重与 AdamW 状态），以及真实入口、原始 Replay、多 shard、warmup、两轮 DAgger、共享权重发布和恢复的短链路验证。后者使用明确标识的诊断轨迹，不代表真实采集。官方原生仿真验收、shadow 状态/RNG 隔离和策略质量仍需实际产物、环境及运行授权。

### 官方学生播放与在线适应（第 5 步）

- 沿用 `scripts/play_fada_online.py` 和同一 playback session。使用 `task=g1_walk_flat/mujoco`、`interactive.action_mode=policy`，将 `training.play_checkpoint_path` 指向官方学生 checkpoint，并显式提供 `official_playback.profile_path`。关闭/开启更新仍由 `online_adaptation.enabled=false/true` 控制，不另建播放或更新循环。
- profile 是从真实 Oracle run 核准的 YAML/JSON，必须包含 `source_identity`、`joint_names`、`default_joint_pos`、`action_scale`、`stiffness`、`damping`、`clip_actions`、`sim_dt`、`ctrl_dt`、`gait_frequency`、`gait_clock_mode`、`phase_init`。身份必须与 checkpoint 一致；关节相关向量按源关节顺序提供 29 项，初始双脚相位为弧度。创建环境前从 profile 注入 `sim_dt`，部署适配器再校验实际物理步长；不得用 task 默认物理步长代替。checkpoint 本身没有完整控制契约，不能用旧 task 的姿态、scale 或 PD 默认值代替。
- 官方 64D 观测从公共物理状态接口组装：世界系角速度与重力转到基座系，关节位置减源默认角，按源关节顺序施加 checkpoint 预处理。不是截取旧 actor 观测的前 64 维，也不重复施加旧 actor 的缩放。完整 7D command 保留速度和四个步态 sin/cos；零速仍保留相位。
- source 归一化动作经 source clip 后作为快照与历史保存，再映射到目标关节顺序、默认角和 scale；PD、控制周期和相位时钟必须匹配 profile。冷启动及 reset 左补零并携带 bool mask，历史不跨 episode。相位持续变化不会被误判为新的速度指令而清空待成熟轨迹。
- 在线路径保留冻结 Planner/独立 base IDM、Q/V LoRA、固定伪标签、`Y_nom` 回归条件和原窗口/Adam/锚定/reset 语义。官方模型通过原生 command/mask 接口调用；旧 66/3 与既有 adapted checkpoint 的加载和播放路径保留。

CPU 诊断已通过正式 loader/factory/session 的推理、完整 K 步收集、实际 LoRA 更新、partial reset、旧路径及缺失/错配 profile 提前拒绝检查。诊断环境不含物理仿真；真实官方产物绑定、原生播放与负重效果尚未验收。

### 可选性能计时（第 6A 步）

在官方源训练命令末尾追加 `training.fada.performance.enabled=true` 即可开启；默认关闭。开关位于原生算法配置之外，不进入 checkpoint、缓存或 Oracle 身份，也不改变采样量、更新量、loss 或权重发布规则。

- 控制台在 warmup 完成、每轮提交和最终关闭后输出 `[FADA PERF]`；结果为累计值，最终返回值另含 `performance`。提交边界可能已经包含下一轮采集，不能把相邻行直接当作严格的单轮物理耗时；同工作量对照优先看完整边界及实际计数。
- `wall_seconds` 覆盖初始化、采集 RPC、Replay 入库、batch 获取等待、IDM/Planner 前反向与 optimizer、发布和关闭。`batch_build` 覆盖实际取得 batch 的等待，不包含随后学习。最终 `total_elapsed` 覆盖训练 owner 初始化至关闭，不含此前 CLI/Python 导入；端到端耗时应另计。
- `worker_wall_seconds` 汇总已完成的 collector、独立 CPU sampler 和配对提交任务耗时。rollout 含原生 Oracle/shadow/环境事务，不冒充纯物理计算；CPU sampler 不冒充 CUDA H2D，未测阶段不补零。RPC、worker、保存与子阶段存在包络和重叠，不能相加当作总时间。
- `cuda_stream_seconds` 仅在 CUDA 路径记录本进程当前 stream 的 event 间隔，不是整个 GPU 或 shadow 子进程的 kernel 总耗时；仅在报告边界统一解析，不逐算子强制同步。CPU 下此字段为空。
- `counters` 区分实际采集的原始 transition、抽取的训练窗口、学习更新和 online 轮数；窗口计数不是新增物理数据量。未执行的阶段不填假零值。

已用一套 CPU 诊断短链路验证计时关闭/开启时模型、两组 AdamW、loss、RNG、保存轨迹及发布版本逐项一致，走通 warmup 和两轮 DAgger。原生仿真、CUDA 计时及官方/TTT 性能对照尚未运行；本次没有实现或宣称训练加速。

### 加速执行契约（第 6.1 步）

- `collection_contract.py` 的 `official_raw_transition_layout(spec, count)` 和 `official_window_batch_layout(spec, count)` 统一给出 raw 13 字段、window 9 字段的 `(shape, dtype)`；现有 batch validator 已复用，H/K 来自 spec，bool mask、int32 env ID、int64 episode ID 不转为 float。
- `execution_contract.py` 的 `FADAExecutionLimits`、`layout_nbytes`、`raw_chunk_rows` 和 `validate_prefetch_budget` 核算显式字节预算。`raw_chunk_rows` 接收每个字段首维为 1 的单行布局；预算覆盖所有槽位的数组 payload。预取调用方须列出每份保留的 host/device staging 副本，预算不代表整个模型或模拟器进程内存。
- `FADAChunkDescriptor.validate` 对照原采集 request 及接收方持有的游标/槽位代次检查版本、来源、行范围和尾块；`validate_phase_boundary` 在成功提交阶段前要求计划、抽取和消费次数相等且无剩余批次，不修改 RNG。

一次 CPU 契约场景已通过现有 batch 消费接口、预算与错误描述符拒绝检查，Python/NumPy/Torch RNG 状态不变。本步没有接入共享内存、锁、预取线程或 CUDA 异步搬运，没有修改训练入口/默认值，也不代表并发安全或提速已经完成；这些属于后续 6.2–6.5。

### 原始轨迹共享通道（第 6.2 步）

- `ipc.shared_tensor_channel.SharedTensorChannel.create(layout, slots=2)` 提供 typed、bounded、lossless 的单生产者/单消费者通道；布局由 `fada.shared_collection.raw_transport_layout(spec, limits, max_rows=...)` 核算。端点通过 `build_official_fada_runtime(shared_endpoint=channel.endpoint, ...)` 的 worker 启动参数注入，不在运行中的请求队列传同步句柄。
- 共享请求显式设置 `OfficialFADACollectRequest.transport_id=channel.transport_id`、`output_path=""`，由 `collect_shared_raw(runtime, request, channel, spec=..., max_rows=..., pending_budget_bytes=..., expected_metadata=prepare.metadata, timeout=...)` 收集；接收泵先于阻塞 RPC 启动。返回值为自有 CPU raw batch 与原 Dagger result，只有尾块、行数、来源/权重版本和 Oracle/预处理身份全部匹配才成功，不提前写入 Replay。
- 共享槽写满会等待；读取是租约，复制后才释放。`abort()` 使阻塞收发退出且通道永久失效；worker 只关闭附件，调用方在结束本地收发并关闭 runtime 后负责 creator 的 `cleanup()`。失败不重置或回退磁盘。不带 transport ID 的现有显式 artifact 请求保持原路径。
- 传输的 `timeout` 覆盖原生采样等待与接收，须高于预期采集耗时；worker 的 `shared_timeout_seconds` 单独限制发送。不能中断任意原生 kernel，既有 Runner 的请求超时/进程组回收继续生效。槽位预算外另有固定元数据/header 开销，创建时会检查共享内存容量；接收暂存区单独受 `max_rows`/`pending_budget_bytes` 限制，不承诺 zero-copy。

CPU spawn 场景已走通真实 Runner/collector 的共享分支，连续两次请求、多于槽位数的分块、partial 尾块与大整数 ID 保真；未生成原始轨迹文件。背压、取消、坏版本与暂存预算拒绝、无接收线程泄漏及共享段回收检查通过。原生资源/物理采样使用明确诊断替身，未跑仿真、GPU或训练。正式训练默认路由尚未切换，接线与默认配置仍由 6.4/6.5 完成；本步通过不代表端到端提速已完成。

### 学习端批次流水线（第 6.3 步）

- `fada.batch_pipeline.OfficialFADABatchPipeline` 显式接收 `sample_cpu=replay.sample_update`、`spec`、`training_config`、`mode`、`planned_updates`、`device` 和 `FADAExecutionLimits`。在 `with pipeline:` 内逐次 `with pipeline.batch() as (idm_batches, planner_batch):` 消费；batch 借用只在租约内有效，不能在退出后长期保存其 tensor 引用。
- 单采样 worker 保持现有 NumPy 抽样顺序，有限阶段只抽计划次数。`prefetch_batches` 是槽位总数，包含正在消费的 batch；预算包含全部保留的 host/device 副本及额外采样/Planner 拼接临时峰值。CPU 使用普通可复用 buffer；CUDA 使用 pinned host、显式设备 buffer、copy stream 和槽位 ready/consumed 事件。保存或修改 Replay 前必须退出阶段，等待采样与本流水线的事件完成；异常取消不伪装回滚已经消耗的 RNG。
- `OfficialFADALearner.update(..., collect_metrics=False)` 仍执行完整 IDM/Planner 更新、裁剪和计数，只返回空指标字典。默认 `True` 保留原报告字段和加权口径，将所需 detached 标量集中转 CPU；正式调度后续可仅在末步/报告边界收集。没有删去零系数 Planner 前向，也没有更改 dropout、增强、精度或来源配额。

第 6.3 步一次 CPU 场景对照了改前 Replay 采样 owner/learner 与新流水线：实际官方小型 Transformer 在噪声、dropout 和裁剪下，窗口、配额、loss、梯度、两 AdamW、权重及 RNG 一致；外部存储是明确的诊断 NumPy 窗口。指标开关、零阶段、预算和异常退出检查通过，无 sampler 线程泄漏。流水线已成为正式入口默认路径；CUDA pinned/stream/event 行为及端到端性能仍需服务器验证。

### 共享执行与配对提交集成（第 6.4 步）

- `run_fada_persistent_async` 必需的 `execution_limits`/`pending_raw_budget_bytes` 由正式入口自动解析传入；无需额外 Python 参数启用，也没有磁盘/顺序执行的过渡分支。配置容量不足直接拒绝，缓存和配对恢复用持久化文件继续保留。
- `PersistentDistillationRuntime.activate_checkpoint(path, state_dict=learner.policy.state_dict())` 使用现有共享权重通道直接发布，不调用推理 loader 或重读磁盘；path 是后续提交目标与请求身份，发布成功不表示该文件已提交。省略 state_dict 的既有文件调用仍可用。
- `Replay.add_raw_batch` 与 artifact 入口共用身份检查、完整 episode 过滤和单事务 ID 重映射。新执行路径所有来源直接入内存事务，不产生 raw trajectory/shard 交接文件；offline cache 与恢复用 NPZ/checkpoint 继续保留。
- 每轮有限学习结束后，先发布最新权重，再由 `FADAIterationCommitter` 保存本轮 Replay/模型；同时采集下一轮到独立暂存区。写盘期间 learner/optimizer/Replay 只读，必须 wait 提交成功后才能入库和学习。writer 最多一个任务，后台错误回传；返回/关闭前等待，未完成的持久化不能冒充成功进度。
- 提交先完成独立 Replay train/val 文件，再用原保存接口原子替换 latest，历史 checkpoint 所引用的 Replay 保留；不修改存储格式、不复制整份 Replay。仅回收本提交器拥有且已无历史引用的旧快照，不删除用户 resume 文件。

第 6.4 步一次 CPU 集成通过 warmup→两轮→恢复，顺序参考隔离文件 loader 随机初始化副作用后，模型、两 optimizer、计数、Replay 大小和指标一致。确定性 writer gate 证明采集可重叠、Replay 入库等提交；提交失败保留旧配对和历史引用。第 6.5 步随后从 Hydra 配置经实际脚本 owner 默认运行同一链路，与 6.4 显式路径一致，并通过 strict 恢复、预算提前拒绝和失败关闭检查；预算未进入保存的算法/source 身份。两步均使用明确诊断 native/外部存储依赖，不是原生/GPU或策略质量验收。

服务器验收由操作者执行，不自动下载、连接或启动训练。保持同一 GPU、官方权重/初始学生、环境数、H/K、batch、source 配额、更新次数和 cache 口径；分别比较冷启动与稳态、实际 transition/update 数、端到端耗时和 RAM/VRAM 峰值。复用已有可比基线，不为“更放心”重跑；没有同工作量改善证据就报告“未证实提速”，不能仅靠局部阶段或 CPU 诊断宣称加速倍率。
