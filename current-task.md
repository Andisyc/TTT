# Current task

Goal: 使用官方公开 G1 PPO Oracle，经 TTT 正式 official_fada 链路训练 IDM 与 Planner，保留官方网络、学习和采集语义。
Do not: 不重训 Oracle，不用旧 v024/66维学生替代官方权重，不改研究方法、生产算法或原实验；不将启动/短链路通过写成收敛或策略质量通过。

Latest delivery (2026-10-31, 用户授权完成 sim2sim):
- 已将最终 `student_iter_0030.pt` 接入正式 TTT MuJoCo session/recorder。选择 `task=g1_walk_flat/mujoco_fada_official`，绑定原 run 的 `acceptance/student_source_profile.json`；200 Hz 仿真、50 Hz 控制、1 Hz 连续相位，按关节名在模型初始化/materialization 时绑定全精度 PD，保持源动作目标映射和站立相位语义。普通 task 的原始 XML PD/时步零步核对通过，未更改共享机器人 XML。
- 修复 G1 三个公开转发接口，以及 MuJoCo backend 误将身体坐标 free-joint angular qvel 作为世界角速度返回的缺陷。零步 yaw=90° 对照证明原接口令官方 adapter 发生重复反旋转；修复后唯一新增回归 `test_mujoco_base_angular_velocity_matches_world_object_velocity` 对照真实 `mj_objectVelocity` 的三个姿态通过（1 passed，0.22 s），不修改物理状态。
- 有界结果：修复角速度前的直行 20 s/1000 步未终止，vx/vy/wz RMSE=0.11581/0.10845/0.08999；首轮八字在 6.94 s 跌倒，原视频/轨迹/日志留在 `sim2sim/body_frame_failure/`。仅修复角速度后，同权重/seed=42/4×2 m/.3 m/s 八字完整 41.6 s/2080 步通过，无终止/重置，观测/动作有限；RMSE=0.08645/0.10918/0.08469，109 个真实落脚点，回到起点误差 0.63590 m。是开环速度指令 demo，不宣称精确轨迹跟踪或多 seed 鲁棒性。
- 证据/视频/已解析配置：正式 run 的 `sim2sim/`（本地已同步）；`figure_eight.py` 仅为调用正式 factory/recorder 的固定 seed 有界观测脚本。首轮直行脚本因清理误调用 session.close 未保存指标，保留日志并修正为 wrapped_env.close 后补跑；本轮含该段最多 88.54 agent-seconds。未重训、改权重、执行全套测试、创建分支或提交/推送 Git。

Current identity:
- 正式续训：`/ssd1/cyx/TTT/logs/fada/official/20261030-g1-idm-planner-fast-vNmpNY`；日志 `train.log`，tmux `ttt-fada-g1-official-fast`，启动器 `.tmp/official_fada/run_training_resume_vectorized.sh <run_dir> <checkpoint> <previous_root>`。
- 恢复来源：`/ssd1/cyx/TTT/logs/fada/official/20261030-g1-idm-planner-8HIivb/20261030_130149_443804_g1_official_full/checkpoints/student_iter_0009.pt`，9/30轮、11000更新，online replay为同目录`online_buffer_0010_iter_0009.npz`。原run是用户授权SIGINT停止，exit 130；Runner及shadow已退出，原日志/checkpoint保留。
- 原生短链路：`/ssd1/cyx/TTT/logs/fada/official/20261030-native-probe-CGCdFh`，exit 0；optimal/suboptimal/online 各1024条 raw transition，两组 learner 各4次 optimizer step，保存官方 schema v2 的64/29/7、H30/K6学生。
- 官方产物：`AngchenXie/fada-checkpoints` 的完整251个G1 checkpoint（约2.01 GB），逐文件校验SHA-256；最终 `model_24999.pt` SHA `97faea45885c71825f37fff8669604af1aed25500050fbc9c2866d0e03ef4261`，原始配置内嵌于checkpoint。
- 运行环境：`.tmp/official_fada/runtime-pinned`，优先使用原bydmic的Python3.11/IsaacSim5.1/Torch2.7.0+cu128/NumPy1.26.0；缺包装在独立overlay，TTT已有发行包仅作末尾路径回退。原两个环境未修改；较早被中止的 `.tmp/official_fada/runtime` 不用于训练。
- 正式配置：`--config-name official_fada`，只将环境数设为256并采用180秒shadow超时；保留batch4096、warmup2000、DAgger30轮/每轮1000更新及20个弱checkpoint来源。CPU限定0–27、BLAS/OpenMP各4线程、GPU0；性能计时开启，不宣称加速。
- 当前任务状态（2026-10-31）：正式训练 exit 0，累计30轮/32000更新；最终checkpoint严格恢复与有限性通过。原生学生/Oracle同初态直行各20秒通过，学生vx RMSE 0.08158m/s，Oracle 0.03794m/s。MuJoCo原接口阻塞及转弯角速度坐标系错误已修复，最终学生完成固定 seed 的 41.6 秒整圈八字。原验收保留于 acceptance/，后续迁移证据在 sim2sim/；原生与跨后端结论分别记录。
Goal: 将官方 FADA 的 G1 Locomotion Oracle→Planner–IDM 源训练接入 TTT，复用 UniLab 的进程/调度基础设施，产物可供现有播放与在线适应使用；接通后测量并优化吞吐。
Do not: 本地实现与CPU验收已交付，不自动连接服务器、下载/安装、运行原生仿真/GPU训练或benchmark，不操作Git/机器人。保留模型/loss/采样与更新量、strict-shadow、最新策略和checkpoint/cache契约；不新增AMP/compile或policy lag。未经用户服务器正确性与同工作量性能证据，不关闭整体原生/加速验收，不以CPU结果宣称GPU倍率。

Current identity:
- 当前任务状态：6.1–6.5本地代码与对应CPU场景已完成。官方Hydra/脚本owner默认使用共享收发、finite pipeline、内存发布和单flight提交；过渡磁盘/顺序分支已删除。预算解析、warmup/两轮、strict恢复及提交失败关闭通过。现有服务器进程未热切换；CUDA、原生shadow和实际端到端性能仍待用户验证，无已证实加速倍率。
- 官方参考：LeCAR-Lab/FADA，commit `290b3a9c5c0ebe2e9851b61350617544be05e9bc`；若实施前版本变化，只按固定版本实现，升级另作明确决定。
- 本地入口：`scripts/train_distill.py:run_fada_training`；配置组 `conf/distill/`；FADA owner 为 `src/unilab/algos/torch/distill/fada/`。
- 历史实验：v24 checkpoint 为 66/29/3、H=30/K=6，仅保留既有实验用途，不当成官方学生权重或默认可转换产物。
- 上一任务已关闭：`expert_trajectory_difference` 在线链路已完成离线工程验证，真实负重效果未验证；保留 README 现有操作说明及已有测试证据，不把旧结果迁移为新基座成功证据。

Confirmed decisions:
- 用户范围：仅接 Locomotion；采用 workflow-governance 轻量记录、system-engineering 总体路线、engineering-plan 逐步细化，Multi Agent 分担独立切片。
- 最新执行边界（2026-10-30 用户授权）：已明确要求连接服务器、寻找公开原始 policy 并开始官方 IDM/Planner 训练；替代此前仅本地接入阶段的服务器禁令。授权不扩展到重训 Oracle、真机、改方法或覆盖原结果。
- 最新续训授权（2026-10-30）：用户明确要求在已保存checkpoint上恢复启动。已部署仅Replay批量组窗优化（`replay.py`及`official_replay_sampling.py`），不改采样量/源比例/学习规则。原parent Replay备份`.tmp/official_fada/replay-optimized/replay.before.py`，无Git提交/推送。现有schema v2不保存完整RNG/仿真现场，恢复是轮次边界续训，不宣称逐步随机轨迹一致。
- 整合边界：官方负责网络/数据/学习语义；UniLab 负责唯一训练入口、任务/进程生命周期和权重交接。不引入第二个完整训练主循环或另一套通信协议。
- 源仿真先保持官方环境和 Oracle 条件。优先把官方采集包装成现有 `DaggerCollectorService` worker；不先移植为 MuJoCo，不伪装成已有 UniLab backend。部署端与源训练端分别校准。
- 当前本地 persistent_async 路径是“等采集→训练→发新权重”，已有子进程不等于采学重叠。先保留轮次语义，性能优化不得悄悄引入陈旧策略采集。
- 接口不按维度硬凑：官方 compact observation 的字段/缩放与 7 维命令（含步态相位），不同于现有 g1_fada_state_v2 66 维状态/3 维命令；Oracle 的 actor 特权输入与学生 compact 输入是两套接口，不能混用。
- 学习语义更正（第3步源码复核）：官方输出K步动作，但默认 `idm_action_loss_horizon_gamma=0.0` 使损失只监督首帧；不能把K步输入/输出等同于全K帧均匀监督。Planner未来状态监督系数默认0，动作误差经冻结IDM反传。先前“官方默认监督全K帧”的记录有误，以固定源码config.py及weighted_horizon_mse为准；不擅自更改horizon权重。
- 现有在线研究方法保持不变：冻结 Planner 与独立 base_idm；用相同历史计算 base(X,Y_nom)-base(X,Y_exec)，叠加保存的 U_snapshot 形成静态目标；执行 IDM 只更新 LoRA，compute_loss 仍以 Y_nom 为未来条件。保留原 reset/批次/锚定/Adam 语义，不借接入改方法或调参数。
- 新基座以明确模型/观测/schema 身份保存，不覆盖旧 checkpoint、旧 run 或已有数据；不兼容即明确拒绝，不截断/补零/静默 strict=False。保留旧实验播放能力不等于维护两套正式源训练主循环。

Progress:
- [x] 完成总体路线与并行边界规划。
  - Evidence: 直接读取本地 workflow_setup、observation、入口和 runtime 接口；两个只读 scout 分别检查官方/本地采集与 learner；直接核对固定版本 compact_obs。未执行任何运行验证。
  - Multi Agent：A=采集/Oracle，B=模型/学习；主 Agent 负责接口、Runner、公共配置/schema 和本板。独立切片可并行，共享文件由主 Agent 单写。子 Agent 不跑测试/格式化/训练，主 Agent 合并后做每步一次针对性证明。

## 总体路线与逐步 engineering-plan

后续每步开始时，只细化当前包：补齐实际符号/调用者/允许文件和唯一检查；不重新全仓探索，不提前为后续步骤写详细伪代码。第 1 步的接口结论冻结后，第 2/3 步可并行，其余按依赖推进。

- [ ] 1. 冻结官方输入、输出与数据来源（已执行，部分完成；待绑定官方PPO产物）。
  - Owner / scope：两个只读scout分别核准输入控制端和时序数据端，主Agent汇总并唯一执行离线对照；未修改生产代码/配置、旧checkpoint或在线差分方法。
  - 已完成源码校准：固定官方commit `290b3a9c5c0ebe2e9851b61350617544be05e9bc`；不把本地SAC/APPO或普通98维actor混同于官方Oracle。下表为源码基准，不代表未取得的实际run overrides。

  | 对象：生产者→消费者 | 已核准语义 / 来源 | 本地接入位置与状态 |
  |---|---|---|
  | Student compact：官方obs terms→Planner/IDM | 64维按 `base_ang_vel(3)*0.25, (q-q_default)(29), dof_vel(29)*0.05, projected_gravity(3)`；角速度/重力在base系，世界重力[0,0,-1]。`compact_obs.py`＋`managers/observation/terms/locomotion.py`。纠正早期判断：官方关节观测也是相对角，不是绝对角。缩放实际应从选定Oracle配置解析，不能只硬编码默认值。 | `observation.py`/官方采集adapter，不能直接复用历史66维投影；语义与临时映射已离线对照。 |
  | Task command：command manager→Planner | 7维=`vx,vy,wz,sin(left/right phase),cos(left/right phase)`；速度3＋相位4，零速度不等于相位4维清零。`common/current_command.py`。 | 原3维command接口需明确扩展；尚未修改生产实现。 |
  | Oracle actor：专用observation manager→PPO | 有序11字段：base_ang_vel、base_lin_vel、dof_pos、dof_vel、actions、projected_gravity、oracle_actuator/contact/terrain/command_gait/randomization_compact_state。`g1_29dof_loco_oracle`，不是普通locomotion98维preset。源码PPO empirical_normalization=False、FADA normalize_io=False；实际权重身份/normalizer仍须随原始run核对。 | `oracle.py`不能走现有SAC专用loader；真实官方PPO checkpoint与配套配置尚未绑定。 |
  | Actions：policy→环境；历史/标签→Replay | 源码默认P控制、clip ±100，q_target=q_default+0.25*executed_action；29关节顺序为左右腿各6、腰yaw/roll/pitch、左右臂各7。初态与各关节PD使用固定官方`config_values/robot.py:g1_29dof`完整精度配置，不手抄近似值；ctrl_dt默认0.02s（200Hz/decimation4）。实际run覆盖值未绑定。`executed_act`含探索噪声后clip的真实下发动作，expert_act/chunk为clip后的Oracle动作；不是力矩。 | `collection_contract.py`及worker适配；源码边界已明确，未验证物理执行。 |
  | Histories：Replay→Planner/IDM | 官方默认H=30/K=6（`planner_idm/config.py`，与历史本地值碰巧相同但来源独立）；hist_obs止于t，hist_act对应各观测前一步、最后止t-1；冷启动左补零＋valid_mask，不是重复初态。 | `windows.py`/`collection_windows.py`及model/checkpoint identity需要承载mask；原始Replay实测通过。 |
  | Future/action targets：shadow或真实trajectory→loss | 真实动作t..t+K-1配未来观测t+1..t+K；strict expert chunks与trajectory targets字段分离，真实未来不足时trajectory_target_valid=False，不得把零填充当标签。缓存按(env_id,episode_id)隔离。严格标签有效性须沿用producer的strict_label_valid/branch-done结果；无官方role0/1枚举可直接照搬。 | `source_artifact.py`/replay reader/采集adapter；原始Replay的组窗、mask、invalid-tail和reset隔离实测通过；原生shadow状态恢复尚未执行。 |
  | Artifact identity：真实run→loader/worker | 必须绑定官方source commit、已解析配置、actor/compact/command/action schema、normalizer、H/K/mask、source角色与权重版本；不兼容即拒绝，不能strict=False或补零截断。 | 第4步已切至官方raw source v8及独立学生checkpoint v2；真实Oracle产物绑定仍待补齐。 |

  - 唯一离线对照：`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt_fada_step1_calibration_290b3a9.py`，一次执行PASS。临时脚本直接执行固定官方的观测/指令函数原始函数体（隔离仿真依赖，不替换函数体）和完整原始 `ReplayBuffer.add_batch/sample_chunk`；对照独立命名状态映射。没有导入/启动IsaacSim，没有用自写reference替换上游实现。
  - Evidence：不同关节非零值＋非零默认角＋非单位姿态下，float32观测最大绝对误差`5.960464477539063e-08`，command误差0（atol/rtol各1e-6）；同次对照检查8个正常/冷启动/reset锚点，obs/action历史止点、左补零mask、future/action首索引、shadow/真实目标分离和不完整轨迹invalid全部通过。H4/K2只是诊断fixture，未冒充官方默认。临时脚本已删除，未新增永久测试或流程产物。
  - 实际产物只读核实：`FADA_Unofficial/model/oracle_official_appo_seed1/appo_final_update_5000.pt` 经 `torch.load(map_location="meta",weights_only=True)` 读取metadata；schema=`fada_official_appo_oracle_v1`、algorithm=`fada_official_appo`、backend=`mujoco`、actor/critic322、student66/action29/command3、obs_normalization=False。虽source_commit同上游，但这是本地APPO迁移分支，不是官方发布PPO/64+7学生链路；不擅自替换本轮目标。相关owner在相邻FADA_Unofficial仓库，TTT未发现该official owner，未移植或修改。
  - 证据范围：未构建/运行官方Oracle、未验证物理动力学/原生shadow、未证明正式adapter接通或策略质量；离线Replay使用人工标明的episode/strict有效性，因此不把它当成真实终止producer验证。原始官方checkpoint/config尚未本地读取，不能把源码默认当作实际run值。
  - Done when：源码表与离线对照已完成；真实官方PPO checkpoint及配套配置/normalizer身份核准后才能整步勾选。第2/3步可依据已核准契约继续规划，但不得悄悄以本地APPO产物代替官方源仿真路线。

- [ ] 2. 接入官方源采集与 Oracle（代码接入已完成；2D原生采集未验证）。
  - Change / Why：在既有UniLab采集服务内包装官方环境、PPO Oracle和影子标注，保留官方源条件。不是把官方PPO权重塞进本地SAC/APPO loader，不迁到MuJoCo，不调用官方完整训练主循环。
  - 主Agent先冻结最小共享接口：资源包提供初始化后的官方env、Oracle推理、strict-shadow服务及close；采集包接受这些资源和真实student推理接口，产出带明确schema/来源身份的采集artifact。资源/采集两包不直接修改彼此或公共文件；字段名称、形状、mask及错误行为在派工前由主Agent给定。
  - [x] **2A 资源与启动（OfficialResources Agent实现，主Agent集成）**：已新增 `fada/official_source_runtime.py`，真实调用固定上游FADAConfig、build_env_and_expert、StrictLabelWorkerClient、build_student_policy和FADATrainer采集helper；optimizer/lr_scheduler均为None，不调用warmup/run_loop。缺原始配置即失败，官方依赖仅worker内导入；共享权重键/形状在复制前严格匹配。主Agent修正预处理helper属于上游train.py而非trainer.py、禁止覆盖产物、允许无关日志/docs而拒绝上游源模块变更；close释放模型引用并处理部分初始化。
  - [x] **2B 采集事务（OfficialCollection Agent实现）**：已新增 `fada/official_source_collection.py`，三种source调用真实上游_collect_with_policy，事务使用独立组窗buffer；online含配置噪声、optimal由最终Oracle驱动、suboptimal由弱权重驱动且strict worker保持最终Oracle，finally恢复最终expert。主Agent要求移除额外64轮补采/禁止replacement策略，现每事务只采一次配置rollout，随后沿用原生sample_chunk(num_windows)。不造数据、不创建optimizer/App/进程，保留历史mask与真实trajectory有效性。
  - [x] **2C 公共接线（主Agent）**：`collection_contract.py`新增OfficialFADACollectionSpec/OfficialFADASourceBatch；同一`source_artifact.py:FADAShardedSourceWriter`支持显式官方schema与独立reader（第3步由v6升至含真实预处理的v7），沿用原子提交、分片身份校验，旧v5 reader/行为保持；`fada.async_runtime.py:build_official_fada_runtime`复用PersistentDistillationRuntime、共享权重与Dagger请求/结果，不引入新Runner或IPC。正式训练CLI未切换（第4步）。
  - 公共payload：保存官方 `history_obs[B,H,64]`、`history_act[B,H,29]`、`history_valid_mask[B,H]`、`current_command[B,7]`、`target_actions[B,K,29]`、`target_next_observations[B,K,64]`、`trajectory_target_actions[B,K,29]`、`trajectory_target_next_observations[B,K,64]`、`trajectory_target_valid[B]`，按source声明哪些目标可用于哪些loss；头部绑定request/iteration、source角色、官方revision、Oracle/config/观测/动作身份与observed_weight_version。来源追踪必须来自真实采集锚点，不从浮点状态反推。writer/reader对新schema显式校验，旧数据不被误认新数据。
  - 有效性规则：strict-shadow标签沿用官方strict有效性，真实轨迹未来不足时仅标trajectory_target_valid=False；不得因此误删仍合法的shadow样本，也不得把补零轨迹送入真实动力学损失。Reader在相应source/用途上筛选，禁止用未定义的role0/1猜映射。
  - 已实现进程接缝：`AsyncRunner`新增可选collector_daemon（默认True不变），经PersistentDaggerCollectorRunner/PersistentDistillationRuntime传递；官方factory设False。POSIX worker独立session，正常关闭先走服务close，超时/异常由父Runner清理整组子进程，request-scoped退休路径也复用清理。修改前已用LSP核对共享接口调用者；未创建第二个Runner。
  - 生命周期/错误：父进程不启动官方仿真；worker独占资源；正常stop与异常均先通知/等待shadow退出，再关闭其IPC及env/App。超时/中断沿现有Runner guard传播并回收资源，不能只把collector改为non-daemon就忽略孙进程。失败抛异常，由已有guard传回父进程，不伪造success=False或半成品DaggerCollectResult；仅在完整artifact落盘并校验后返回成功结果。
  - Preserve / Forbidden：不改官方奖励/随机化/控制参数、模型loss/网络/在线目标或默认超参数，不运行完整DAgger/warmup训练，不覆盖旧run/artifact；工厂显式选择官方source身份，不静默改变旧配置含义。旧源训练的最终切换/移除在第4步处理，不承诺永久保留两条正式训练链。
  - [ ] **2D 唯一官方原生采集验证**：尚未执行。需要绑定官方PPO产物、官方运行环境和真实学生模型后，经正式Dagger请求做一次有界采集；核对reader、实际shadow前后状态/RNG、真实reset、动作/future语义、权重版本与原生进程释放。离线进程/存储smoke不替代此项，也不证明学习效果/加速。
  - Done when：2A/2B接线、显式schema与生命周期全部实现，且2D有实际证据；之前只标子包完成，不标第2步整体完成。第1步离线sample_chunk结果不替代此处真实采集。该验证不评价长期策略效果或稳定提速。
  - Depends on：第1步剩余的官方PPO checkpoint/config身份核准；可用的官方运行环境；第3步真实student模型/权重加载接口。2A初始化及2B源数据处理可按已确认契约独立开发，但完整学生闭环与2D不能在第3步前用假student验收。下载/安装、目标机器访问与真实采集须有各自授权，本轮均不执行。
  - 集成Evidence：主Agent执行 `PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt_official_source_step2_smoke.py`。首次在产物commit前因测试临时目录未resolve真实路径而触发身份拒收；仅修正fixture为Path(directory).resolve()后重跑失败及尚未完成的场景，PASS。九字段往返完全一致、无效真实尾段仍保留合法shadow；坏mask/错误source/版本/格式混用/分片篡改均被拒，旧v5读写保持；真实SharedWeightSync发布版本1→2，嵌套spawn计算值31→62，collector/子进程关闭后均退出；真实官方factory在缺checkout时通过错误管道明确失败，未导入native或生成成功artifact。最后一条预期缺依赖错误有进程trace，不是隐藏的成功fallback。
  - 验证范围：未跑测试套件/formatter/独立审查。子Agent曾做一次局部导入/AST检查；主Agent集成smoke之外，仅因发现close仍持有模型引用而补做一个针对性弱引用检查，确认正常/部分初始化close幂等并释放实际torch模型对象；未重跑已通过集成场景。临时smoke脚本已清理；README记录API/前置条件和证据边界，无新流程文档。

- [x] 3. 接入官方网络与学习规则（代码及唯一CPU数值对照已完成）。
  - **3A / OfficialStudentModel Agent**：新增 `official_model.py`，移植固定上游纯Torch Planner/IDM和位置编码；`build_official_fada_student`保留原生键/形状、pre-Norm、delta/mask/历史命令flag，完整K步输出。默认64/29/7、H30/K6，不经safe_torch_import或native模块。
  - **3B / OfficialStudentLearner Agent**：新增 `official_learning.py:OfficialFADALearner`，保留默认增强/掩码、horizon权重、两个AdamW与每批IDM→Planner；Planner反传时IDM参数固定，之后恢复。真实分支先过滤无效行，但source加权仍用原始行数。主Agent去掉重复序列化的派生来源开关，绑定模型共享配置，并接通计数恢复。
  - **3C / 主Agent**：`replay.py:OfficialFADAReplayBuffer`与`replay_sampling.py:official_replay_source_counts`接收九字段窗口，保留三种物理来源、五种IDM用途、offline/mixed配额、Python取整、Planner先于IDM采样与原始replacement规则。完整原始轨迹保留/淘汰分布仍在第4步处理，不以minibatch一致替代整轮一致。
  - **预处理身份 / 主Agent**：worker metadata导出Oracle run实际解析的compact scale/noise；`source_artifact.py`官方schema升至v7，缺值的旧v6产物拒收，旧v5不变。learner必须显式绑定预处理，不从hash反推或套旧66/3默认值。
  - **3D / 主Agent**：`checkpoint.py`增加官方学生保存、严格学习恢复与`load_official_fada_student_checkpoint`纯推理loader；完整模型/训练/预处理、原生state_dict、两optimizer与更新计数持久化。独立格式拒收旧学生/PPO/APPO，不加权重hash或另一manifest；loader可供既有runtime使用。
  - **3E Evidence**：主Agent统一执行一次CPU原代码对照场景（`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-official-step3-1793209402180/verify.py`）。首次仅因fixture把源码中待推断的obs_dim=None当作64默认值而停止，修正该断言后场景PASS：真实上游ReplayBuffer生成窗口与采样顺序/配额相同；两组位置编码/delta/历史命令flag前向相同；默认噪声、掩码、dropout及实际触发裁剪下的weighted loss、IDM→Planner权重与AdamW状态逐项相同；gamma0后续动作标签不改变梯度；strict恢复两optimizer/计数及SharedWeightSync通过。未启动任何仿真或完整训练。临时reference与脚本在交付前清理。
  - **Preserve / 边界**：历史66/3模型/旧checkpoint/在线LoRA方法未修改；正式CLI、完整warmup/DAgger循环、播放适配与提速仍分别属于第4–6步。子Agent仅写各自模块并做局部静态检查；未跑测试套件、formatter或独立审查。第1/2步真实产物和原生采集缺口不随本步关闭。

- [ ] 4. 接通唯一正式训练入口（代码与离线验证完成；待原生验收）。
  - [x] 共享接口：同一Runner队列增加显式prepare；初始化学生仅注册共享布局，实际Oracle metadata返回后才建立learner、保存并发布真实checkpoint。阶段请求显式区分optimal/suboptimal/online，采集输出13字段原始transition（source v8）；旧v5及旧学生播放不变，官方窗口v7不冒充raw。
  - [x] 采集owner：`official_source_runtime.py` / `official_source_collection.py`复用同一native env/Trainer/fixed Oracle shadow，取消worker提前抽窗口；optimal按horizon分次采集，弱策略轮换至官方目标预算。实际native路径尚未运行。
  - [x] Replay/外循环owner：`replay.py` / `persistent_workflow.py`复用原始NumPy ReplayBuffer，跨shard完整episode判断、轨迹抽取/划分、跨轮online累积、整episode淘汰与动态sample_chunk；完整offline缓存、train/val保存恢复、两optimizer/采集计数配套。第一轮online前发布warmup权重，每轮发布不受周期快照间隔影响；恢复不重跑warmup。
  - [x] 配置入口owner：`scripts/train_distill.py`、`workflow.py`、`workflow_setup.py`、官方runtime factory、checkpoint与`conf/distill/official_fada.yaml`接通；唯一正式配置为`--config-name official_fada`，去掉旧SAC/66/3源训练装配及废弃legacy外循环。原始学习默认值与固定官方FADAConfig对照一致，不增IO normalization；旧源入口明确拒绝，非FADA蒸馏及在线LoRA语义不变。
  - [x] 近端离线集成：真实入口及真实persistent Runner/共享内存，使用明确标识的诊断轨迹完成prepare→offline/warmup→两轮DAgger→发布→第三轮恢复。核对原始13字段多shard往返、完整episode保留、原版轨迹split/random淘汰、动态采样、NPZ恢复、warmup后首轮权重及每轮策略输出改变；save_every_iter=2仍逐轮发布，恢复不重复warmup。另验证缺checkout由已有错误通道回传且不生成假checkpoint。
    - Evidence: `PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-official-step4-1793212935353/verify.py` 的Replay与完整workflow段通过；只继续未完成的`entry failure`段后全部通过。初次诊断脚本的文件名碰撞及错误入口开关已修正；未因此改动生产逻辑。临时诊断脚本/参考数据清除，不新增永久测试或报告。
  - [ ] 最终Done when：绑定真实官方PPO、弱checkpoint历史及保存配置，在获授权的Linux/CUDA/IsaacSim环境由正式入口完成有界offline/warmup和两轮DAgger，核对发布版本、实际raw累积、严格恢复及worker/shadow回收。离线证据不替代该原生验收。
  - Multi Agent：主Agent先冻结共享接口，再并行A采集端、B Replay/外循环；主Agent完成入口/持久化集成与统一离线验证，无独立审查步骤。未进行采学重叠、提速、物理播放或策略质量实验。

- [ ] 5. 接回播放与现有在线适应（5A–5D代码与CPU验证完成；真实原生播放验收待前置）。
  - [x] 5A 共享接口：新增纯Torch runtime spec/原生调用接缝；明确format先于数字schema分流，官方strict loader与旧/适配模型分支分开；snapshot携带64/66契约与bool mask。
  - [x] 5B 官方播放：新增官方64/29/7 controller；物理状态投影、完整相位command、source clip和首动作快照、左补零/mask及partial reset已接通，旧controller不变。
  - [x] 5C 在线接口：collector/updater按runtime spec接收command/mask；冻结base双路差分与当前IDM回归使用同一原生接口。相位变化不再清空K步待成熟轨迹。保留LoRA层发现、固定标签、loss、窗口、Adam/锚定及reset保留已学参数规则。
  - [x] 5D 正式session：factory按明确模型身份选择controller；显式source-bound profile在创建env前核准，公共物理接口负责观测与source→target动作映射，检查PD/dt/相位契约。缺失profile不回退旧task；无新增在线checkpoint格式、env/backend业务逻辑或RoboJuDo_Real改动。
  - Evidence：主Agent统一CPU场景3项PASS（本轮终端）：非单位缩放/非零默认角/关节重排/相位、原生前向与不可变快照；正式session完整K步成熟、实际LoRA-only更新、Planner/base不变及reset保留Adam；旧schema2播放/更新和官方format分流、缺失/错配profile在创建env前拒绝。首次场景发现partial reset泄漏上个episode动作，修复后同场景通过。诊断transition不是物理仿真；临时脚本/模型已清理。
  - Done when / Depends on：代码实施无需重做；完整验收仍需第1步真实官方权重/run配置绑定、第4步相关原生前置和运行授权，再通过正式session关闭/开启更新的真实短运行。当前不宣称sim2sim、负重效果或加速。

<<<<<<< HEAD
- [ ] 6. 测量并实现有依据的训练提速（6A完成；6B–6D待运行前置及实测）。
  - **目标 / Preserve**：在相同硬件、官方checkpoint/初始学生、配置、env/rollout/batch/更新量、source比例、strict-label、采样顺序/seed、cache状态与权重版本条件下衡量端到端吞吐。区分官方原生→TTT的整合收益与TTT改前→改后的优化收益；不把不同工作量或缓存命中当算法加速。
  - [x] **6A 测量接线**：新增轻量`fada/timing.py`复用已有StageObservationAccumulator；`workflow.py`与`conf/distill/official_fada.yaml`提供默认关闭的`training.fada.performance.enabled`，不进入原生算法配置/缓存/checkpoint。`persistent_workflow.py`汇总worker耗时，计量初始化、RPC、artifact/Replay、batch、学习、保存/发布和close；`replay.py`记录原始sample与H2D。结果在warmup/每轮/close后由现有日志输出，最终返回performance，无新进度/报告/哈希体系。
  - [x] **6A Multi Agent**：主Agent先落地共享计时接口，再并行A采集端（`fada/async_runtime.py`、`official_source_runtime.py`、`official_source_collection.py`）和B学习端（`official_learning.py`）。A计量资源初始化、已完成的rollout/CPU打包，并保留原metrics；B计量IDM→Planner前向/反向/optimizer。主Agent集成Replay、配置、正式外循环与统一验证；未设review席位，子Agent仅做局部编译/AST检查。
  - [x] **6A唯一离线Evidence**：`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-timing-1793242829867/verify.py`一次运行PASS。计时OFF/ON经过正式owner/factory、真实resident worker prepare/collect、采集事务/打包、raw v8、固定官方原始NumPy Replay、真实Torch learner、checkpoint及SharedWeightSync；仅native资源bootstrap替换为明确CPU诊断数据生产者，不冒充PPO/物理仿真。warmup+两轮DAgger共3次更新、128条online transition，模型/两AdamW/loss/RNG/持久化轨迹/发布版本逐项相同，collector平面metrics、实际计数与close后汇总通过。临时源码/诊断文件清除，不新增永久测试或报告。
  - **6A口径**：默认关闭时不创建timer/CUDA event、不新增tensor或取CPU标量；开启时CUDA event只在报告边界统一解析，原生worker只报已完成事务wall time。parent RPC/worker/rollout属于包络与子阶段，不重复相加；未测阶段不填零。累计输出区分raw transition、sample window与update；total_elapsed覆盖训练owner初始化至close，不含此前CLI/Python导入。原生GPU计时尚未验证，CPU诊断耗时不当成吞吐基线。
  - **本轮局部优化与已授权恢复（2026-10-30）**：原运行实测每轮采集约144秒、batch构造161秒（其中原sample约150秒）、learner约232秒；只将官方逐样本组窗替换为NumPy批量索引，继续使用官方有效锚点/cache/np.random.choice。一次离线对照222组覆盖逐字节输出、RNG、冷启动/reset、fallback/strict、尾部、K1、FIFO/random eviction、clear/load/split；Mac上128000转移/batch4096稳态组窗48.6ms→14.5ms（3.36倍局部，不是端到端或GPU收益）。恢复启动唯一核验：新进程发布的完整checkpoint与原9轮checkpoint递归精确一致，两组AdamW分别80/42状态均为11000步；实际第10轮已开始。尚未取得恢复后首个完整轮次的端到端计时，不宣称官方对照完成。
  - **6B 有界基线测量（主Agent；依赖6A、4+5验收前置及运行授权）**：使用固定版本官方原生入口作为reference，和当前TTT正式入口串行运行同一已核准工作量；不改官方训练主循环、不在TTT建立第二条正式训练路线。执行前固定具体设备、预算、初始模型/optimizer、env数、horizon、warmup/DAgger更新量、cache与日志开关及独立输出目录；分别记录冷启动、稳态、实际transition/update/source配额、版本与阶段占比。该TTT结果作为后续候选的改前基线，不重跑已取得基线。Done when：能够指出哪个阶段实际主导端到端时间及收益上限；没有官方原生对照时不能声称相比官方加速。
  - **6C 条件式Multi Agent优化（依赖6B，测到后才冻结具体工程小包）**：只对有实测占比的独立热点派工。候选A拥有`official_source_collection.py`/`official_source_runtime.py`/`source_artifact.py`的采集打包或I/O冗余；候选B拥有`replay.py`/`official_learning.py`的staging/复制或学习同步冗余。公共schema、正式入口、checkpoint/Runner发布及共享buffer均由主Agent单写；若热点只在一侧就只派一位，若热点耦合就合并，不为填团队而派工。A/B开始前必须冻结实测问题、精确符号、输入/输出、独占文件和等价性判据；两者不依赖对方未来产物且主Agent有集成工作时才同批派出。子Agent不跑GPU性能任务、测试套件或review，不再派助手。本轮不预选compile/AMP/关闭shadow/缩减样本或更新、跨轮prefetch/采学重叠；后两者涉及版本/RNG语义时单独决定。
  - **6D 集成与对照（主Agent；依赖已接纳6C）**：按改动边界统一做一个近端等价场景，核对原始trajectory/样本顺序、mask/source配额、loss与梯度/optimizer更新、checkpoint恢复及下一轮发布版本，不默认重跑全套。再在同一GPU串行执行一次候选固定工作量，与6B既有基线比较端到端时间、phase占比及吞吐；不能只报局部kernel改善。GPU比较由主Agent独占资源，模型协议/RNG和数值容差在选定优化包时明确，不默许精度下降。
  - **Done when / Stop**：有固定工作量、身份一致的实际对照和等价性证据，只有端到端实测改善才保留优化并报告倍率；无改善撤回候选，记录未取得加速，不降低工作量或自动追加训练。短时吞吐不替代策略质量；6B仍依赖原第6步的4+5前置及单独运行授权。6A完成不等于热点优化/性能对照完成，未预选compile、AMP或采学重叠。

Blockers:
- 当前迁移无阻塞；固定 seed 的 MuJoCo 八字通过。尚未验证多 seed、其他指令/地形、负重或 MuJoCo Oracle 同条件对照；未做官方/TTT固定工作量性能对照。这些不在本次有界迁移范围。

Next:
- 本次 sim2sim 交付完成，等待用户下一项需求；不自动重训、调参或扩展仿真/测试。fada heartbeat 保持暂停。
=======
- [ ] 6. 官方FADA接入UniLab高吞吐数据与执行架构（6A及6.1–6.5本地实现/CPU验收完成；原生GPU与端到端性能证据待用户）。
  - Preserve：同工作量、同原始13字段/source/episode/strict标签、同采样与随机调用顺序、同loss/两optimizer更新和每轮最新权重。保留唯一正式入口、Runner/shadow生命周期、旧学生播放和持久化恢复；不移植APPO算法或新增第二个完整训练循环。
  - [x] 6A 已有计时与CPU等价证据：正式入口、resident worker、raw v8、NumPy Replay、真实Torch learner/两AdamW、checkpoint与共享权重的warmup+两轮DAgger，计时OFF/ON的模型/loss/RNG/发布轨迹一致（本轮前终端证据，临时脚本已清理）。现有`[FADA PERF]`继续复用；GPU端到端性能未验证。
  - [x] **6.1 冻结加速接口、内存预算与执行边界（代码及CPU契约验证完成）**。
    - `collection_contract.py`：新增`official_raw_transition_layout`/`official_window_batch_layout`，现有raw/window validator复用唯一shape/dtype布局；H/K动态，保持13/9字段、原调用、CPU/device/finite/非负ID/mask后缀与零填充拒绝规则。
    - 新增`execution_contract.py`：完整实现`FADAExecutionLimits`、字节核算、双槽chunk行数和预取预算检查、`FADAChunkDescriptor.validate`及成功阶段提交检查。描述符对照现有request与调用方游标/槽位代次；不从chunk序号猜跨request的代次。预算明确只覆盖声明的buffer，不冒充全进程内存限制。
    - Multi Agent：按用户本轮明确要求，在主Agent冻结`Mapping[str, (shape, torch.dtype)]`后，A独占布局文件，B独占执行契约且不导入A的新函数，二者无未来产物依赖；主Agent准备独立输入、集成与统一验证。子Agent仅做局部静态/导入检查，未开review或测试套件。
    - Evidence：主Agent一次`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-step61-1793353892394/verify.py`，终端4项PASS：不同H/K的真实tensor经现有from_mapping/validate；bool/int32/int64（大ID）、mask/finite/device；独立字节核算、预算不足与错误request/source/version/slot/代次/partial尾块拒绝；阶段缺失/超额抽取/剩余批次拒绝，Python/NumPy/Torch RNG未变。临时场景已清理，README已更新。
    - Boundary：没有分配共享内存/锁或实现预取线程、CUDA搬运；没有改Runner/RPC、入口/YAML、模型/loss/optimizer、artifact/checkpoint格式。字段布局和纯校验不证明真实并发安全。6.2/6.3可据此细化并独立实现，调度接线仍属6.4。
  - [x] **6.2 共享内存原始轨迹通道（显式收发分支与CPU集成完成；默认路由未切换）**。
    - 主Agent落地`ipc/shared_tensor_protocol.py`、request显式`transport_id`/空输出路径和官方factory端点注入。A独占新增`ipc/shared_tensor_channel.py`，B独占新增`fada/shared_collection.py`及collector共享分支，两者并行、通过已落地结构协议解耦。无额外review；子Agent实际额外执行了局部静态/行为检查，不将其当正式集成证据。
    - typed双槽通道在发布后可见、lease退出后释放，背压不覆盖、代次跨请求连续；creator独占unlink，attached只close。元数据与payload分离，共享内存容量预检包含固定metadata/header开销。FADA接收泵先于阻塞RPC启动，按行数/字节上限分配自有暂存，核准尾块、结果与来源/版本/Oracle及预处理身份后才返回完整raw batch。
    - worker共享请求在采样前验证端点/ID/布局，传输失败永久abort且不回落磁盘；非共享artifact调用保持显式旧路径。无raw轨迹写盘/hash交接，未改变native采集helper、模型/loss/配额/RNG、公共Runner、Replay/learner或存储格式。
    - Evidence：主Agent`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-step62/verify.py`一次运行4项PASS：真实spawn/Runner/prepare/SharedWeightSync/collector共享分支连续两次请求、双槽多块/partial、bool/int32/int64大ID与顺序/自有数据保持、无raw文件；resident worker退出及段不可重attach；背压释放/取消；producer失败、坏版本、暂存预算拒绝且无接收线程泄漏。仅native资源bootstrap/物理采样由明确CPU诊断替身提供，未跑原生/GPU/模型前向或训练，临时脚本已清理。
    - Scope：公开收发与factory注入可调用；正式`persistent_workflow`默认仍是artifact路径，切换/发布/持久化集成仍属6.4–6.5。未跑项目测试套件或独立审查，不宣称已实现跨轮采学重叠或取得速度提升。README已记录调用与资源所有权。
  - [x] **6.3 学习端预取与搬运流水线（可调用接口与CPU对照完成，CUDA未实测）**。
    - Multi Agent：主Agent冻结有限阶段batch lease/预算和`collect_metrics`接口、在修改前保存临时参考；A新增`batch_pipeline.py`，B仅改`official_learning.py`指标边界；两者同批并行。现有Replay CPU接口够用，`replay.py`未改；无额外review/项目测试套件，子Agent做了局部静态检查。
    - `OfficialFADABatchPipeline`：单worker按原`sample_update(mode,device='cpu')`顺序采样；planned次数、全部host/device槽与临时采样峰值受预算约束。context-managed租约只在消费期间有效；CPU不用CUDA，CUDA显式device/pinned/copy stream，ready/consumed事件保护读取/复用，失败/阶段结束等待自有工作。主Agent集成纠正显式CUDA分配、部分拷贝失败fence及首次分配流→copy流顺序；这些GPU分支未经本机实测。
    - `OfficialFADALearner.update(...,collect_metrics=True)`默认保留全部报告字段/原Python加权口径；False仅省去指标计算与回传，完整更新/计数仍执行。报告标量集中物化，无多步autograd图缓存。未删Planner零系数前向、合并来源或改任何算法/增强/精度参数。
    - Evidence：主Agent一次`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-step63/verify.py`，3项PASS。实际Replay采样owner/官方小型Transformer learner，offline2步+mixed3步，在噪声/dropout/裁剪下对照改前参考：窗口/配额/mask、loss/梯度/两AdamW/权重/计数/Python-NumPy-Torch RNG逐项完全相同；全指标与仅阶段末指标一致；零阶段、预算不足无抽样、producer/consumer错误与未完成阶段拒绝、无sampler线程泄漏。外部Replay存储明确替换为诊断NumPy窗口（不冒充官方Replay/物理采集重验）；临时参考和脚本已清理。
    - Boundary：6.3不改模型/配额/存储格式；pipeline已在6.4接线并于6.5成为官方默认，无跨轮旧策略采集。CPU证据不证明CUDA pinned/stream/event真实运行、显存峰值或速度。
  - [x] **6.4 内存发布、配对持久化与正式外循环接线（实现与CPU集成完成，已由6.5设为默认）**。
    - Multi Agent：A修改`runtime/persistent_runtime.py`内存激活，B新增`fada/persistence.py`单flight提交器；主Agent接通`persistent_workflow.py`及`replay.py`raw事务入库。使用既有Runner、共享权重/收发、有限pipeline和保存/strict-load，无第二训练循环或新checkpoint格式；未加review/项目测试套件，子Agent仅做局部静态检查。
    - `activate_checkpoint(path,state_dict=...)`严格验证现有布局/float32/finite后同步发布，loader/torch.load不参与；默认文件调用保持。`run_fada_persistent_async`新增显式`execution_limits`/`pending_raw_budget_bytes`，CLI/YAML未改；共享slot在spawn前创建，收发按真实prepare身份/行数预算核准，offline/online复用原Replay事务/筛选/重映射规则，无raw文件交接。
    - 单一学习循环消费pipeline；采样线程使用独立CPU recorder，阶段join后主线程合并计时，末步收集指标。学习完成→内存发布→submit配对存盘→下一轮采集到自有暂存→wait提交成功→入Replay/学习。存盘期间状态只读，不深拷贝Replay；最后一轮/返回/关闭等writer，无多采集。模型/两optimizer/计数与Replay配对原子提交，历史引用保留，外部resume不删除；发布与已提交进度分开。
    - Evidence：主Agent一次`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-step64/verify.py`，4项PASS：真实Runner/SharedWeightSync/collector共享分支/pipeline/learner/保存，warmup+两轮与顺序参考的模型/两AdamW/计数/Replay大小/末步指标一致，内存发布loader调用0、无raw落盘；确定性事件阻住writer仍能开始下一轮最新权重采集，但入Replay等提交；strict恢复续第3轮不重复warmup；Replay写完checkpoint提交前注入失败，旧latest配对和历史引用有效，无writer/receiver/sampler残留。native与外部NumPy存储为明确诊断实现，不冒充物理/上游Replay验证；临时参考、输入和脚本已清理。
    - RNG/范围说明：顺序参考特意用fork_rng隔离推理loader临时模型初始化；新内存路径移除该额外初始化，因此不声称与未经隔离的旧文件路线逐位RNG一致。6.3已有同采样/学习RNG下的严格数值对照。6.4未改算法/采样量/precision、公共Runner生命周期、checkpoint/schema或CLI默认值；CUDA/性能仍未实测，README已更新。
  - [ ] **6.5 正式配置切换与交付验收（本地子项完成；服务器正确性/性能待用户）**。
    - [x] 默认配置：`conf/distill/official_fada.yaml`新增独立`training.fada.execution`，transport128MiB/prefetch1GiB/pending raw2GiB、双传输槽/双学习槽。`workflow_setup.py:resolve_official_fada_execution(cfg,spec=...)`拒绝缺键/未知键/bool/非正整数和不够承载声明工作量的组合；与native config/保存算法/source身份隔离。不猜设备/工作量、不自动降采样或fallback。
    - [x] 唯一正式路径：`workflow.py`自动解析和传入必需预算，保留旧setup五元组及teacher override。`persistent_workflow.py`只走共享raw收发/事务入库、finite pipeline、内存发布、单flight提交，删除磁盘/source_batches、顺序学习/同步轮次保存及其helper/imports。校正batch_build覆盖真正获取lease的等待，不包含学习。通用runtime文件激活、artifact缓存/API和checkpoint/schema未删改。
    - [x] Multi Agent与交付：A独占配置/解析，B独占外循环，主Agent接composition并统一CPU验收。未新增review/项目套件；子Agent未执行项目测试。README启动/恢复/性能命令及预算、/dev/shm、提交屏障与口径限制已更新，无新增流程报告，临时参考/脚本/诊断输入已清理。
    - Evidence：主Agent一次`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-step65/verify.py`，4项PASS：Hydra compose→实际scripts.train_distill.run_fada_training→正式owner/真实Runner/共享collector/pipeline/发布/保存，warmup+两轮与6.4显式路径的权重/两AdamW/计数/末步指标一致；预算错误在model/worker前拒绝，default共享端点生效、loader调用0、无raw交接文件；strict恢复续第3轮且不重复warmup，预算未进保存算法/source身份；提交失败保持旧配对，worker/线程/共享段回收，既有[FADA PERF]输出实际执行。native/外部NumPy存储为明确诊断实现，不冒充原生或GPU。未跑全套测试/仿真/benchmark。
    - [ ] 用户服务器验收：上传更新代码后以真实官方产物/原生环境验证CUDA/strict-shadow、发布/恢复与资源回收，再在相同GPU、初始状态、env/H/K/batch/source配额/更新量/cache口径下对照已有基线，分开冷启动和稳态、实际transition/update数、端到端耗时及RAM/VRAM峰值。提交报告可能包含已采集的下一轮，不逐行误算物理耗时或相加重叠包络。缺真实改善证据时报告未证实提速；不自动连接或重跑服务器任务。
  - Frozen interfaces：raw字段/类型与流式消费、slot所有权和内存上限；batch/RNG/阶段结束屏障；最新发布版本与checkpoint/Replay提交边界。公共接口/配置由主Agent单写，A/B不互改文件或等待对方未来产物。
  - Multi Agent执行波次：6.1主Agent先冻结共享接口；6.2 A与6.3 B同批并行，主Agent负责公共接线；6.4主Agent集成后进入6.5。每步开始再用engineering-plan细化具体符号、允许/禁止范围和唯一近端证明，不一次展开全部函数方案。
  - 唯一未决语义：真正跨轮“采集下一轮＋学习当前轮”并行需要允许有界策略滞后，属于DAgger时序改变；当前默认不允许，不因接UniLab而悄悄开启。现有APPO ring只支持float32且覆盖过期槽，FADA需typed无损适配，不是加几项字段即可照搬。
  - 总体规划阶段的只读证据：核对APPO ring/staging与FADA原始字段、版本和生命周期差异，两个scout明确传输/持久化边界；后续实现和CPU证据见6.1–6.5子项。实际GPU速度仍受collector/shadow/学习占比制约，不预设倍率。

Blockers:
- 唯一外部验收缺口：用户尚未提供更新后正式默认路径的服务器运行/原生CUDA正确性及同工作量性能对照。现有服务器任务不会热切换代码，不能用本机CPU诊断替代GPU计时、原生shadow或策略效果。所有已授权本地实现已交付，不再追加“收尾阶段”、扫描旧权重或重复CPU场景；按用户要求不连接服务器。

Next:
- 用户上传当前代码并按README启动或恢复，反馈真实错误或性能记录后按具体证据处理；当前无待执行的本地实现。保留6.5/原第6步整体待服务器验收，不宣称已加速、不自动启动训练。
>>>>>>> 674954faf65b10b43223b68dfd8e73080945f7aa
