# OAT 论文实验执行交接

更新日期：2026-11-05。用途：交给另一个实验执行会话，延续已确认计划；不是实验完成报告，不是训练/真机运行授权。

## 1. 接手时先做什么

先读本文件，简短复述本次准备执行的实验组。依据用户在执行会话中的指令，核对当前配置、checkpoint 和数据记录能力；只解决该组的执行前提，不重新展开指标检索、画图调研或整篇研究设计。

**实验结构和评价方向已确认，但取值、重复次数、完整测量协议尚未全部确定。** 不能把代码默认值、空图图例或上一 Agent 的推测当作者批准的配置。不要看到本交接就自动启动整套实验、训练、仿真或真机部署。

## 2. 权威与文件入口

| 路径 | 地位与用途 |
|---|---|
| 本文件 | **DERIVED**：汇总本会话用户确认及直接读取的代码/稿件事实。新指令优先；区分已决定、待确定和建议。 |
| `/Users/chengyuxuan/ArtiIntComVis/CVPR2027/sec/4_experiments.tex` | **AUTHORITATIVE（当前稿件）**：实验结构、两张主表、组件表及参数图。当前全是空数据，不证明协议已实现。 |
| `/Users/chengyuxuan/ArtiIntComVis/CVPR2027/sec/3_method.tex` | **AUTHORITATIVE（当前稿件中的方法表述）**：含 OAT 标签公式；部分优化/执行说明仍为注释。 |
| `/Users/chengyuxuan/ArtiIntComVis/TTT/src/unilab/algos/torch/distill/fada/online_updater.py` | **AUTHORITATIVE（实现）**：标签模式、LoRA 回归与参数锚定。写交接时定向核对过。 |
| `/Users/chengyuxuan/ArtiIntComVis/TTT/src/unilab/algos/torch/distill/fada/online_controller.py` | **AUTHORITATIVE（实现）**：参数字段、参考模型选择、实际更新触发。写交接时定向核对过。 |
| `/Users/chengyuxuan/ArtiIntComVis/TTT/src/unilab/algos/torch/distill/fada/online_collector.py` | **AUTHORITATIVE（实现入口）**：未来窗口成熟、命令切换、最近成熟样本窗口；本会话曾读取，执行前只在相关时复核。 |
| `/Users/chengyuxuan/ArtiIntComVis/TTT/handoff_ttt_experiments.md` | **DERIVED／部分 STALE**：早期方法与实验交接。其 Demo、指标和消融组织被本次后续决定覆盖。 |
| `/Users/chengyuxuan/ArtiIntComVis/TTT/handoff_ttt_paper_metrics.md` | **DERIVED／部分 STALE**：历史检索边界与执行失控教训仍有效；表格状态和后续选定指标以本文件与当前 TeX 为准。 |

代码仓库：`/Users/chengyuxuan/ArtiIntComVis/TTT`；LaTeX 仓库：`/Users/chengyuxuan/ArtiIntComVis/CVPR2027`。

## 3. 共同方法与比较条件

- 本文方法称 **OAT**。**FADA-TTT** 是作者把 FADA 改为在线更新的版本，不是 FADA 原论文官方在线方法名。
- OAT 与 FADA-TTT 从同一个预训练 Planner–IDM checkpoint 开始，使用相同策略 I/O、命令与动作接口。
- Planner 保持固定；在线更新 IDM 的 LoRA。采集自身执行轨迹，等对应未来观测完整后用于更新；不用测试奖励或专家示范更新。
- 真机执行数据可以用于在线更新，不是“真机只做评价”。独立评价所用状态/位置真值不得偷偷输入策略或学习标签。
- 单组件比较保持相同初始化、LoRA 容量、数据可用规则、优化设置、指令和负载协议，只有被检验的组件变化。改变更新间隔 T 时实际更新次数可能改变，需披露，不伪称完全等预算。
- 不直接比较不同学习目标各自的训练 loss 来证明控制优劣；主表不加 IDM loss。

### 用户观察（不是本会话新运行的结果）

- FADA-TTT 通常仍不摔倒，但可能持续偏姿态、八字变形或不闭合。
- 用户查看了仿真，报告在所看条件下只有 OAT 能恢复正常直线行走能力。这是选简单直线消融的依据；不得把它扩写成全部种子、负载或真机均验证成功，也不要为确认用户观察而原样重跑。
- 用户关心速度/指令执行，也关心标称躯干姿态恢复；位置恢复、姿态恢复和路径闭合是不同证据，不能互相替代。

## 4. 主实验：4.2 Sim2Real、4.3 Sim2Sim

两节保留同结构主表；方法行为 FADA-TTT 与 OAT，没有 Pretrained/No Updates 第三行。所有结果仍为横线。

| 任务组 | 已定方向 | 主表指标 |
|---|---|---|
| Payload Figure-Eight | 持续负载八字；此前有渐进加载/卸载构思，具体操作参数未定 | E_pos（m）↓、归一化 E_v（无单位）↓ |
| Asymmetric Payload | 偏载绕柱 | E_pos（m）↓、SR（%）↑ |
| Sand Locomotion | 沙地八字或绕柱，路线仍未最终确定 | 当前 TeX 暂列 E_pos + 归一化 E_v；若确定绕柱，后者改为 SR |

**主实验没有恢复“单独偏载直线 Demo”。直线是作者后来明确接受的 4.4/4.5 消融任务，两者不要混淆。** 不默认沙地为硬地—沙地—硬地切换。

### 4.1 指标口径

#### E_pos：平面位置跟踪 RMSE

\[
E_{\mathrm{pos}}=\sqrt{\frac{1}{N}\sum_{t=1}^{N}\|p^{xy}_{t,\mathrm{actual}}-p^{xy}_{t,\mathrm{ref}}\|_2^2}.
\]

- 单位 m，低为好；逐时刻对齐实际与参考轨迹，反映偏移、累积漂移和执行进度差。
- 使用共同参考和固定场地坐标系；只在起点统一位置/朝向，不逐圈重新平移、旋转或缩放对齐。
- 形状相同但走慢也会产生误差，因此不能称为纯几何形状指标。没有采纳 DTW、nDTW 或 Fréchet，不要擅自引入时间扭曲匹配。
- 参考八字如何由任务/指令生成、绕柱参考如何固定、真机实际位置来源和同步方式仍需在具体执行协议中确定，不能事后拟合一条最接近结果的参考。

#### 归一化 E_v：FADA 原文定义

\[
E_{\mathrm{lin}}=\sqrt{\frac{1}{N}\sum_t\|v^{xy}_{t,\mathrm{cmd}}-v^{xy}_{t,\mathrm{base}}\|_2^2},\qquad
E_{\mathrm{ang}}=\sqrt{\frac{1}{N}\sum_t(\omega^z_{t,\mathrm{cmd}}-\omega^z_{t,\mathrm{base}})^2},
\]
\[
\bar E_v=\frac{E_{\mathrm{lin}}+E_{\mathrm{ang}}}{E_{\mathrm{lin}}^{\mathrm{TFD}}+E_{\mathrm{ang}}^{\mathrm{TFD}}}.
\]

- 包括平面线速度和转向角速度，不是纯线速度 RMSE；不是两个归一化比值的平均。
- 原文分母是相同任务、评价条件下的 TF–DAgger 基线。我们使用哪个 checkpoint/运行生成共同分母，**尚未最终落实**；不能把各变体分别归一化，或无说明换基线。
- 论文表头写 `\bar E_v`，无单位，不得标 m/s。原始 E_lin 为 m/s，E_ang 为 rad/s，应分别留存，便于核查合成指标。
- 仿真可用状态真值；真机可靠线速度来源仍未核实。不能把 IMU 积分或未经验证的估计当真值，也不默认已有/没有动捕、VIO、LiDAR。

#### 主表 SR 与消融存活率必须区分

- 绕柱 **SR**：按规定方向完成整条路线，且无碰柱、摔倒或人工接管。路线、完成边界、接触判定及超时需预先固定。
- 消融 **Survival**：规定运行时长内未摔倒的试验比例；不等于完成绕柱任务。
- 提前终止不得只用存活片段计算更低误差后宣称更好。失败标记必须留存；如何联合呈现终止与连续误差尚需固定，不能擅自补零、只删失败或自创惩罚。

## 5. 4.4 Component Ablations：简单负载直线任务

**已确认：不用八字或绕柱做这组消融；统一前进指令、负载、初始条件与运行时长，主要指标为存活率和归一化 E_v。**

组件表使用三列设计标记（Fixed reference / Trajectory pair / Anchor）加两列性能。不是逐步叠加所有组合，不要求全因子扫描。

| 版本 | 唯一变化 | 当前实现对应 | 注意 |
|---|---|---|---|
| FADA-TTT | FADA 原监督目标的在线版本，整体基线 | 执行会话核对真实基线入口与配置 | 不是 `action_base`，也不是 gamma=0；不能拿多项差异称单组件消融 |
| Online ref. | 用当前在线 Tracker 构造修正标签，替代固定参考 | `mode="trajectory_lora"`；其余保留完整 OAT | 构造标签仍 no_grad，每批标签固定，不是让梯度进入物理环境 |
| Action snapshot | 修正量第一项从固定 Tracker 对预测轨迹的输出，换成采集时动作快照 | `mode="action_base"`；其余保留完整 OAT | 不叫“仅状态误差”；不是 FADA 原目标 |
| No anchor | 去掉参数锚定 | `mode="trajectory_base"`, `sigma=0` | 不同时改 gamma、频率或 LoRA 容量 |
| Full OAT | 完整设计 | `mode="trajectory_base"`，sigma/gamma 采用正式配置 | 正式值未在本会话定下来 |

### 三种标签与锚定的准确含义

省略共同历史、命令等条件，只写首动作。令 I0 为固定参考 Tracker，Itheta 为当前在线 Tracker，Sp/Sr 为预测/实际的完整未来观测序列：

- 完整 OAT：`target = stopgrad[a_exec + gamma * (I0(Sp) - I0(Sr))]`。
- 在线参考变体：`target = stopgrad[a_exec + gamma * (Itheta(Sp) - Itheta(Sr))]`，构造发生于该批优化之前。
- 动作快照变体：`target = stopgrad[a_exec + gamma * (a_exec - I0(Sr))]`。
- OAT 学习输入为保存的 Sp，回归上述 target；附加 `sigma/2 * ||theta-theta_init||²`。theta 仅指可训练 LoRA；theta_init 是部署初始化时的固定副本，不保证是零。
- FADA 原文：`Itheta(Sr) -> a_exec`，部署仍输入 Sp。与 OAT 同时改变输入/标签，是整体目标比较。
- `I0(Sp)-I0(Sr)` 不等于 `I0(Sp-Sr)`；不要把“轨迹差分”误写为先减状态再输入网络。
- 两个动作差分变体在 `a_exec=I0(Sp)` 时可能相同，不能仅凭第一次更新判断它们长期是否等价。

当前 `online_updater.py` 明确支持三个 mode；这只是代码确认，**不代表三个实验路径均已端到端验证**。

## 6. 4.5 Hyperparameter Sensitivity：三个不同的问题

全部先用简单负载直线任务，不强迫每参数换任务，不机械只看相同整段均值。共同记录失败/存活情况。具体取值网格、默认点和重复次数未定。

| 参数 | 作者接受的实验思路 | 图与指标 | 不能擅自增加的东西 |
|---|---|---|---|
| 更新间隔 T | 正常走一段，在同一位置挂相同负载；从挂载完成开始比较适应过程 | 横轴挂载后时间；不同 T 的速度误差、Roll/Pitch 姿态误差过程；记录实际更新时刻/次数 | 不必反复加载卸载，不等完全适应后才开始统计；不把设定 T 当实际更新节奏 |
| 锚定系数 sigma | 固定同一负载，改变参数回拉强度，比较带载行走 | 横轴 sigma；归一化 E_v，以及共同标称姿态下 Roll/Pitch 偏差（分两系列） | **不默认加入“带载适应后关更新再卸载”**。这是之前讨论过但作者未理解/未采纳为必做的额外保持诊断，不是当前计划 |
| 修正增益 gamma | 固定任务负载，改变动作修正力度 | 横轴 gamma；归一化 E_v + 相邻动作命令变化量，观察跟踪与连续性 | 不把gamma叫已有严格物理意义的导纳参数；不把大gamma必抖、必摔、必最优写成结论 |

### 姿态与动作指标尚需落实的口径

- **姿态**：机载 IMU 提供 Roll/Pitch；评价相对共同、预先固定的标称姿态的偏差，不能直接把有符号角度均值或“越接近零越好”当通用定义。标称目标、刚体/IMU安装坐标、角度单位、聚合方式（当前只定偏差方向，尚未锁定 RMS/MAE）需确定。
- **时间曲线**：图里速度/姿态误差在共同固定时长窗口统计；不能把逐时刻误差冒称整段 RMSE。窗口长度、归一化参考与跨重复运行误差带定义未定。
- **动作变化**：已讨论用相邻动作差 `||a_t-a_(t-1)||²` 的时间平均，最终需明确动作是归一化网络输出还是执行目标、是否做关节维度平均及单位。它是命令连续性辅助诊断，不是机械冲击或稳定性证明。值小也不必然好，必须同时看跟踪误差。
- **sigma**：参数范数小不证明行为无遗忘；基座网络冻结/可重置 LoRA 也不证明当前策略保留了原工况能力。当前只做带载表现，不声称已验证抗遗忘。
- **gamma**：改的是标签修正量；有 Adam、锚定与梯度裁剪，不能等同线性放大参数步长。gamma=0 不等于 FADA。

### 更新时机的实现边界（代码确认）

`online_controller.py` 同时要求达到 warmup、`step_count % update_interval == 0`、collector.ready，才构造标签、做一次 updater.update 并清空成熟窗口。collector 保留最近成熟样本，命令切换可能使未成熟未来窗口失效。

因此：
- T 当前是控制步间隔，不是 Hz；换算秒需读取实际 ctrl_dt。
- 窗口未就绪时会跳过触发，实际次数不保证等于总步数/T。
- T 改变通常同时改变更新时间与次数；先如实研究部署设置的联合效果，不轻率宣称隔离了纯时效因果。
- “大T必导致老数据”“误差拐点对应首次更新就证明因果”都不是已确认事实。

代码 dataclass 可见默认 `window_size=10, warmup_steps=40, gamma=0.05, sigma=1e-4, update_interval=10, learning_rate=1e-4, max_grad_norm=1.0, lora_rank=8, lora_alpha=16, lora_dropout=0.05`。**这仅是代码默认值，不是作者已选超参数或当前加载运行配置；不要据此填实验计划或直接启动。**

## 7. 启动某组实验前必须落实的事实

仅补齐当前准备执行组所需内容，不把这张清单变成全套预审流程：

1. 真实入口、backend owner配置、checkpoint 与基线实现；检查模式是否真改变目标组件。
2. 该组负载质量/位置、挂载方式与时刻、速度指令、总时长/路线、重复次数与初始条件。
3. 正式 T/sigma/gamma、样本窗口、warmup、学习率、每次优化步数与 LoRA 配置；参数扫描只改目标量。
4. 度量输入、参考轨迹/标称姿态、坐标系与时间同步；真机测量能力不能由仿真真值推定。
5. E_v 的共同归一化分母、统计窗口、重复运行的聚合、失败/早停处理。
6. 运行资源及用户本次授权范围。遵守仓库AGENTS：始终用 uv run；不能单独 override training.sim_backend 切后端；不绕 env/backend contract；不未经批准建分支。

建议沿用现有实验产物目录与日志规范，至少能追溯：run配置、checkpoint、随机种子/初始条件、命令和动作、必要观测/真值、挂载事件、实际更新事件、终止/接管标记。**本会话没有创建可运行命令、新评测器或实验日志 schema，不要把建议当已存在接口。**

若执行会话已有配置/记录能回答以上问题，直接读取，不重新让用户提供工具可查信息；确实涉及未定实验语义时简短说明并确认，不擅自填值。

## 8. 当前图表与素材状态

- 4.2 主表 label `tab:sim2real-main`；4.3 为 `tab:sim2sim-main`。
- 4.4 组件矩阵 label `tab:component-ablations`，五行，三组件列＋Survival/E_v性能列；FADA-TTT组件列是破折号，表示目标不同，不是三个组件简单全关闭。
- 4.5 合成一个跨栏参数图 `fig:hyperparameter-sensitivity`：顶部 T 三面板，底部 sigma/gamma 各双面板。子图 label：`fig:update-interval`、`fig:anchor-coefficient`、`fig:correction-gain`。
- `T_short/T_mid/T_long` 只是图例样式，不是正式三个值；图里隐藏的0–1内部坐标仅供空轴排版，不是测量范围。无数据、误差带、趋势或最优点。
- `preamble.tex` 已加入 pgfplots/colortbl 与蓝、青、橙及浅色背景。当前 PDF 第5–6页展示组件表/参数图；最后已编译并实际查看，修复文字换行，无溢出/编译警告。
- **本会话没有执行任何 OAT/FADA 训练、仿真、真机实验或实测评估，没有结果可填入表格。**

图表素材库（非实验必要前提）：
`/Users/chengyuxuan/OneDrive/Files-Embody/FigureTable_References/CVPR2025/README.md`。
共18篇，8篇提取原生表格代码；真实曲线绘图实现0份。只在后续排版确有需要时打开，不再检索或扩库。当前绘图是自建 LaTeX 空图，不冒充作者原码。上一轮选定的视觉参考是 PointLoRA 组件勾选/浅底色、共享顶部图例、EVOS淡色包络；原论文数据不是OAT结果。

## 9. 文献定位与旧讨论边界

FADA 原文路径：
`/Users/chengyuxuan/OneDrive/Files-Embody/Sim2Real_Adaptation/extracted_text/2026 0626 FADA Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control.txt`。
- §4.2/公式4.5：真实未来输入预测实际动作，LoRA适应；主干与Planner冻结。
- 附录A.4：线/角速度RMSE及相对TF–DAgger的归一化。
- §5.3：K与未来信息遮蔽；§5.4训练目标消融；§5.5 LoRA/全参及数据量；§5.6角色分离诊断。

本地 RTR 原文 `.../extracted_text/2025 0826 Robot Trains Robot Automatic Real-World Policy Adaptation and Learning for Humanoids.txt:441-444` 明确IMU测Torso Roll/Pitch；它支持测量可行性，不自动规定我们的标称姿态误差公式。SuperMimic的Action smoothness是奖励中的相邻动作差量，借作本实验诊断不等于该论文已经证明gamma效应。

**不得重新恢复的越界内容：**
- 不自动推荐nDTW/Fréchet、增加动捕/LiDAR/VIO为前提，或用IMU双积分凑真机位置真值。
- 不自动扩大多Agent、论文搜索、全参数扫描或全套训练来填补一个局部事实缺口。
- 不用冻结reference输出稳定、标签不变、参数范数有界来证明任务改善、无遗忘或抗谱坍塌。
- 不预设某变体必抖、必摔、必遗忘；不把Agent共识当实验事实。
- 先前“固定reference独有自退火”“非负一致性gap过零”“gamma线性放大Adam步长”等说法已撤回，不得带入机制解释。
- 已确认计划不等于每个细节已可执行。表达先说明当前要跑哪组、为什么、还缺什么，不给用户一串术语和无边界菜单。

## 10. 可粘贴到实验执行会话的提示

```text
请先读 /Users/chengyuxuan/ArtiIntComVis/TTT/handoff_ttt_experiment_execution.md。
这是 OAT 论文实验执行交接。保留已确认的4.2/4.3主实验、4.4负载直线组件消融、4.5 T/sigma/gamma研究方向，不重新检索指标或扩展图表素材库。
请结合本会话已有的实验配置和记录，先核对准备执行那一组的checkpoint、真实模式、负载/指令协议、测量与统计口径。简短说明已具备和确实未定的事实；不要把代码默认值或空图图例当我的决定，也不要未经本会话明确授权就启动整套训练、仿真或真机运行。
```
