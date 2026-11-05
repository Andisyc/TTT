# TTT 论文实验与执行止损交接

## 0. 新会话首先做什么

先读本文件，再读取下面列出的 LaTeX 当前文件。用简短中文确认接管范围即可，不主动重新启动指标检索、多 Agent 讨论、训练或修改。上一会话最后讨论的是“如何防止执行失控”，并未授权修改全局规则或 skill。真机指标仍未确定。

用户因上一会话反复改换指标、长时间检索、忽视真机测量可行性和复杂表达而明显不满，要求重新开会话。不要将上一 Agent 最后的推荐当作用户已接受的决策。

## 1. 文件与权威层级

- AUTHORITATIVE（作者意图）：本文件第 2 节逐项记录的用户确认、纠正与范围限制。
- AUTHORITATIVE（当前可编辑稿件状态）：`/Users/chengyuxuan/ArtiIntComVis/CVPR2027/sec/4_experiments.tex`。这只代表当前写进文件的内容，不代表实验已经完成或所有指标已最终确认。
- AUTHORITATIVE（当前排版依赖）：`/Users/chengyuxuan/ArtiIntComVis/CVPR2027/preamble.tex`，第 4 行加入了 `\usepackage{placeins}`。
- DERIVED（较早实验设计交接）：`/Users/chengyuxuan/ArtiIntComVis/TTT/handoff_ttt_experiments.md`。其中 Demo 候选状态须以本文件的后续用户决定为准。
- DERIVED：本交接对修改与讨论的总结；发生差异时以当前文件及用户新指令为准。
- HYPOTHESIS / 未采纳：本会话末尾提出的 nDTW、命令积分参考等指标方案。用户明确批评其测量负担，不能视为已选方案。

代码仓库：`/Users/chengyuxuan/ArtiIntComVis/TTT`。
LaTeX 仓库：`/Users/chengyuxuan/ArtiIntComVis/CVPR2027`。
本会话未改训练或部署代码，未运行训练、仿真或真机实验，未改任何 skill 或全局 AGENTS。

## 2. 作者已经确认的研究意图

### Demo

1. 走八字；此前讨论包含渐进加载与卸载，具体质量、圈数、时间与操作协议尚未最终确定。
2. 偏载绕柱。
3. 沙地走八字或绕柱；最终路线和沙地铺设方式未定。
4. 放弃单独的“偏载直线”Demo，原因是难以拍出好的展示效果。不得自行恢复它。

### 方法与观察

- 本文方法目前称 OAT；FADA-TTT 是作者将 FADA 改成在线更新的版本，不是原论文官方的在线算法名。
- 两方法从同一个预训练 Planner–IDM checkpoint 开始，使用执行中已完成的轨迹窗口更新；不使用测试时奖励或专家示范。
- 用户观察：FADA-TTT 即使不能回到标称姿态，通常仍不摔倒；可能仍能遵循部分指令，但八字形状会变化或不合拢。
- 用户观察/预期：OAT 适应后能恢复标称姿态，并希望以更稳定、可闭合的八字展示运动执行改善。
- 用户明确认为“速度幅值不完全准确，但实际路径与指令意图接近”也有评价价值。
- 用户要能评价过程、相对输入指令的成熟指标；不满意只看终点或相对任意预画路线的指标。
- 不能把姿态恢复必然导致位置/航向恢复或八字闭合写成已证明机制；这几项需分开建立证据。
- 不能默认动捕、视觉里程计、激光定位或可靠基座线速度已经具备，也不能反向断言用户一定没有。当前硬件测量条件未核实。

## 3. LaTeX 当前状态（交接时已实际读取）

`sec/4_experiments.tex` 的章节为：

- 4.1 Experimental Setup：已有 Tasks and Platforms 正文，其余主要是中文架构注释。
- 4.2 Sim2Real Evaluation：架构注释；没有新建真机主表。
- 4.3 Sim2Sim Evaluation：已有 Table 1。
- 4.4 Ablation Studies：架构注释。

### Sim2Sim Table 1

标签：`tab:sim2sim-main`。当前只保留 FADA-TTT、OAT 两行，用户要求删除了 Pretrained (No Updates)。所有结果仍是横线，没有真实数值。

任务组及当前列：

- Payload Figure-Eight：SR (%) ↑；E_v (m/s) ↓。
- Asymmetric Payload：SR (%) ↑；E_tilt (°) ↓。
- Sand Locomotion：SR (%) ↑；E_v (m/s) ↓。

这组指标是此前落盘的规划。后续用户指出持续八字不能只用不摔倒/完成率区分方法，尚未授权将新讨论替换进主表。不要把表中的 SR 自动沿用到 Sim2Real。

当前 caption 原文：

> Sim-to-sim evaluation of FADA-TTT and OAT on payload figure-eight, asymmetric-payload pole traversal, and sand locomotion tasks. Both methods start from the same pretrained Planner--IDM checkpoint and update online using trajectory windows collected during task execution.

排版偏好与已落实修改：

- 一张主表；方法为行、任务分组为列，不拆三张表。
- 指标、单位、箭头在同一行，不额外换行。
- 组内紧凑，组间留间距；不要用 `tabular*` 的 fill 把两个指标拉得很远。
- 删除了第二层的 Loading and unloading / Sustained asymmetric load / Figure-Eight / Pole Traversal 行。
- 使用 booktabs、普通 tabular、显式组内 8pt 与组间 20pt 间距；FADA-TTT 与 OAT 之间没有额外横线。
- Caption 应交代任务与必要实验条件，而非重复指标释义；指标定义属于 Experimental Setup。
- 用户指定对应参考应是 FADA Table 2（Sim2Sim），不是其硬件 Table 1。

表格使用 `table* [!t]`，4.3 结束、4.4 开始之前放了 `\FloatBarrier`。跨栏表当前在 4.3 标题之后的下一页顶部、4.4 之前。此处理会让稀疏草稿出现较大留白，不应宣称与标题同页。

曾尝试 cuted/strip 行内跨栏，但与 CVPR 审稿行号严重冲突、造成重叠，已经撤销。不要恢复这个失败方案。

验证历史：使用 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`；多次返回 up-to-date，推测有自动构建。曾将 PDF 第 3、4 页渲染为 PNG 观察，最终标准浮动方案没有文字重叠；最后一次 caption 编辑后检查编译产物为最新。不要未经需要重复编译或重查全仓库。

## 4. 文献事实与证据边界

### FADA（直接相关）

本地原文：
`/Users/chengyuxuan/OneDrive/Files-Embody/Sim2Real_Adaptation/extracted_text/2026 0626 FADA Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control.txt`

公开 v2：`https://arxiv.org/html/2606.28476v2`。

已核对：

- Table 1 是真机：每任务同时有 IDM loss 和任务行为指标；坡道/拖篮用成功率，负载行走用归一化速度误差，Kungfu 用 MPJPE。
- Table 2 是 Sim2Sim：每任务两列为源端 IsaacSim 与目标端 MuJoCo，绝不是 IDM loss 与另一个指标。
- Table 2 caption（本地 766–769 行）交代源/目标条件、归一化结果、未适应行的数据来源、适应行相同目标 rollout 预算。
- §5.2（832–840 行）明确说明源端评价目的：不同架构/训练目标的策略在源端表现相近，以减少“MuJoCo 优势只是源策略本来更强”的解释。
- 适应后 FADA / TF-CoPred-ft 的 IsaacSim 列为空，不能用该表宣称适应后保持源端性能或没有遗忘。
- 我们两方法同 checkpoint，因此不必为了模仿表格强行补两套相同的源端起点结果。
- A.1（1383–1418 行）是任务定义；A.4（1442–1471、1644–1688 行）是指标。
- 归一化速度误差为 `(E_lin + E_ang) / (E_lin_TFD + E_ang_TFD)`，不是两个归一化比值的平均。
- FADA 的 IDM loss 用实际未来轨迹预测执行动作，取首动作误差；各策略在自身 rollout 窗口上评价，因此还混有访问状态分布差异。
- 我们两个学习目标不同，各自训练 loss 不能直接横比。是否做统一独立 IDM 诊断还未决定；主表未加入 IDM loss。

尚未核实：FADA 真机速度/关键点的完整采集与处理链路。论文给了公式，但未找到足以复现的传感器、估计器、滤波、坐标与同步说明。不得编造为 contact-IMU EKF。

官网 `https://lecar-lab.github.io/FADA-humanoid/` 和官方仓库 GitHub 在本轮 native read 多次 ECONNRESET；gh 返回 EOF；Jina 返回401，不要继续循环重试。

搜索引擎对 README 的生成性返回曾提到 `mocap_unified.npz`，但直接源文件未读取成功，而且说明涉及 MuJoCo；这只是线索，不能证明 Table 1 真机用了某套动捕。图中的“No Mocap”也不能自动推广到所有评价测量。

### WASABI（DTW 真机先例，不是已采纳方案）

Li et al., Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations，CoRL 2022 / PMLR 2023。
`https://arxiv.org/html/2206.11693`

- §4.3 / Table 3：Solo 8 真机用 DTW 评价动作；基座10个反光标记，100 Hz Vicon。
- §3.2、B.1、H：其 DTW 特征含线速度、角速度、重力投影、高度，不是纯位置几何路径。因此只能说容许时间错位，不能说原版指标完全不惩罚慢走。
- 原文 Mori asymmetric / open-ended matching 设置不能直接搬到持续循环八字，尤其不能允许提前截断尾部掩盖未完成。

### nDTW（用户未接受，不能继续默认推荐）

Ilharco et al., General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping，2019。
`https://arxiv.org/html/1907.05446`

- §2：全过程、保序动态规划；支持连续空间点间欧氏距离；nDTW = exp(-DTW/(参考点数 × 距离阈值))。
- 来自具身导航，不是现成的人形速度指令跟踪基准。
- 将实际速度指令积分成参考路径，是我们额外提出的评价协议，不是该论文已经验证的G1方案。
- 需要实际位置来源。用户对“花很久最后仍给需要额外定位的指标”强烈不满。
- 不能说“完全速度不敏感”；采样、边界、距离与对齐方式仍影响结果。
- DTW 是保序动态规划，不是贪心最近邻；“八字交叉必然短路”“必须额外加入航向”曾是子 Agent 的错误推断，已否决。

### ADAPT（测量条件明确，原指标不适合直接移到八字）

Lyu et al., ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion，arXiv:2606.16542，2026。
`https://arxiv.org/abs/2606.16542`
本地：`/Users/chengyuxuan/OneDrive/Files-Embody/Sim2Real_Adaptation/extracted_text/2026 0615 ADAPT Analytical Disturbance-Aware Policy Training for Humanoid Locomotion.txt`

- §3.4（364–388 行）、部署说明（980–993 行）：FAST-LIO提供速度与全局定位，包含LiDAR到root的运动学转换。
- App D（1108–1163 行）：前向速度RMSE、世界系横向漂移、航向漂移，针对固定前向指令。
- 这些漂移定义不是连续八字的通用过程指标，不得偷偷恢复用户已放弃的偏载直线Demo。

## 5. 本会话失控的具体教训

用户已明确指出：这是执行失控，不是再找一篇论文就能解决。

上一 Agent 已承认的错误：

1. 没先核查真实测量条件，就不断推荐依赖额外观测的指标。
2. 反复切换评价对象：速度、存活、预画路线、终点、DTW；把用户纠正当成换一套方案的信号，没稳定问题。
3. 在没查成熟工作前先自创指标，让用户承担筛选负担。
4. 把“文献用过”当成“当前实验可用”，把未知设备和协议推到推荐之后。
5. 多 Agent 扩张了检索范围，且初稿含错误事实；纠错不是任务价值，Agent共识不是证据。
6. 表达越来越复杂，用术语包装尚未落地的结论。

后续严禁：

- 默认继续大规模指标检索或更多 Agent 轮次。
- 宣称已确定真机主表，或nDTW已获用户采纳。
- 擅自增加动捕、VIO、LiDAR、训练、仿真或真实试验作为任务前提。
- 为了给出正向推荐而隐去“现有证据还不能确认可用”的事实。
- 用“全领域唯一标准”“不存在任何成熟指标”这种超过检索证据的断言。
- 只重复道歉或反思而不遵循明确的新任务范围。

## 6. 执行止损规则讨论的当前状态

用户问：是否没有skill约束失控、怎么解决。

实际读取了：`skill://one-shot-execution`、`skill://multi-agent-discussion`、`skill://multi-agent`、`skill://skill-review`、`skill://skill-forge`。

事实：已有约束，并非完全没有。例如one-shot要求每次追加工作必须能改变交付/必要验证、充分后停止；multi-agent讨论要求针对实质分歧且足够后结束。本次有违反已有规则的责任，不能全部归因于缺skill。

上一 Agent 提出的方案（仅建议，未授权实施）：

- 不新建一个默认加载的防失控skill；把跨任务底线放全局AGENTS，由one-shot承载具体停止动作，research-lit与multi-agent-discussion只补自身扩张边界。
- 可行性先于优劣比较；依赖同一未满足前提的候选，不反复替换搜索。
- 改变问题、设备需求或关键假设前说明边界，不自动扩大。
- 有证据边界的否定结论也是有效交付，不为给推荐而持续检索。
- 多Agent应围绕能改变决策的缺口分工，而不是增加材料和流程。
- Markdown仅是软约束。硬性的任务超时/调用上限要执行器支持，不能写提示词后宣称已硬限制。

未修改任何skill或AGENTS，未调整运行器配置。新会话不能把上述建议当成已批准改动。

## 7. 下一会话可粘贴提示

请先读 `/Users/chengyuxuan/ArtiIntComVis/TTT/handoff_ttt_paper_metrics.md`，再按需读其中列出的当前LaTeX文件。上一会话在真机实验指标讨论中执行失控，我要求重新开始。不要继续默认推荐nDTW、启动多Agent检索或修改文件。先用简短中文确认：已落盘的Sim2Sim表是什么状态、真机指标仍有哪些未决事实、执行止损规则尚未实施；随后按我的新指令继续。明确区分作者确认、Agent建议与未核实事实，优先处理实际可行性，不扩大范围。
