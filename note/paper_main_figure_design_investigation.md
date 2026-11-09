# 论文方法主图制作、线条与配色调查报告

- 调查日期：2026-11-09。
- 项目：`/Users/chengyuxuan/ArtiIntComVis/TTT`。
- 对应论文：`/Users/chengyuxuan/ArtiIntComVis/CVPR2027`。
- 交付目的：让绘图会话知道调查如何进行、实际看过什么、哪些结论已有证据、哪些只是本项目的设计建议；不是仅提供一份绘图提示词。
- 范围：方法主图的制作工具、矢量输出、连接线、配色、字体及布局。实验曲线图的历次修改不纳入本报告的检索统计。
- 本报告是调查记录和设计依据，不是方法语义、算法结构或实验结果的权威来源。方法内容须以当前论文正文与用户确认的设计为准。

## 1. 结论与证据等级

**结论：论文主图中的 LaTeX 风格，不等于整张图由 LaTeX 绘制。纯 TikZ 和“矢量编辑器＋LaTeX 公式”都存在直接证据。对于本项目，用户已接受“TikZ 结构骨架＋外部素材＋独立编译为矢量 PDF”的路线。**

主图视觉上优先采用：白底、浅填充、深色文字、蓝色常规模块、暖橙强调贡献、中性灰辅助信息、规则连接线和固定间距。精确色号和线宽是本项目的起始建议，不是从顶会主图中测量出来的标准。

本文区分四类证据：

| 标记 | 含义 | 例子 |
|---|---|---|
| 直接事实 | 工具输出、源码、正文或已查看的图片能直接支持 | MetaPEFT PDF 的 Creator 为 Illustrator |
| 作者主张 | 小红书作者或教程自身的说法 | 某帖建议用一个强调色 |
| 推断 | 由多个事实推测，但未取得完整制作工程 | Illustrator 排版时导入了公式素材 |
| 本项目建议 | 为当前论文提出的可调整设计选择 | 0.9 pt 主流程箭头、蓝橙灰配色 |

**未进行：正式色盲模拟、最终主图打印测试、全平台使用率统计、对所有候选帖逐篇阅读。**没有据此宣称某工具占据行业多数，也没有将“低饱和”当成无障碍合格证明。

## 2. 小红书访问与登录：实际采用了什么

### 2.1 工具与后端选择

使用本机已有的 Agent Reach 路由说明，先运行：

```bash
agent-reach doctor --json
```

实际结果：小红书默认优先后端显示 OpenCLI，并提示 Chrome 扩展未安装。这只说明默认 OpenCLI 路径不可用，**不能证明另一个 MCP 后端没有配置或账号未登录**。

随后直接检查已有 MCP 配置：

```bash
agent-reach-mcporter call 'xiaohongshu.check_login_status()' --timeout 120000
```

首次返回 `ECONNREFUSED 127.0.0.1:18060`。这是本地服务未监听，不是账号认证失败，也不是外网 DNS 故障。

Agent Reach 命令曾被 lean-ctx 的命令白名单阻止，随后通过原生 shell 工具执行。未修改白名单、DNS、代理或用户浏览器配置。

### 2.2 本轮如何复用已经建立的登录态

实际发现以下已有文件：

```text
~/.agent-reach/tools/xiaohongshu/
  xiaohongshu-mcp-darwin-arm64
  xiaohongshu-login-darwin-arm64
  cookies.json
```

没有读取或展示 `cookies.json` 内容。检查了已有 MCP 二进制的 `-help`，然后在上述工作目录启动：

```bash
./xiaohongshu-mcp-darwin-arm64 -port 127.0.0.1:18060
```

本轮服务名为 `xiaohongshu-login-check`，监听本地回环地址。启动日志确认 MCP 版本 `v2.7.2`、注册 18 个工具，并使用缓存的 Chromium。随后再次运行 `check_login_status()`，返回“已登录”，账号显示为“滋滋乱崩”。

**本轮认证成功的实际机制：已有服务程序加载本地持久登录态，启动后重新验证。不是本轮重新扫码，也不是填写账号密码。**服务进程是否仍在运行具有时效性；后续会话先检查再决定是否启动，不重复占用同一端口。

复现顺序：

1. 执行登录状态检查。
2. 若端口拒绝连接，确认已有安装和端口占用，再启动已有服务。
3. 若状态明确为未登录，才进入二维码登录流程。
4. 登录确认后再检索，不把检索挂起误判成没有结果。

常规二维码工具入口为：

```bash
agent-reach-mcporter call 'xiaohongshu.get_login_qrcode()' --timeout 120000
```

这是后续失效时的工具入口，不代表本轮执行过该操作。首次成功登录的历史记录另列在下一节。

### 2.3 最初成功登录：历史核对结果

最初安装和扫码发生于**北京时间2026-11-05凌晨**。源会话时间戳使用UTC，因此日志里的Nov 4不能直接当成本地登录日期。该段历史通过只读历史咨询核对，未在本轮重装或重新扫码。

| 本地时间（UTC+8） | 实际过程 | 结果与边界 |
|---|---|---|
| 00:45–00:47 | 读取用户分享帖；普通网页跳转登录，备用入口401，浏览器桥超时 | 当时未打通认证访问 |
| 00:48 | 用户询问如何配置登录后端；研究OpenCLI，并执行安装dry-run | 仅dry-run，不是OpenCLI已安装或已登录的证据 |
| 01:11–01:19 | 用户不习惯Chrome，询问Firefox；之后授权安装独立MCP | 最终用独立Chromium，不导入Firefox Cookie |
| 01:20–01:24 | 尝试从GitHub下载二进制，遇TLS/ECONNRESET失败 | 自动下载没有成功；未修改网络配置 |
| 01:32–01:34 | 用户将login和MCP两个ARM64二进制手动下载到Downloads；助手以权限700安装到工具目录 | 主服务和独立login程序都存在，但独立login程序只见help调用 |
| 01:35 | 注册home范围MCP，地址为http://127.0.0.1:18060/mcp | 配置落在~/.mcporter/mcporter.json |
| 01:35–01:37 | codesign完整性检查通过，但spctl拒绝；用户选择自行手动批准，随后help可运行 | 没有关闭全局Gatekeeper，也没有证据表明助手移除了隔离属性 |
| 01:37–01:38 | 启动xiaohongshu-mcp-local；首次下载并解压Chromium | 初次30秒readiness超时，随后同一服务ready；不是反复重装 |
| 01:38:56 | check_login_status返回未登录 | 确认需要扫码 |
| 01:39:15 | 经MCP调用get_login_qrcode，保存并展示二维码 | 用户用手机小红书App扫码确认 |
| 01:39:55 | 用户回复“已登录” | 用户确认 |
| 01:40:10 | 再次调用check_login_status，工具返回已登录 | 原始认证成功的直接证据 |
| 01:41–01:43 | 成功读取原始分享帖、博主公开列表、实验图帖和表格帖，并查看相关图片 | 不只是状态检查通过，真实内容访问也成功 |

原始安装时使用过的注册命令（**已配置后不必重复执行**）：

```bash
agent-reach-mcporter config add xiaohongshu \
  http://127.0.0.1:18060/mcp --scope home
```

原始成功的状态检查与二维码保存命令：

```bash
agent-reach-mcporter call xiaohongshu.check_login_status \
  --timeout 45000 --output text

# 仅在确认未登录时使用；二维码临时文件不进入仓库。
agent-reach-mcporter call xiaohongshu.get_login_qrcode \
  --timeout 45000 --save-images /tmp/xiaohongshu-login-qr --output text
```

原始登录后成功读取的历史帖子包括：

- [overleaf中顶会顶刊latex表格绘制-（252）](https://www.xiaohongshu.com/explore/6aa90dfc000000002900c615)。
- [顶会实验图原图复现-（1）](https://www.xiaohongshu.com/explore/69e04b640000000021007b2d)：正文描述Origin复现，不能误记成取得TikZ源码。
- [overleaf中顶会顶刊latex表格绘制-（268）](https://www.xiaohongshu.com/explore/6ae972f50000000007000d20)。

这些旧帖用于说明原始登录后端确实工作，不计入第3节本轮主图风格的检索/深读数量。

**持久化路径的证据边界：**原始Nov 5片段未找到明确打印Cookie保存位置的日志；Nov 9才通过目录元数据观察到工具目录内的cookies.json，并在重启后验证复用成功。外部发布说明曾提及~/.xiaohongshu/cookies.json，不能用该文档路径覆盖本机实测路径。

历史取证位置（只供本机追溯，不需要把原始会话转交或上传）：

```text
/Users/chengyuxuan/.omp/agent/sessions/-ArtiIntComVis-TTT/2026-11-04T13-59-22-991Z_01a1a6db-766f-7486-8bce-59607af9fd43.jsonl
```

稳定消息锚点：初始未登录47459a7b；二维码返回5d82cabb；用户确认b079e5c8；成功登录b7480522；原帖读取c3887ffe；Origin实验图帖bb8df7cf；近期表格帖3f562cb6。本报告未复制这些调用中的认证或笔记访问token。


## 3. 调查怎么检索、怎么筛选

### 3.1 两个阶段

| 阶段 | 问题 | 检索/检查方式 | 实际完成 |
|---|---|---|---|
| A：制作路线 | 主图究竟是不是 LaTeX 绘制？ | 小红书、原生网页搜索、官方说明、GitHub 公开源码、本地 PDF 元数据/字体/图片检查 | 读取工具说明及真实 TeX；检查 MetaPEFT 两个图件；读取 1 篇相关小红书正文 |
| B：线条与配色 | TikZ 除了能画出来，怎样画得清楚好看？ | 小红书 4 组关键词，筛选正文和图片 | 80 条检索结果，70 个去重笔记；深读 6 篇正文，实际查看 5 张帖子图片 |

阶段 A 的小红书检索词为 `论文 方法图 LaTex`，返回 20 条。阶段 B 的 80 条不包含这次检索；两阶段不能直接相加称为 100 篇独立帖子。阶段 A 深读的 TikZ 帖也在阶段 B 候选中出现，但未在阶段 B 重读正文。两个阶段共读取 7 个不同笔记的详情。

### 3.2 阶段 B 的完整检索词

| 顺序 | 实际检索词 | 返回结果条数 | 用途 |
|---:|---|---:|---|
| 1 | 论文 框架图 连线 | 20 | 找方法图、主线组织与连线布局 |
| 2 | 科研绘图 配色 | 20 | 找色板、冷暖对比与配色经验 |
| 3 | TikZ 架构图 美化 | 20 | 找代码绘图的风格与工作流 |
| 4 | 科研绘图 线宽 箭头 | 20 | 补查线宽、箭头与连接规范 |

未指定特殊排序过滤，没有翻页或声称穷尽平台内容。搜索结果可能受账号、时间和平台排序影响。完整的 80 条返回结果保留为附录 A；附录明确区分检索候选和已读正文。

### 3.3 实际调用方法

```bash
agent-reach-mcporter call \
  'xiaohongshu.search_feeds(keyword: "科研绘图 线宽 箭头")' \
  --timeout 120000
```

从检索结果提取 `id`、`xsecToken`、标题、作者。使用同一次检索返回的参数读取详情：

```bash
agent-reach-mcporter call \
  'xiaohongshu.get_feed_detail(feed_id: "<搜索结果的id>", xsec_token: "<同一结果的xsecToken>")' \
  --timeout 120000
```

工具返回正文与 `imageList`，再用原生读取工具打开选定图片的 `urlDefault`。请求间隔约 2.5 秒，没有批量高并发抓取、评论、点赞或收藏。

实际执行时用持久 JavaScript 环境调用本机 CLI，并将 JSON 结果保留在内存中：`styleSearches` 保存 4 轮检索，`styleDetails` 保存 6 篇详情。该内存不保证跨会话存在；本报告保存了可恢复的标题、ID、检索词和判断依据，未保存账号 Cookie 或笔记访问 token。

### 3.4 筛选原则

- 优先与模型/方法图直接相关的条目，而非纯数据曲线、表格或海报背景。
- 同时保留“实际论文图案例”和“规则讲解”，不只看标题或点赞数。
- 对配色帖查看具体图片，不把“Nature 风格”“顶会级”等营销措辞当成质量证明。
- 生物机制图只迁移通用连接原则，不迁移生物学箭头语义。
- 不把收藏数、点赞数视为审美规则的验证；附录不以互动数据排名结论。

## 4. 实际读了哪些帖子，得到什么

### P01：代码直接生成主图的例子

- 标题：[从此，绘图进入 ai 时代，latex 绘图是神！](https://www.xiaohongshu.com/explore/6acca43a0000000015001a95)
- 作者：特务兔。
- 范围：阶段 A 读取正文和返回的评论；未逐张检查图片。
- 作者主张：从 PPT 转为让模型生成 TikZ；介绍分区流程图、时序图及内嵌公式，评论中描述把问题和公式交给模型生成示意图。
- 结论：确有实践者使用代码绘制整个流程图；不能据此推断某篇成熟论文也是同一制作方式，更不能把“单次生成成功”当作质量保证。

### P02：从具体会议图件提取配色

- 标题：[跟着 ICML 和 CVPR 26学习论文模型图及配色](https://www.xiaohongshu.com/explore/6a578bc9000000001c024b38)
- 作者：多模态信息融合。
- 范围：读取正文；帖子返回 13 张图片，实际查看第二张，不是全部 13 张。
- 正文提及 Holmes 与 DreamPRVR，并提供作者 GitHub 地址；本轮没有独立核实其全部发表信息，也未将这些模型机制移植进本项目。
- 实际图片是一张色卡：主色 `#10CB88`，并列出互补/邻近/对比色。它不是被逐项检查过的完整模型框架图。
- 结论：帖子能提供具体色号，但不能把整套高饱和色卡不加选择地搬入主图。主图需要角色映射，而非搜集尽可能多的颜色。

### P03：真实框架图中的层次感

- 标题：[ICML的清爽绘图风格！学了就能Accept‼️](https://www.xiaohongshu.com/explore/683e6c99000000002300d4a5)
- 作者：莱昂的科研日记。
- 范围：读取正文；3 张图片中实际查看第一张。
- 帖子指向 `Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach`，并给出 OpenReview ID `UZlMXUGI6e`。
- 已看图片：左侧总体输入输出流程，右侧局部机制展开；分组使用虚线框，模块采用浅黄、浅绿、浅蓝等底色，深色文字和灰色连接线。
- 可迁移之处：总览与细节分工；模块浅填充；重复结构保持一致；箭头不比模块内容更醒目。
- 不照搬之处：图片中也有较复杂的外围回路与多层分组，不能据“好看”就认为每一处都适合当前主图。帖子提及阴影，本项目并未因此决定增加阴影。

### P04：冷暖配色与数据点/背景的明度区分

- 标题：[科研配色｜海盐珊瑚篇](https://www.xiaohongshu.com/explore/6acef2240000000015015ff3)
- 作者：U米。
- 范围：读取正文；10 张图片中实际查看第二张。
- 已看图片：PCA 散点图，蓝、粉、浅橙、青绿四组；点的颜色比对应椭圆背景更深。
- 可迁移之处：浅底＋较深同系强调、冷暖对照、不同视觉层使用不同明度。
- 限制：这是数据图配色例子，不是方法框架图；低饱和、冷暖搭配本身不等于已验证色盲可辨。

### P05：主线组织与模块一致性

- 标题：[技术路线图怎么画逻辑清晰｜万能框架分享](https://www.xiaohongshu.com/explore/6aa3b78d000000002900f3ca)
- 作者：泡泡学术。
- 范围：读取正文；7 张图片未逐张查看。
- 作者主张：主流程只保留一条，侧边用虚线放补充内容；同层级模块统一尺寸、字号和对齐。
- 采纳：先突出主要阅读路径，重复模块使用统一样式。
- 不采纳为方法约束：不能把其“起点—方法—验证—结论”通用四段式强行套入算法主图。真实方法存在多分支、闭环时仍须表达，不为单主线删去核心关系。

### P06：箭头规范

- 标题：[机制图里的箭头和线条怎么画更规范？](https://www.xiaohongshu.com/explore/6aea908f000000001500661c)
- 作者：研研。
- 范围：读取正文；2 张图片中实际查看第一张。
- 正文建议：统一线宽、45°/90°走线、减少交叉、避免悬空、控制颜色数量，主流程顺着阅读方向。
- 正文给出 1.5–2 pt 线宽建议；这不是本项目最终线宽标准，需考虑整图缩放。
- 已看图片：细胞信号示意，包括直/弯箭头、虚线、抑制符号，说明该帖的箭头含义属于生物机制图。
- 采纳：连接端点准确、线型语义一致、减少交叉、主次分明。
- 不迁移：实心箭头=激活、T形=抑制、虚线=推测等生物学定义，不直接用于机器人/学习算法主图。

### P07：与本项目最直接相关的视觉规则

- 标题：[顶会论文配图的配色秘密](https://www.xiaohongshu.com/explore/6acd64530000000018011c12)
- 作者：小红薯F31A6AA7。
- 范围：读取正文；9 张图片中实际查看第一张。第一张是标题封面，不是已核查的完整案例。
- 正文六条规则：浅填充同色深描边；一个强调色突出贡献；一种颜色一种含义；弱化箭头图例注释；统一字体；横竖连线与同层对齐。
- 采纳：角色颜色固定、避免全图争抢注意力、浅底深边、横竖连接和一致对齐。
- 本项目修正：主流程箭头仍应清晰，不能一律弱到看不见；字体保持全图一致，但不机械要求所有说明文字与正文都采用同一种字体。一个强调色指一个主要强调角色，不限制所有数据图只能两种颜色。

### 4.1 阅读深度统计

阶段 B 六篇分别为 P02–P07：读取 6 篇正文；返回图片总数 44 张；**实际打开 5 张**（P02/P03/P04/P06/P07 各 1 张）。未把返回图片列表当作已经看过全部图片。P01 是制作路线阶段另行读取的 1 篇。

## 5. 制作路线：网页与本地图件的交叉证据

### 5.1 互联网检索词与来源

通过环境原生网页搜索工具执行过以下查询；没有将其冒称为 Exa 调用：

1. `machine learning paper architecture diagram figure TikZ Inkscape Illustrator draw.io LaTeX source`
2. `PlotNeuralNet TikZ neural network diagrams Ipe Inkscape latex paper figures`
3. `Inkscape LaTeX pdf_tex Ipe drawing editor latex scientific figures official`

部分初始结果是 AI 工具/技能推广，未作为核心证据。随后直接读取以下原始来源：

| 来源 | 实际核查 | 支持的结论 |
|---|---|---|
| [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) | README、目录及 [U-Net TeX](https://github.com/HarisIqbal88/PlotNeuralNet/blob/master/examples/Unet/Unet.tex) | 模块、3D块和连接可由 TikZ 代码完整生成；不等于其示例是原始论文作者的制作源码 |
| [Ipe 官方说明](https://otfried.github.io/ipe/01_intro.html) | 正文 | 可视化几何编辑器与 LaTeX 文字排版可以结合 |
| [TexText 官方文档](https://textext.github.io/textext/) | 正文 | Inkscape 中可插入且重新编辑 LaTeX/Typst 图形元素 |
| [Inkscape 官方 PDF/LaTeX 导出说明](https://wiki.inkscape.org/wiki/Release_notes/0.48#New_LaTeX_export_for_PDF/EPS/PS) | PDF与 `.pdf_tex` 的机制 | 图形和文字可分离，论文编译时叠合；该页较早，仅用于证实机制，不照抄旧CLI语法 |
| [牛津 OPIG 科学绘图教程](https://www.blopig.com/blog/2026/02/creating-scientific-figures-in-inkscape/) | 正文 | 真实研究工作流采用 Matplotlib＋Inkscape＋TexText，导出矢量图 |

### 5.2 本地 MetaPEFT 图件：不是凭外观猜软件

本地参考库：

```text
/Users/chengyuxuan/OneDrive/Files-Embody/FigureTable_References/CVPR2025/
```

重点文件：

```text
existing/metapeft/source_tree/fig/framework_v2.7.pdf
existing/metapeft/source_tree/fig/model_arch.tex
existing/metapeft/source_tree/fig/teaser_v4.0.pdf
```

执行的检查：

```bash
pdfinfo framework_v2.7.pdf
pdffonts framework_v2.7.pdf
pdfimages -list framework_v2.7.pdf
pdftoppm -scale-to 1600 -png -singlefile framework_v2.7.pdf /tmp/ttt-framework-provenance
```

并实际查看了框架图预览、读取 `model_arch.tex` 的嵌入与图注。

| 文件 | 直接观察 |
|---|---|
| framework_v2.7.pdf | Creator：Adobe Illustrator 29.3 (Macintosh)；Producer：Adobe PDF library 17.00；字体包含 cmmi7、cmmi10、cmsy10、cmr10、STIXGeneral、DejaVuSans；图片列表为空 |
| model_arch.tex | 使用 `includegraphics` 插入 framework PDF；图注说明 AdaptFormer、LoRA、Adapter 和双层优化机制 |
| teaser_v4.0.pdf | Creator：Adobe Illustrator 29.1 (Macintosh)；包含 MyriadPro、TimesNewRoman、DejaVuSans 等字体和 2 个嵌入图片对象 |

**事实**：方法主图最终由 Illustrator 导出，含数学字体且没有嵌入位图。**推断**：很可能在 Illustrator 组织布局并导入数学公式素材；仅凭字体和 Creator，不能确定公式究竟通过哪一个插件或中间格式进入图中。

PDF不是截图的同义词。主图可直接导出矢量PDF；TeX里只有 `includegraphics` 也不说明其上游由什么工具制作。参考库中的原图PDF与嵌入TeX不等于已经取得作者的 Illustrator 工程或完整绘图代码。

## 6. 本项目的综合判断与选型

### 6.1 制作路线

用户已接受：**TikZ 绘制结构骨架＋外部素材＋独立编译成 PDF**。

- TikZ承担：模块、分组、连接、数学标签、统一定位和状态标记。
- 外部资源承担：已有机器人渲染、真实场景照片、必要的轨迹/小图。
- 保留主图源码和素材；论文通过 `includegraphics` 插入PDF。
- 不以截图作为最终交付，不强迫用TikZ手工重画复杂照片/渲染。
- 若后续图明显偏向场景拼贴，自由布局可使用矢量编辑器精修；当前没有因此安装软件或启动制作。

### 6.2 配色：三种角色，不是每个框一种颜色

以下为本项目建议，**不是对参考图的取色测量**：

| 角色 | 描边/强调 | 浅填充 |
|---|---|---|
| 常规方法模块 | `#3569A0` | `#EEF4FA` |
| 需要强调的方法贡献 | `#C87942` | `#FBF0E7` |
| 输入输出、辅助信息 | `#6B7280` | `#F3F4F6` |

文字用深灰，暖橙保留为主要强调色；同角色跨分图颜色一致。冻结/训练状态应有明确文字或符号，不让一个颜色承担多种含义。尚未将上述颜色硬绑定到某个具体网络/损失模块。

### 6.3 连线：主次分层、方向统一

按**最终论文插入尺寸**给出起始参数：

| 元素 | 起始建议 | 解释 |
|---|---|---|
| 模块边框 | 0.6 pt | 不压过标签 |
| 主流程箭头 | 0.9 pt | 保留主要阅读引导 |
| 辅助连接 | 0.6 pt | 与主流程有层次 |
| 分组边界 | 0.5 pt，浅灰 | 只说明范围 |
| 箭头头部 | 长约2 mm、宽约1.3 mm | 同一类箭头统一 |
| 折线圆角 | 约1 mm | 缓和折角而非大弧线 |

不是把每条线都画成同样粗：**相同语义使用相同样式，不同层级可以有受控差别**。主流程默认左到右；更新回路沿外侧绕回；连接必须锚定边缘；优先消除交叉。虚线可用于更新路径，但必须核对方法并说明含义。冻结模块不自动使用虚线框。

### 6.4 字体与布局

- 以最终尺寸8–9 pt图内说明文字为起点，数学符号使用LaTeX；字号只定义少数层级。
- 先建立同层等高、统一内边距和约4 mm倍数的间距网格，不逐个目测摆放。
- 使用相对定位、锚点、共享TikZ样式；不能靠压小字体掩盖结构拥挤。
- 小圆角、白底、极浅分组背景；初稿不加渐变、投影和装饰性3D块。
- 先看黑白版本是否能读懂主线，再加颜色。必要闭环、分支不能为美观删除。

### 6.5 不应从调查中推出的结论

- 没有证据表明所有成熟论文都用TikZ，或多数都用某个软件；本轮不是软件使用率调查。
- 好看的公式字体不证明整图由LaTeX生成。
- 帖子标题里的“Accept”“顶会级”“Nature同款”不是审稿效果证据。
- 低饱和不自动等于可读或色盲友好；浅颜色不能用于弱对比的小字号正文。
- 箭头弱化不是箭头不可见；通用四段式模板不是本方法结构。
- 全部精确色号、线宽和间距都尚未经过本项目最终主图的实尺寸验证。

## 7. 绘图会话如何接手

第一步读取当前 `/Users/chengyuxuan/ArtiIntComVis/CVPR2027/sec/3_method.tex` 及用户在绘图会话中给出的内容设计，先确认：核心主张、必须呈现的模块、前向/监督/更新关系、冻结与可训练边界。

然后制作具有实际方法内容的TikZ构图，而不是仅返回工具推荐或抽象风格提示。制作时遵循第6节，保留可编辑源码并导出矢量PDF，按论文实际插入大小检查。

待确定事项：

1. 主图精确模块与分图内容；本调查未冻结这些科学语义。
2. 主图在论文中的最终宽度/高度与素材选择。
3. 每种线型、颜色、冻结/更新标记的最终语义映射。
4. 真实内容填入后的交叉连线、字体、灰度和色觉可辨性检查。

本轮只生成调查报告，不修改方法主图、论文TeX、实验配置或训练代码。之前对实验参数图的修改属于其他任务，不作为本报告的制作验证。

## 8. 可复现性、权限与资料边界

- 小红书登录状态是2026-11-09工具实测事实，不保证后续不失效。Cookie路径可记录，内容不得写入报告或转发。
- 正文链接保留稳定笔记ID，不写入访问token。后续若公开链接打不开，应重新搜索获取新token，不能据此认定报告中的阅读没有发生。
- 图片CDN地址可能过期；本报告记录已查看的图片序号和内容，不依赖临时CDN链接或 `/tmp` 截图才能理解结论。
- 小红书配图仅作研究参考，没有取得可直接复制进论文的授权；不将参考图整体或其中素材直接当作本项目资产。
- 未安装新绘图依赖、未升级工具、未发布小红书内容，也未导出Cookie。Agent Reach检查发现v1.6.1，而本机为v1.5.0；本轮未更新，不把更新当作阅读前提。
- 检索有采样和排序偏差；完整候选索引不等于逐篇审阅。直接事实、作者主张、推断和建议的区分见第1节。

## 附录 A：阶段B的完整检索返回清单

说明：每轮20条，共80条；跨查询去重为70个笔记。表内顺序为本轮工具返回顺序。“仅检索候选”不表示已验证其内容。“前阶段已读正文”指P01，避免重复记数。标题由平台返回，表格中的竖线替换为全角分隔以防Markdown断列。

### A.1 论文 框架图 连线

| 排名 | 帖子 | 作者 | 深读 |
|---:|---|---|---|
| 1 | [技术路线图怎么画逻辑清晰／万能框架分享](https://www.xiaohongshu.com/explore/6aa3b78d000000002900f3ca) | 泡泡学术 | 已读正文 |
| 2 | [跟着 ICML 和 CVPR 26学习论文模型图及配色](https://www.xiaohongshu.com/explore/6a578bc9000000001c024b38) | 多模态信息融合 | 已读正文 |
| 3 | [Codex 操控 Visio 画科研图](https://www.xiaohongshu.com/explore/6a6816bb000000000f03d880) | CCHAO4 | 仅检索候选 |
| 4 | [跟着顶刊学习论文框架图及配色](https://www.xiaohongshu.com/explore/6a8d3bd100000000170087cd) | 佳佳怪 | 仅检索候选 |
| 5 | [ICML风格设计Agent 主题的顶会配图](https://www.xiaohongshu.com/explore/6a60a737000000000f03ec33) | AI垂钓者 | 仅检索候选 |
| 6 | [overleaf中顶会顶刊latex表格绘制-（116）](https://www.xiaohongshu.com/explore/69b4b01c000000001b001942) | 爱芝芝爱科研 | 仅检索候选 |
| 7 | [框架图](https://www.xiaohongshu.com/explore/6a0d17870000000006020453) | 苏格拉没有底 | 仅检索候选 |
| 8 | [深度学习，神经网络绘图模板](https://www.xiaohongshu.com/explore/6a4b7a6a0000000016027197) | Ray的科研绘图日记 | 仅检索候选 |
| 9 | [开题论文救命！闭环 ／ 发散框架图直接抄✅](https://www.xiaohongshu.com/explore/6aed78aa000000002503954d) | 研图百宝箱 | 仅检索候选 |
| 10 | [一个门外汉来设计顶会论文图片](https://www.xiaohongshu.com/explore/6a5769c9000000000f01689b) | AI垂钓者 | 仅检索候选 |
| 11 | [多巴胺式科研绘图，舒适清爽，快来学习‼️](https://www.xiaohongshu.com/explore/6843b86900000000230120a5) | 莱昂的科研日记 | 仅检索候选 |
| 12 | [SCI 中好看的框架图](https://www.xiaohongshu.com/explore/6abfcd87000000001c01cf6f) | EGO（科研前沿版） | 仅检索候选 |
| 13 | [那些让审稿人一眼惊艳的科研绘图（09）！](https://www.xiaohongshu.com/explore/6add62780000000033023c2f) | 科研配图不秃头😏 | 仅检索候选 |
| 14 | [overleaf中顶会顶刊latex表格绘制-（202）](https://www.xiaohongshu.com/explore/6a4df9ef0000000011007422) | 爱芝芝爱科研 | 仅检索候选 |
| 15 | [overleaf中顶会顶刊latex表格绘制-（231）](https://www.xiaohongshu.com/explore/6a7c16ed00000000320336d1) | 爱芝芝爱科研 | 仅检索候选 |
| 16 | [经典的信息量拉满的大模型绘图！学不完‼️](https://www.xiaohongshu.com/explore/68494bbe0000000022036e7f) | 莱昂的科研日记 | 仅检索候选 |
| 17 | [overleaf中顶会顶刊latex表格绘制-（220）](https://www.xiaohongshu.com/explore/6a65a6e70000000011005a8e) | 爱芝芝爱科研 | 仅检索候选 |
| 18 | [参考计算机顶会ICLR风格的AI绘图初步尝试](https://www.xiaohongshu.com/explore/69d396140000000023012282) | AI垂钓者 | 仅检索候选 |
| 19 | [我导说：这框架图已经是天花板了](https://www.xiaohongshu.com/explore/69397ef7000000001e00dfe0) | 🎓 学姐的Paper工具箱 | 仅检索候选 |
| 20 | [那些让审稿人一眼惊艳的科研绘图（04）！](https://www.xiaohongshu.com/explore/6ad6b47d0000000028027b84) | 科研配图不秃头😏 | 仅检索候选 |

### A.2 科研绘图 配色

| 排名 | 帖子 | 作者 | 深读 |
|---:|---|---|---|
| 1 | [跟着 Nature 学配色。](https://www.xiaohongshu.com/explore/6a8d0a7b00000000280253e5) | 一名埋头苦干的博士僧 | 仅检索候选 |
| 2 | [ICML的清爽绘图风格！学了就能Accept‼️](https://www.xiaohongshu.com/explore/683e6c99000000002300d4a5) | 莱昂的科研日记 | 已读正文 |
| 3 | [科研配色｜海盐珊瑚篇](https://www.xiaohongshu.com/explore/6acef2240000000015015ff3) | U米 | 已读正文 |
| 4 | [Nature系绝美科研配色分享](https://www.xiaohongshu.com/explore/6875d1df000000000d026db3) | 小智科研站 | 仅检索候选 |
| 5 | [分析一组Nature系类配图](https://www.xiaohongshu.com/explore/6a9b01610000000028028d40) | 新博研科研测试 | 仅检索候选 |
| 6 | [跟着顶刊学配色❗️](https://www.xiaohongshu.com/explore/6a8af635000000002803131b) | 学术调色师 | 仅检索候选 |
| 7 | [跟着 ICML 和 CVPR 26学习论文模型图及配色](https://www.xiaohongshu.com/explore/6a578bc9000000001c024b38) | 多模态信息融合 | 已读正文 |
| 8 | [科研人必存｜5组双色渐变背景，做封面/机制图](https://www.xiaohongshu.com/explore/6ab9ce23000000000b0068b9) | 一期一绘 | 仅检索候选 |
| 9 | [不同数据，其实应该用不同配色](https://www.xiaohongshu.com/explore/6a13e7290000000006021e91) | 吃不胖的衰哥 | 仅检索候选 |
| 10 | [6 大顶刊科研配色分享✨画图直接抄](https://www.xiaohongshu.com/explore/6ae31961000000001f006e13) | 半妆先生 | 仅检索候选 |
| 11 | [【配色灵感】｜清新柔和配色🌿](https://www.xiaohongshu.com/explore/6acd99a0000000001303d25a) | 一期一绘 | 仅检索候选 |
| 12 | [跟着顶刊学配色❗️](https://www.xiaohongshu.com/explore/6a61e0a10000000013027a02) | 学术调色师 | 仅检索候选 |
| 13 | [清晰而简约的配色❗️](https://www.xiaohongshu.com/explore/692e67d7000000001e003afd) | 科研绘图酱 | 仅检索候选 |
| 14 | [顶会论文配图的配色秘密🎨](https://www.xiaohongshu.com/explore/6acd64530000000018011c12) | 小红薯F31A6AA7 | 已读正文 |
| 15 | [顶刊配色直接抄！Nature同款色板也太绝了](https://www.xiaohongshu.com/explore/6a4f88a3000000001102cdbf) | Her.o | 仅检索候选 |
| 16 | [大可科研配色｜10套配色合集08](https://www.xiaohongshu.com/explore/6ae432ff0000000019001a5d) | 绘图的大可博士 | 仅检索候选 |
| 17 | [审稿人一眼好感｜学习可视化统一色卡](https://www.xiaohongshu.com/explore/6ae048f6000000001503a99e) | 沁言学术研究所 | 仅检索候选 |
| 18 | [顶刊柱状体配色，审美在线不翻车](https://www.xiaohongshu.com/explore/6a699e930000000013025af9) | 耶耶师姐 | 仅检索候选 |
| 19 | [Image2.6直出顶刊级科研配色方案！](https://www.xiaohongshu.com/explore/6aeef57c000000000e034a1f) | 清风学长 | 仅检索候选 |
| 20 | [Nature顶刊配色灵感🌷](https://www.xiaohongshu.com/explore/6a4dd901000000002103ddc8) | 花本兰 | 仅检索候选 |

### A.3 TikZ 架构图 美化

| 排名 | 帖子 | 作者 | 深读 |
|---:|---|---|---|
| 1 | [分享下最近让ai做的TikZ可视化绘图编辑器](https://www.xiaohongshu.com/explore/6aeb51120000000016033efb) | 云w | 仅检索候选 |
| 2 | [一招解决 ChatGPT“画图丑”的问题！](https://www.xiaohongshu.com/explore/68afd253000000001b037bce) | 二狗 | 仅检索候选 |
| 3 | [用LaTeX绘制生物试题中的图形](https://www.xiaohongshu.com/explore/6adb3e9d000000002c0114a2) | 朴数 | 仅检索候选 |
| 4 | [开源TikZ论文配图skill](https://www.xiaohongshu.com/explore/6a9e9264000000002802a5ac) | HMT | 仅检索候选 |
| 5 | [萌新看过来：画深度学习的网络图](https://www.xiaohongshu.com/explore/681dc5df000000002202f7ff) | hakone suguru | 仅检索候选 |
| 6 | [从此，绘图进入 ai 时代，latex 绘图是神！](https://www.xiaohongshu.com/explore/6acca43a0000000015001a95) | 特务兔 | 前阶段已读正文 |
| 7 | [tikz绘图](https://www.xiaohongshu.com/explore/69de7b0b000000002200df0a) | 禤翛 | 仅检索候选 |
| 8 | [真正舒服的Tikz画图粘贴大法！](https://www.xiaohongshu.com/explore/695ccf7b000000000a02fe6c) | 礼貌小猫咪 | 仅检索候选 |
| 9 | [讲义排版终于攻克了 LaTeX 表格插图](https://www.xiaohongshu.com/explore/6a929f3c0000000004028a8d) | 1920高中数学 | 仅检索候选 |
| 10 | [八年沉淀 2300页TikZ绘图经典资料 - 合集下](https://www.xiaohongshu.com/explore/65f6abbe00000000120207a6) | LaTeX 工作室 | 仅检索候选 |
| 11 | [征服审稿人的 LaTeX 表格+热力图技巧](https://www.xiaohongshu.com/explore/68fec14b000000000700fbe4) | 小熊 | 仅检索候选 |
| 12 | [7 个高颜值 tcolorbox 文本框](https://www.xiaohongshu.com/explore/6a7d15900000000005028835) | LaTeX 工作室 | 仅检索候选 |
| 13 | [GitHub项目｜LaTeX绘制机器学习算法示意图](https://www.xiaohongshu.com/explore/67f51533000000001e000b83) | dataworker | 仅检索候选 |
| 14 | [tikz-cartes 使用手册](https://www.xiaohongshu.com/explore/6a4595eb0000000006023525) | Arcadian | 仅检索候选 |
| 15 | [最慢的往往是最快的论据➕1](https://www.xiaohongshu.com/explore/67a39330000000002900a47b) | 🎄 | 仅检索候选 |
| 16 | [最近用tikz帮别人画的矢量图](https://www.xiaohongshu.com/explore/697f7a080000000021028bd9) | jam | 仅检索候选 |
| 17 | [圆锥曲线基础知识点·LaTeX 表格分享](https://www.xiaohongshu.com/explore/6a9fd2dc000000001103088a) | 1920高中数学 | 仅检索候选 |
| 18 | [AI教你用Tikz画高次不等式的“波浪线图”](https://www.xiaohongshu.com/explore/69ff00ad000000003502a413) | 普通师匠Hugo | 仅检索候选 |
| 19 | [tikz + tcolorbox + latex 简直是好伙伴](https://www.xiaohongshu.com/explore/67e41e31000000001c0062e0) | LaTeX 工作室 | 仅检索候选 |
| 20 | [用Jupyter一键预览 TikZ，让图形所见即所得](https://www.xiaohongshu.com/explore/6847686e000000002202fa08) | LaTeX 工作室 | 仅检索候选 |

### A.4 科研绘图 线宽 箭头

| 排名 | 帖子 | 作者 | 深读 |
|---:|---|---|---|
| 1 | [机制图里的箭头和线条怎么画更规范？](https://www.xiaohongshu.com/explore/6aea908f000000001500661c) | 研研 | 已读正文 |
| 2 | [ICML风格设计Agent 主题的顶会配图](https://www.xiaohongshu.com/explore/6a60a737000000000f03ec33) | AI垂钓者 | 仅检索候选 |
| 3 | [如图弧度的箭头是怎么画出来的](https://www.xiaohongshu.com/explore/6794b910000000002901c7e8) | vs布布 | 仅检索候选 |
| 4 | [顶会论文配图的配色秘密🎨](https://www.xiaohongshu.com/explore/6acd64530000000018011c12) | 小红薯F31A6AA7 | 已读正文 |
| 5 | [科研绘图／30秒绘制机制图中常用的8种箭头](https://www.xiaohongshu.com/explore/68cf87120000000012032d1f) | 芝士就是力量 | 仅检索候选 |
| 6 | [overleaf中顶会顶刊latex表格绘制-（116）](https://www.xiaohongshu.com/explore/69b4b01c000000001b001942) | 爱芝芝爱科研 | 仅检索候选 |
| 7 | [codex科研绘图无敌了](https://www.xiaohongshu.com/explore/6a6935d0000000000100265c) | Cgamer | 仅检索候选 |
| 8 | [可编辑论文框架图模板合集（PPT版）来啦！](https://www.xiaohongshu.com/explore/69f021c80000000038021ce3) | 科研火柴人 | 仅检索候选 |
| 9 | [不擅长画图？AI生图不能改？直接转可编辑PPTX](https://www.xiaohongshu.com/explore/6a6b8beb0000000032032cf3) | Lancelot | 仅检索候选 |
| 10 | [多巴胺式科研绘图，舒适清爽，快来学习‼️](https://www.xiaohongshu.com/explore/6843b86900000000230120a5) | 莱昂的科研日记 | 仅检索候选 |
| 11 | [博士生codex用科研绘图skills（4）](https://www.xiaohongshu.com/explore/6a3bd6a40000000021017a1c) | 嗜血锅包肉（读博版 | 仅检索候选 |
| 12 | [overleaf中顶会顶刊latex表格绘制-（202）](https://www.xiaohongshu.com/explore/6a4df9ef0000000011007422) | 爱芝芝爱科研 | 仅检索候选 |
| 13 | [ICML的清爽绘图风格！学了就能Accept‼️](https://www.xiaohongshu.com/explore/683e6c99000000002300d4a5) | 莱昂的科研日记 | 已读正文 |
| 14 | [征服审稿人的 LaTeX 表格+热力图技巧](https://www.xiaohongshu.com/explore/68fec14b000000000700fbe4) | 小熊 | 仅检索候选 |
| 15 | [参考计算机顶会ICLR风格的AI绘图初步尝试](https://www.xiaohongshu.com/explore/69d396140000000023012282) | AI垂钓者 | 仅检索候选 |
| 16 | [审稿人Accept的SCI论文图配色方案💡折线图](https://www.xiaohongshu.com/explore/687dbcd800000000130111e3) | 爱科研的豆包 | 仅检索候选 |
| 17 | [overleaf中顶会顶刊latex表格绘制-（231）](https://www.xiaohongshu.com/explore/6a7c16ed00000000320336d1) | 爱芝芝爱科研 | 仅检索候选 |
| 18 | [academic-figure-skill科研绘图使用教程](https://www.xiaohongshu.com/explore/6a54f3980000000007013da0) | 科研绘图酱 | 仅检索候选 |
| 19 | [顶会实验图原图复现-（1）](https://www.xiaohongshu.com/explore/69e04b640000000021007b2d) | 爱芝芝爱科研 | 仅检索候选 |
| 20 | [完全可编辑！论文画图有救了[ICLR2026]](https://www.xiaohongshu.com/explore/6983fd670000000028023443) | Always | 仅检索候选 |
