# OpenAdobe 架构设计 v0.1

> 目标：让 agent 操作**真实的剪辑/合成/音频工程**，在 agent 常用的剪辑、合成、音频工作上达到 Adobe Premiere Pro / After Effects / Audition 的水准。
> 原则：用真实剪辑/合成/混音引擎的语义（轨道、片段、转场、关键帧、效果图、混音台）做非破坏式编辑，由引擎渲染；**不**写代码去逐帧“画”视频（Remotion / HTML 渲染这一类不算）。工程**完全由 AI 管理**，人只看成片、给反馈（见 §1.1）。
> 日期：2026-10-02（版本信息以此为准）

---

## 1. 对原方案的评估

| 组件 | 原定角色 | 结论 | 理由 |
|---|---|---|---|
| Kdenlive | 主时间线 | **移出技术栈，只用它的引擎 MLT** | `.kdenlive` 本质是 MLT XML，可以用 `melt` 无头渲染。Kdenlive 没有稳定的脚本 API；工程又不再给人打开（§1.1），GUI 也就没有用处了。25.04 起原生支持导入 OTIO，可以当作应急出口。 |
| Natron | 实拍合成 | **降为可选** | 官方最后一个稳定版是 2.5.0（2022-11），macOS 只有 x86_64；2.6 一直是预发布版。2026 年出过第三方 fork 的 Windows 版。有 `NatronRenderer` 和 Python，但不适合当生产核心。 |
| Friction | 动态图形 | **不用作核心** | 工程文件 `.friction` 是二进制格式，没有无头渲染 CLI，“脚本”只是 ECMAScript 表达式，不是自动化 API。agent 读不了也写不了工程。 |
| Ardour | 多轨音频/混音 | **保留，升级到 9.x** | Ardour 9.0 在 2026-02 发布。`ardour9-lua`（luasession）能无头加载会话、执行 Lua，`ardour9-export` 能无头导出。会话格式是 XML，支持 LV2/VST3。 |
| Audacity 3.7.x | 波形/降噪/频谱 | **降为可选** | mod-script-pipe 要求 GUI 进程在运行，不是真正的无头。Audacity 4.0（2026-09-03）暂时去掉了脚本管道，所以要固定在 3.7.x。它的核心功能可以用无头 DSP / AI 工具替代。 |
| 自研控制层 | 让 agent 操作 | **方向对，但严重低估了** | 这是 80% 的工作量：统一工程模型、跨软件往返（Dynamic Link）、感知/QC 闭环、版本锁定，原方案都没写到。 |

### 需要补上的核心组件
1. **Blender 5.x**：作为 AE 层的主引擎。合成器、2D/3D 跟踪、抠像、遮罩、3D、文字和动态图形都能用 `bpy` 无头完整控制，开发很活跃。5.0 起合成节点树可以复用，序列编辑器也能直接挂合成修改器。
2. **Glaxnimate 0.6（KDE）**：替代 Friction，负责矢量/文字/形状动画。格式是 Lottie JSON，PyPI 上有 Python 模块，不需要 GUI。**MLT 自带 glaxnimate producer**，所以动画可以直接放到 Kdenlive 时间线上。
3. **FFmpeg / SoX + AI 音频工具**（DeepFilterNet、Demucs、Whisper 等）：负责无头的修复、分离、转写和响度处理。
4. **感知层**：agent 必须能“看见”和“听见”自己的成品。没有这一层，一切都是盲剪。

### 1.1 决策：工程完全由 AI 管理（2026-10-02）
人不再打开或编辑工程，只看成片、给反馈。这样做的好处：
- **单向编译**：OpenAdobe 工程模型是唯一事实来源。MLT XML、`.blend`、`.ardour` 都是编译产物，放在 `.build/` 里随时重新生成，**不用再解析人在 GUI 里的改动、做双向同步**。往返同步原本是最难的部分。
- **引擎选型只看无头能力**：GUI 好不好用、工程能不能被 GUI 打开，都不用再考虑。Kdenlive 退出技术栈，Natron 和 Friction 的问题也就不存在了。
- **测试更简单**：golden 测试只比较渲染结果，不再验证“GUI 能打开”。

前提条件（缺一不可）：
1. **人管成片，不管工程**：要有审片入口，包括预览渲染、版本对比，以及带时间码的评论。评论会映射到片段 ID，再变成 L3 操作。这是人和工程之间唯一的接口。
2. **可读、可审计**：工程模型（JSON）和操作日志（`ops.jsonl`）是人类可读的文本，开发者出问题时可以直接排查。另外保留 `export otio` 作为应急出口（Kdenlive 能原生导入），但不保证效果完全一致。
3. **不滑向“代码画视频”**：OpenAdobe 和 Remotion 这类方案的界线在于引擎语义，不在于人用不用 GUI。所有画面都必须由专业引擎（MLT、Blender、Ardour）按剪辑语义渲染，控制层只做编排、不渲染像素。

### 1.2 决策：OpenAdobe 不是一个 agent，是一层与宿主无关的工具（2026-10-02）
OpenAdobe 是“给 agent 用的 Adobe”，不是 agent 本身。核心是一个 Python 库，对外提供三种接口，优先级依次如下：
1. **CLI**（JSON 输入输出，每个动词都可以独立调用）：所有宿主都能直接用，也不额外占上下文。
2. **Skills**（`SKILL.md` 剧本，比如“剪口播”“做片头”“混音到 -14 LUFS”）：按需加载，pi、OpenClaw、Hermes、Claude Code 和 Codex 的格式基本通用。
3. **MCP server**：给只接 MCP 的宿主用。动词很多，要按领域分组，配合工具搜索，避免一次把所有工具都塞进上下文。

宿主（harness）可以随时替换，壁垒在工具层（工程模型、引擎编译器、感知/QC），不在 agent 的循环本身。
- **pi**（MIT，TS，极简）：OpenClaw 底层用的就是它的 agent 循环。适合以后做专用的“OpenAdobe agent”产品：用 TS 扩展把 CLI 封装成带类型的工具，再自定义 TUI。注意它没有内置权限系统，需要自己控制文件系统和进程的边界。
- **OpenClaw**（基于 pi，加上网关和二十多个消息渠道）：适合当“前台”：用户在聊天软件里发素材、收成片。只作为分发渠道，不作为核心。
- **Hermes Agent**（Python，支持 MCP + skills，可以自己沉淀新 skill，执行后端有 Docker/Modal 等）：技术栈和我们同为 Python，远程渲染后端也现成。自己学出来的 skill 必须经过 QC 评测才能采用，防止效果漂移。
- 硬性要求：宿主必须能把**图像**（抽帧、接触表、频谱图）作为工具结果交给多模态模型，否则 L4 感知层就失去了作用。

### 原方案最大的结构性问题
5 个软件、5 种工程格式，彼此之间**没有单一事实来源，也没有往返机制**。Adobe 之所以好用，靠的是 Dynamic Link 和共享的媒体缓存。这一层必须由 OpenAdobe 自己实现，而且应该是整个系统的中心，不是附属的“控制脚本”。

---

## 2. 目标架构

```
┌──────────────────────────────────────────────────────────────────┐
│ L6  Agent 接口：MCP Server + CLI(JSON in/out) + Playbooks(技能)   │
├──────────────────────────────────────────────────────────────────┤
│ L5  媒体智能：ASR/词级时间戳、说话人、镜头切分、节拍、CLIP 检索、│
│     人脸/显著性(自动重构图)、SAM2 遮罩、抠像、点跟踪              │
├──────────────────────────────────────────────────────────────────┤
│ L4  感知与 QC：抽帧/接触表、波形/频谱图、R128 响度、黑场/静音/冻帧│
│     检测、字幕 CPS/重叠、OCR、VLM 审片、低清快速预览              │
├──────────────────────────────────────────────────────────────────┤
│ L3  操作 API：类型化动词(insert/ripple/trim/keyframe/apply_fx…)  │
│     事务 + 操作日志(可撤销) + 校验(lint) + dry-run diff           │
├──────────────────────────────────────────────────────────────────┤
│ L2  OpenAdobe 工程模型（唯一事实来源）                             │
│     assets / timeline(基于 OTIO) / comps / audio_session /        │
│     captions / deliverables / render_cache(按内容哈希)            │
├──────────────────────────────────────────────────────────────────┤
│ L1  引擎适配器：读写原生工程 + 渲染 + golden 测试                 │
│     mlt(.kdenlive) | blender(.blend) | glaxnimate(.json/.rawr) |  │
│     ardour(.ardour) | ffmpeg | [natron(.ntp)] | [audacity]        │
├──────────────────────────────────────────────────────────────────┤
│ L0  引擎（无头、版本锁定、Linux 容器）                             │
│     melt · blender -b · glaxnimate(py) · ardour9-lua/-export ·   │
│     ffmpeg/sox · [NatronRenderer]                                 │
└──────────────────────────────────────────────────────────────────┘
          ↑ 人只接触：成片预览 + 版本对比 + 时间码评论 → 反馈操作
          （应急出口：export otio）
```

### 对标关系
| Adobe | OpenAdobe 引擎 | agent 操作的对象 |
|---|---|---|
| Premiere Pro | MLT（Kdenlive 工程） | MLT XML / OTIO |
| After Effects | Blender 合成器 + Glaxnimate（可选 Natron） | `.blend`（bpy）、Lottie JSON |
| Audition | Ardour 9 + DSP/AI 工具链 | Lua 会话脚本、渲染出的处理后素材 |
| Dynamic Link | OpenAdobe comp 节点 + 渲染缓存 | 工程模型中的 `comp` 引用 |
| Media Encoder | melt + ffmpeg 交付管线 | deliverable 配置 |

---

## 3. 核心流程

### 3.1 导入（Ingest）
1. 用 `ffprobe` 读取元数据，按内容哈希登记资产。
2. **统一转成 CFR 中间格式**（ProRes 422 HQ / DNxHR HQ），音频统一 48 kHz。手机拍的 VFR 素材是时间轴错位的头号来源。
3. 生成代理文件、波形和缩略图。
4. 做媒体理解：转写（词级时间戳）、镜头切分、节拍和嵌入向量，写入资产元数据。

### 3.2 剪辑（Pr 层）
- agent 调用 L3 的动词修改工程模型，模型再编译成 MLT XML，由 `melt` 渲染。
- 统一时基：用有理数帧率，时间一律存成项目帧率下的整数帧。
- 字幕：转写 → 断句 → 生成 ASS 样式，放到 MLT 字幕轨或直接烧录。
- 编译出的 MLT XML 只是构建产物，不对外承诺格式稳定。

### 3.3 合成与动态图形（AE 层）—— Dynamic Link
- 时间线上的 `comp` 片段引用一个 `.blend` 或 Lottie 文件，外加参数和输入素材。
- 渲染缓存的键 = hash(comp 源文件 + 参数 + 输入素材哈希 + 引擎版本)，缓存内容是无损 FFV1 RGBA（.mkv，带 alpha）——实测 MLT 会把 ProRes 4444 的 alpha 压成黑底，FFV1/PNG 则正确合成。
- comp 变了就只重渲那一段，时间线自动指向新缓存。
- AI 操作（SAM2 遮罩、抠像、CoTracker 跟踪）输出遮罩序列或跟踪数据，导入 Blender 节点使用，用来替代人工 roto。

### 3.4 音频（Au 层）
1. 按时间线的音频片段**生成 Ardour 会话**。轨道按角色分组：DIA / MUS / SFX / AMB，区域位置帧级精确。
2. 用 Lua 加插件、做自动化（闪避、EQ、压缩、限幅）。
3. 用 `ardour9-export` 导出各角色 stem 和总混音，响度按目标标准处理（-14 LUFS 网络 / -23 LUFS 广播）。
4. 混音回贴到时间线，同时静音原音频。
5. **Conform**：剪辑改了之后，按片段 ID 和角色重建区域，保留轨道上的混音设置。
6. 修复类处理（降噪、分离、去混响）走无头 DSP/AI，产出新资产并记录来源，不做破坏性修改。

### 3.5 交付与 QC
- 渲染 → 编码 → 字幕（外挂或烧录）→ 多画幅（16:9 / 9:16 / 1:1，用显著性自动重构图）。
- 自动 QC：响度、真峰值、黑场、静音、冻帧、字幕越界和 CPS、帧率/分辨率/色彩标记，最后用 VLM 审片。不通过就自动回到 L3 修正。

---

## 4. 工程纪律
- **版本锁定**：每个引擎都放进固定版本的 Linux 容器（`docker/engines.Dockerfile`）。工程目录按原样的绝对路径挂载进容器，所以编译产物里的路径在宿主机和容器里完全一致。
- **Golden 测试**：每个适配器都要有“生成工程 → 原生引擎渲染 → 和参考帧/参考音频比对”的回归测试。升级引擎前必须全部通过。
- **色彩**：v1 只做 Rec.709 SDR。Blender 和 Natron 共用一份 OCIO 配置。HDR 等 MLT 的 10-bit 落地后再做。
- **许可证**：GPL 引擎通过独立进程或文件格式调用，控制层的许可证可以自选。分发包含 GPL 二进制的镜像时，要履行提供源码的义务。
- **数量克制**：每多一个引擎就多一个长期维护面。核心只保留 MLT、Blender、Ardour、FFmpeg、Glaxnimate；Natron 和 Audacity 只作为插件可选接入。

---

## 5. 与 Adobe 的差距（诚实评估）

以下是**按 agent 常用工作量粗略估计的覆盖度，不是实测结果**。

| 领域 | 预计覆盖度 | 主要差距 | 补偿手段 |
|---|---|---|---|
| Pr | 75–85% | 专业调色/示波器、10-bit、多机位同步 | LUT/avfilter；用音频互相关做多机位同步 |
| AE | 50–60%（起步） | 文字动画器、表达式生态、Trapcode/Element 这类插件 | Blender 几何节点 + 模板库；AI 遮罩/抠像替代 roto |
| Au | 75–85% | 频谱修复精细度 | DeepFilterNet / Demucs / Whisper 等 AI 工具 |

agent 时代的真正优势不在于把 AE 的每个按钮都复刻一遍，而在于**用 AI 操作替掉最耗人力的环节**（roto、转写、对齐、卡点、重构图），再加上**可验证的自动 QC 闭环**。

---

## 6. 路线图

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 仓库骨架（Python）、锁定版本的容器、工程模型 schema、导入（probe/中间格式/代理）、MLT 编译器、基础感知（抽帧/接触表/响度/QC）、操作日志、最小 CLI + MCP | 端到端跑通：口播粗剪 + 字幕 + BGM 闪避 → mp4，自动 QC 通过 |
| P1 | 字幕样式、转场/特效目录映射、Ardour 适配器（会话生成/stem/响度）、Conform | 改剪辑后音频自动对齐，混音设置保留 |
| P2 | Blender comp 适配器（标题/下三分之一/抠像/跟踪贴图模板）、Glaxnimate 模板、Dynamic Link 缓存 | comp 修改只重渲受影响的片段 |
| P3 | AI 操作（SAM2、抠像、重构图、卡点）、VLM 审片、多规格交付 | 10 分钟口播 → 60 秒竖屏成片，一次通过 QC |
| P4 | 加固：参考工程回归集、引擎升级测试、GPU 和分布式渲染 | 引擎升级零回归 |

### 实现状态（2026-10-02）
- P0 完成：工程模型、导入、操作 API、MLT 编译器、字幕、闪避、两遍响度、感知、QC、撤销/批处理、CLI/MCP/Skill、OTIO 导出。验收脚本是 `examples/talking_head_demo.sh`。
- P1–P4 的进度见 `docs/PLAN-P1-P4.md`：
  - 全部完成（T1–T18）：转场、特效目录、变速、字幕预设与动画、音频修复、Ardour 混音引擎、Lottie 图形、Blender 合成（抠像/3D 标题/辉光/3D 卡片）、合成/Dynamic Link、AI 分析与抠像、竖屏重构图、卡点混剪、审片包、交付预设、代理文件、金标准回归。
  - P1–P3 验收脚本 `examples/full_showcase.sh`（纯 CLI，约 9 分钟）：16:9 主版与 9:16 Short 均 QC 通过，Ardour 混音，-14.1 LUFS / -1.5 dBTP。
- 已知限制：
  - Blender 3D 模板只能用 Cycles CPU（容器内 EEVEE 无 GPU 不可用、无 OIDN 降噪），1080p 3 秒标题约 2 分钟，1:1 观看有颗粒；
  - Ardour 适配器不处理变速、片段级音频效果和没有 Ardour 对应插件的轨道效果（如降噪，应在素材层用 `asset.process` 完成）。遇到这些时，交付会自动退回 MLT 混音并在结果中写明 `mix_fallback`，保证成片永远与模型一致；
  - macOS 的 Docker Desktop 文件共享会缓存目录项：宿主机删除的目录，下一个容器仍可能认为存在。需要“目录不存在”前提的步骤（如 Ardour 建会话）因此改在容器内删除；
  - 竖屏交付时叠加图形按比例缩放，不会按竖屏重新排版（Lottie 下三分之一会偏小）；
  - 抠像（RVM）在 CPU 上约 0.6 s/帧（512 工作尺寸）；
  - 渲染缓存键包含模板源码哈希，但不含引擎二进制版本（引擎镜像版本锁定在 `/etc/openadobe-engines.lock`）。
- 与 §1 原方案相比的调整：
  - Kdenlive 退出技术栈，只用它的引擎 MLT；
  - Friction 换成 Lottie（由 MLT 内置的 glaxnimate producer 渲染）；
  - Natron 没有采用，合成由 Blender 承担；
  - Audacity 没有采用，修复由 ffmpeg/RNNoise 承担。

### 基准任务集（作为“达到 Adobe 水准”的验收标准）
1. 口播精剪：去口误和气口、J/L 剪辑、动态字幕、BGM 闪避、-14 LUFS
2. 产品宣传片：卡点剪辑、标题动画、Lower third、调色 LUT
3. 绿幕合成：抠像、溢色抑制、光影匹配、跟踪贴图
4. 播客转视频：多机位同步、说话人自动切机位、降噪、竖屏重构图
5. 音频修复：降噪、去混响、人声和背景分离、响度合规

---

## 参考
- Kdenlive 2026 状态：https://kdenlive.org/news/2026/state-2026/
- Natron 发布：https://github.com/NatronGitHub/Natron/releases
- Friction 工程格式与导出：https://friction.graphics/documentation/usage.html · https://friction.graphics/documentation/export.html
- Ardour 9.0：https://ardour.org/news/9.0.html
- Audacity 4 脚本现状：https://digitalproduction.com/2026/09/08/audacity-4-rebuilds-the-fast-audio-editor/
- Glaxnimate 0.6：https://glaxnimate.org/news/releases/0.6.0/ · MLT producer：https://www.mltframework.org/plugins/ProducerGlaxnimate/
- Kdenlive OTIO 适配器（已并入原生）：https://github.com/KDE/kdenlive-opentimelineio
- Blender 5.0 合成器：https://developer.blender.org/docs/release_notes/5.0/compositor/
