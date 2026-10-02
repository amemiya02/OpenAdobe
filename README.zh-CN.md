# OpenAdobe

[English](README.md) · 简体中文

**让智能体操作真实的后期引擎。**

OpenAdobe 给 AI 智能体一间没有界面的后期工作室。MLT、Blender、Lottie（glaxnimate）、Ardour、ffmpeg
这些真实的剪辑、合成与混音引擎，统一藏在一套带类型、可校验、可撤销的操作 API 后面。智能体在真实工程里剪辑、
做标题、抠像、修音、混音，再自己抽帧看画面、测量音频，检查做得对不对。工程归智能体管理，人只审成片。
没有一帧画面是用代码画出来的。

设计文档：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · 智能体手册：[`skills/openadobe/SKILL.md`](skills/openadobe/SKILL.md)

## 能力
| 层 | 智能体能做什么 | 引擎 |
|---|---|---|
| **剪辑** | 音画联动；插入、波纹、修剪、分割、移动；变速（保持音调）；去静音；按文字剪；17 种转场（叠化、划像、推拉、闪黑/闪白）与音频交叉淡化 | MLT |
| **画面** | 18 种可加关键帧的效果：调色（eq、lift/gamma/gain、LUT）、模糊、锐化、暗角、裁切、变换（推镜头、Ken Burns、画中画）、色键 | MLT / avfilter |
| **图形** | Lottie 模板：标题、下三分之一、标注、行动号召、进度条、动态文字（支持中日韩） | MLT 内的 glaxnimate |
| **合成** | Blender 合成：抠像与溢色抑制、3D 标题、辉光、3D 卡片。按内容哈希缓存，改了才重渲（Dynamic Link） | Blender |
| **音频** | 修复链（RNNoise/FFT 降噪、去嗡声、去咔哒、去齿音、EQ、门限、压缩、电平、响度），产物是新的派生素材；闪避；轨道和总线插件；Ardour 会话混音与分轨导出 | ffmpeg / Ardour |
| **字幕** | 词级时间戳转写（faster-whisper），每次剪辑后字幕自动跟随；8 种样式预设，含卡拉 OK 逐字高亮 | libass |
| **AI** | 镜头检测、节拍与速度、人脸/主体跟踪、人物抠像（RVM）、卡点混剪、跟随主体的竖屏重构图 | ffmpeg / librosa / OpenCV / ONNX |
| **交付** | 平台预设（YouTube、Shorts/抖音/Reels、方形、B 站、EBU 广播）、两遍响度加限幅、SRT、每次渲染自动 QC、预览用代理文件、一键渲染全部规格 | MLT + ffmpeg |
| **审片** | 抽帧、带时间码的缩略图墙、波形、频谱、响度、黑场/静帧/静音检测、审片包（可选用 Claude 视觉模型点评） | ffmpeg |
| **可靠性** | 每个操作提交前先 lint，全部记录在日志里且可撤销；批处理是原子的；金标准参考渲染（`doctor --verify`）防止引擎升级带来回归 | — |

接口：JSON **CLI**、**MCP** 服务（图像以图片内容返回）、**Skill** 手册，以及作为应急出口的 **OTIO** 导出（Kdenlive 可导入）。

## 快速开始
```bash
uv sync --extra asr --extra mcp --extra otio --extra analysis
uv run openadobe doctor --build --verify      # 构建锁定版本的引擎镜像，并校验金标准渲染
examples/talking_head_demo.sh                 # 口播 → 剪辑 → 字幕 → 混音 → QC
examples/full_showcase.sh                     # 图形、抠像、修音、卡点混剪、Ardour 混音、竖屏短视频
```

## 引擎
`docker/engines.Dockerfile`（Debian trixie）：melt 7.30、ffmpeg 7.1、frei0r、Blender 4.3、Ardour 8.12、
RNNoise 模型、Noto CJK 字体。目录以相同的绝对路径挂载进容器，所以编译出的引擎文件在宿主机和容器里都能用。
设置 `OPENADOBE_ENGINE=local` 则改用宿主机上的程序。

## 测试
```bash
uv run pytest -m "not engine"   # 快速，只测纯逻辑
uv run pytest                   # 加上经过真实引擎的端到端渲染（较慢）
```
