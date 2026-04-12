# Pipecat Quickstart 当前框架与面试助手建议

## 文档范围

本文档说明当前 `pipecat-quickstart/server/bot.py` 中已经实现的实时语音对话框架，以及在此基础上演进为“面试助手”的推荐方案。

当前机器人采用的是级联式语音链路：

`Transport -> STT -> 用户回合聚合 -> LLM -> TTS -> Transport`

它现在面向的是通用中文实时语音对话，还不是一个结构化的面试系统。

## 当前入口与运行方式

当前入口文件是 `pipecat-quickstart/server/bot.py`。

主要运行职责如下：

- `bot(runner_args)`：根据运行参数选择具体的传输层实现。
- `run_bot(transport)`：组装服务、上下文、处理管线和任务生命周期。
- `PipelineRunner`：驱动整条管线运行，直到连接断开或任务被取消。

当前 bot 支持的传输方式：

- `DailyTransport`，用于 Daily WebRTC 房间。
- `SmallWebRTCTransport`，用于轻量级 WebRTC 连接。

## 当前核心组成

### 1. Transport 层

Transport 层负责接收用户音频，并把机器人音频发回客户端。

当前 quickstart 中使用的是：

- `DailyTransport`
- `SmallWebRTCTransport`

在 Pipecat 中，transport 会暴露两个处理器：

- `transport.input()`
- `transport.output()`

这两个处理器会被直接插入整个 pipeline。

### 2. STT 层

当前 bot 使用 `DeepgramSTTService`。

职责包括：

- 接收 transport 输入的流式音频
- 执行中文语音识别
- 把识别结果以 transcript frame 的形式送入后续处理链

当前配置：

- model: `nova-3-general`
- language: `Language.ZH`

### 3. 回合检测与上下文聚合层

当前 bot 使用 `LLMContextAggregatorPair`，并结合 `SileroVADAnalyzer` 做回合切分。

职责包括：

- 检测用户何时开始说话、何时停止说话
- 将识别到的文本聚合成完整用户回合
- 维护 LLM 使用的上下文
- 在机器人说完之后，把 assistant 回合写回上下文

当前 VAD 调优参数用于减少过早截断：

- `stop_secs=0.3`
- `confidence=0.6`

### 4. LLM 层

当前 bot 使用 `OpenAILLMService`，接入一个 OpenAI 兼容接口。

职责包括：

- 接收聚合后的对话上下文
- 生成助理文本回复
- 遵循适合中文口语播报的系统提示词

当前行为：

- 使用 `ARK_BASE_URL`
- 使用 `ARK_MODEL`
- 通过 `extra_body.thinking.type = disabled` 关闭推理输出
- 保持回答简洁，适合语音朗读

### 5. TTS 层

当前 bot 使用一个自定义的 `MiniMaxWSTTSService`。

职责包括：

- 与 MiniMax TTS 建立 WebSocket 连接
- 流式接收 PCM 音频块
- 生成 `TTSAudioRawFrame` 回传给 transport

之所以采用自定义实现，主要是为了：

- 更适合中文语音输出
- 利用流式 WebSocket TTS 降低延迟
- 兼顾国内网络可达性

### 6. Pipeline 层

当前 pipeline 是一个线性处理链：

1. `transport.input()`
2. `stt`
3. `user_aggregator`
4. `llm`
5. `tts`
6. `transport.output()`
7. `assistant_aggregator`

这就是 Pipecat 的关键架构模式：

- 每个组件都是一个 frame processor
- 所有 processor 被串成一条 pipeline
- frame 在链路中向下游或上游流动

### 7. Task 与 Runner 层

当前 bot 用 `PipelineTask` 包裹 pipeline，再由 `PipelineRunner` 驱动执行。

职责包括：

- 启动与停止整条 pipeline
- 管理 metrics 与 usage metrics
- 连接 observer 和生命周期事件
- 在连接断开时取消任务
- 集成 RTVI ready 事件

当前 quickstart 的首轮回复是在 `task.rtvi.on_client_ready` 触发后，通过 `LLMRunFrame()` 拉起的。

## 当前对话流程

运行时整体流程如下：

1. 客户端通过 WebRTC 建立连接。
2. Transport 接收用户麦克风音频。
3. Deepgram 将音频转成文本。
4. 用户聚合器把 transcript 聚合成一个完整用户回合。
5. LLM 读取当前上下文并生成回答文本。
6. MiniMax TTS 将回答文本转换为 PCM 音频流。
7. Transport 把语音结果发回客户端。
8. Assistant 聚合器把机器人回合写回上下文。

这套架构已经具备较好的模块化能力，因为每一层都可以单独替换或扩展。

## 当前框架的优势

- Transport、STT、LLM、TTS 分层清晰。
- 语音基础设施与业务逻辑耦合较低。
- 已具备基于 VAD 和聚合器的回合管理能力。
- 更换服务供应商时不需要改动整体架构形态。
- 很适合在现有语音能力上叠加 tool calling 和业务状态管理。

## 当前框架在面试场景下的缺口

当前 bot 仍然是一个通用语音助手，还缺少下面这些面试系统必需的能力：

- 显式的面试 session 状态
- 题库管理
- 可控的抽题逻辑
- 按题记录回答
- 评分与评价逻辑
- 面试报告生成
- 基于岗位或主题的面试编排能力

所以现在这套框架更适合被看作“语音运行时底座”，而不是完整的面试产品。

## 面试助手的推荐改造方向

### 总体建议

推荐方向是：

- 保留当前级联式语音链路
- 在 LLM 之上增加面试编排层
- 用 tool calling 管理抽题、记录回答和评分
- 把 LLM 当作“面试官”，不要把它当作“唯一状态存储”

换句话说：

- 语音基础设施继续由 Pipecat processors 负责
- 面试状态交给服务端显式业务对象管理
- prompt 与 tools 协同工作，但面试进度必须由代码控制

### 为什么建议保留当前级联链路

对于面试助手，当前这条 `STT -> 回合聚合 -> LLM -> TTS` 的链路，通常比完全托管式 speech-to-speech 更合适。

原因有：

- 更容易检查 transcript 和候选人答案
- 更容易按题持久化结果
- 更容易插入确定性的业务规则
- 后续替换评分策略或抽题策略更容易
- 对打断、重问、结束条件的控制更强

相比微小的延迟差异，这些能力在面试场景里更重要。

## 推荐目标架构

### 1. Voice Runtime 层

这一层建议基本保持不变：

- transport
- STT
- VAD 与 turn aggregation
- LLM service
- TTS
- pipeline task 与 runner

这层应继续只负责实时媒体和对话回合处理。

### 2. Interview Session 层

建议新增一个专门的服务端 session 对象，例如：

- `InterviewSession`

建议承担的字段与职责：

- session id
- 候选人元信息
- 目标岗位
- 目标级别
- 面试模式
- 当前题目 id
- 已提问题目 id 列表
- 回答历史
- 分数记录
- 当前面试状态

这个对象应该成为面试过程的唯一真实状态来源。

不要仅依赖 prompt 记忆来追踪：

- 当前正在问哪道题
- 回答是否已经完整
- 是否已经追问过
- 是否应该进入下一个主题

### 3. 题库层

建议引入结构化题库，而不是把题目直接写死在系统提示词里。

建议的数据模型：

- `Question`
- `QuestionBank`
- `QuestionSelector`

每道题建议包含：

- `id`
- `role`
- `level`
- `topic`
- `difficulty`
- `question_text`
- `expected_points`
- `follow_up_prompts`
- `scoring_rubric`

第一阶段可以先用：

- 本地 JSON 或 YAML 文件

后续再演进到：

- SQLite
- Postgres
- 外部 CMS 或后台配置系统

### 4. 面试编排层

建议在业务状态与 LLM 之间增加一个编排组件，例如：

- `InterviewOrchestrator`

建议职责：

- 启动面试
- 选择下一题
- 决定是否追问
- 判断当前回答是否足够
- 结束面试
- 生成总结与报告

这一层应该拥有面试状态机。

建议状态包括：

- `idle`
- `intro`
- `asking_question`
- `waiting_for_answer`
- `asking_follow_up`
- `evaluating_answer`
- `completed`

### 5. Tool Calling 层

建议把 tool calling 作为“自然语言面试官”和“确定性业务逻辑”之间的桥梁。

第一批推荐工具：

- `draw_question(role, level, topic, asked_ids)`
- `save_answer(question_id, transcript, metadata)`
- `score_answer(question_id, transcript)`
- `complete_interview(session_id)`

推荐行为方式：

- LLM 先通过 tool 请求一道题
- 后端返回明确的题目 payload
- LLM 用自然口语把题目问出来
- 候选人回答完成后，后端保存答案并按需要评分

这样可以让机器人保持自然语音交互，同时让系统行为仍然可控。

### 6. Transcript 与评估层

建议复用 Pipecat 现有的 turn 事件做记录与持久化。

当前框架里已经可用的关键 hook：

- `on_user_turn_stopped`
- `on_assistant_turn_stopped`

这些事件可用于：

- 记录用户回答 transcript
- 记录面试官提问内容
- 计算回答时长
- 在答案完整后触发评分逻辑

评分建议区分两种模式：

- 实时轻量评分，用于决定是否追问
- 延迟重评分，用于生成最终报告

这样可以避免评分逻辑阻塞实时语音交互。

### 7. 持久化层

建议为面试数据引入显式持久化。

建议存储的实体：

- interview sessions
- 已提问题目
- 原始 transcript
- 归一化后的回答文本
- 评分记录
- 最终报告

第一版使用 SQLite 就足够。

### 8. 报告层

建议在面试结束后增加报告构建能力。

建议输出：

- 每道题的回答摘要
- 每个主题的评分
- 优势项
- 薄弱项
- 后续学习建议
- 最终推荐意见

这一层可以在实时会话结束后执行，不必放在语音主链路上。

## 对当前 Quickstart Bot 的具体改造建议

### 最小重构方案

当前 `bot.py` 不需要大改，可以在保持装配职责的前提下逐步演进。

建议近一步的改造方式：

- 在 `pipecat-quickstart/server/` 下新增本地 `interview/` 包
- `bot.py` 继续作为 wiring 层
- 业务逻辑迁移到独立模块

建议的模块拆分：

- `server/interview/session.py`
- `server/interview/question_bank.py`
- `server/interview/orchestrator.py`
- `server/interview/tools.py`
- `server/interview/scoring.py`
- `server/interview/reporting.py`

### Prompt 策略

系统提示词不应该继续扮演“通用聊天助手”，而应该收敛为“受控面试官”。

建议提示词具备以下特征：

- 使用简洁的中文口语
- 每次只问一个问题
- 不暴露评分标准
- 不在没有工具结果的前提下擅自跳题
- 简短确认并平滑过渡
- 仅在业务逻辑或工具结果指示时进行追问

### Turn Management 策略

面试场景对回合控制的要求比普通闲聊更高。

推荐：

- 保留当前 Silero VAD
- 评估启用不完整回答过滤
- 当候选人长时间沉默时增加 idle prompt
- 区分“短暂停顿”和“答案真正结束”

这是因为候选人回答通常比普通对话更长、更容易出现思考停顿。

## 推荐实施阶段

### Phase 1：POC

目标：

- 跑通一个单岗位 5 道题的模拟面试

范围：

- 本地题库文件
- 固定流程
- 保存 transcript
- 不做复杂评分，只输出总结

### Phase 2：结构化面试

目标：

- 让面试过程具备可复现、可追踪、可复盘能力

范围：

- 显式 session 状态
- 确定性抽题
- 追问逻辑
- 按题评分
- 完整面试报告

### Phase 3：产品化面试助手

目标：

- 支持多种面试风格与个性化能力

范围：

- 基于岗位的题集
- 基于级别的评分 rubric
- 结合简历或 JD 做动态抽题
- reviewer dashboard 或导出能力
- 面试数据分析

## 推荐的第一步落地路径

如果目标是先低风险、快速推进，建议优先这样做：

1. 保持当前语音链路不变。
2. 增加一个本地题库。
3. 增加 interview session 状态对象。
4. 增加 `draw_question` 与 `save_answer` 两个基础工具。
5. 通过 turn 事件记录每次候选人回答。
6. 在面试结束后生成一个简版总结。

这条路径可以在不破坏当前实时语音体验的前提下，先补齐面试助手最关键的业务骨架。

## 个人练习版初版规划

如果当前目标只是个人日常练习，初版不建议继续走复杂产品化架构，而应该先收敛成一个最小可跑通闭环。

### 初版目标

初版只实现下面这个流程：

1. 系统按固定顺序完成 3 道题练习。
2. 用户逐题口头回答。
3. 系统记住每一题的题目和对应回答。
4. 三题全部结束后，统一给出一份总点评。

初版不做：

- 每题即时点评
- 复杂评分系统
- 数据库
- 后台管理
- 多用户
- 复杂 tool calling
- 产品化状态机

### 初版的 3 类题型

固定为下面三类：

1. 综合分析
2. 计划组织 / 人际沟通
3. 应急应变 / 情景模拟

每轮练习一共 3 题，每类随机抽 1 题，顺序固定。

### 初版框架

初版框架建议收敛成两层：

#### 1. 语音运行层

直接保留当前链路，不做大改：

`Transport -> STT -> 用户回合聚合 -> LLM -> TTS -> Transport`

继续复用：

- WebRTC transport
- Deepgram STT
- Silero VAD
- ARK / Doubao LLM
- MiniMax TTS

#### 2. 练习控制层

只增加一个极轻量的会话对象，例如：

- `PracticeSession`

建议只负责：

- 当前第几题
- 本轮 3 道题的题目内容
- 用户的 3 次回答
- 是否已完成
- 最终点评内容

这层不承担复杂业务，只负责把本轮训练跑通。

### 初版需要具备的功能

初版功能只保留这些：

- 从本地小题库中抽出 3 道题
- 按固定顺序播报题目
- 自动记录题目
- 自动记录用户回答
- 三题结束后生成统一总点评
- 语音播报最终总点评

### 题库建议

初版不需要做完整题库系统，只需要一个本地小题库文件。

可以直接用：

- Python 常量
- JSON 文件

题库按三类组织即可，每类准备若干题目。

### 题目与答案的记忆方式

初版的“记忆”建议只做最小实现：

- `questions`: 记录本轮三道题
- `answers`: 按顺序记录三次回答

也就是说，系统要明确知道：

- 第一题问了什么，用户怎么答
- 第二题问了什么，用户怎么答
- 第三题问了什么，用户怎么答

这样在最后生成点评时，就能基于本轮完整记录做统一评价。

### 总点评的生成方式

初版只保留“最后统一点评”，不做逐题点评。

建议总点评固定覆盖这几个方面：

- 总体表现
- 亮点
- 主要问题
- 改进建议

点评风格建议：

- 简洁
- 适合语音播报
- 更像个人陪练教练
- 少空话，直接指出问题

### 初版代码结构建议

初版建议最多拆成两个或三个文件：

- `server/bot.py`
- `server/interview_practice.py`
- 可选：`server/question_bank.py`

其中：

- `bot.py` 负责语音链路和整体装配
- `interview_practice.py` 负责题库、会话状态、流程控制、最终点评 prompt

### 推荐的最小闭环

初版最终应实现的最小闭环是：

1. 进入练习。
2. 系统出第 1 题。
3. 用户回答第 1 题。
4. 系统出第 2 题。
5. 用户回答第 2 题。
6. 系统出第 3 题。
7. 用户回答第 3 题。
8. 系统根据 3 道题和 3 次回答，生成统一总点评。
9. 系统播报总点评。

这就是当前个人练习版初版最合适的框架边界。
