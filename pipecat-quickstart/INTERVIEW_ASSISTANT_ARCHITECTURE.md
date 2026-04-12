# Pipecat Quickstart Current Framework

## Scope

This document describes the current real-time voice conversation framework implemented in `pipecat-quickstart/server/bot.py`.

The current bot is a cascade voice pipeline:

`Transport -> STT -> User Turn Aggregation -> LLM -> TTS -> Transport`

It is oriented toward general real-time Chinese voice conversation, not yet toward structured interviews.

## Current Entry And Runtime Model

The current entrypoint is `pipecat-quickstart/server/bot.py`.

Main runtime responsibilities:

- `bot(runner_args)`: selects the transport implementation.
- `run_bot(transport)`: assembles services, context, pipeline, and task lifecycle.
- `PipelineRunner`: runs the pipeline task until disconnected or cancelled.

The current bot supports:

- `DailyTransport` for Daily WebRTC rooms.
- `SmallWebRTCTransport` for lightweight WebRTC connections.

## Current Core Components

### 1. Transport Layer

The transport layer is responsible for receiving user audio and sending bot audio back to the client.

Current implementations used by the quickstart bot:

- `DailyTransport`
- `SmallWebRTCTransport`

In Pipecat, transports expose two processors:

- `transport.input()`
- `transport.output()`

These are inserted directly into the pipeline.

### 2. STT Layer

The current bot uses `DeepgramSTTService`.

Responsibilities:

- receive streaming audio from transport input
- perform Chinese speech recognition
- emit transcript frames into the pipeline

Configuration details in the current bot:

- model: `nova-3-general`
- language: `Language.ZH`

### 3. Turn Detection And Context Aggregation

The current bot uses `LLMContextAggregatorPair` with `SileroVADAnalyzer`.

Responsibilities:

- detect when the user has started and stopped speaking
- aggregate recognized user utterances into complete turns
- maintain conversation context for the LLM
- aggregate assistant output back into context after speaking

The current implementation tunes VAD to reduce premature cutoff:

- `stop_secs=0.3`
- `confidence=0.6`

### 4. LLM Layer

The current bot uses `OpenAILLMService` against an OpenAI-compatible endpoint.

Responsibilities:

- receive aggregated conversation context
- generate assistant text responses
- follow a system instruction optimized for spoken Chinese replies

Current behavior:

- uses `ARK_BASE_URL`
- uses `ARK_MODEL`
- disables reasoning output through `extra_body.thinking.type = disabled`
- keeps answers concise and speech-friendly

### 5. TTS Layer

The current bot uses a custom `MiniMaxWSTTSService`.

Responsibilities:

- open a WebSocket connection to MiniMax TTS
- stream synthesized PCM audio chunks
- emit `TTSAudioRawFrame` back to the transport

The current implementation is custom because it is optimized for:

- Chinese voice output
- lower latency through streaming WebSocket TTS
- domestic network accessibility

### 6. Pipeline Layer

The pipeline is created as a linear processor chain:

1. `transport.input()`
2. `stt`
3. `user_aggregator`
4. `llm`
5. `tts`
6. `transport.output()`
7. `assistant_aggregator`

This is the key architectural pattern in Pipecat:

- each component is a frame processor
- processors are linked into a pipeline
- frames move downstream or upstream through the chain

### 7. Task And Runner Layer

The current bot wraps the pipeline in a `PipelineTask` and runs it using `PipelineRunner`.

Responsibilities:

- start and stop the pipeline
- manage metrics and usage metrics
- connect observers and lifecycle handlers
- handle disconnect cancellation
- integrate RTVI readiness events

The quickstart bot currently starts the first response from `task.rtvi.on_client_ready`, then queues `LLMRunFrame()`.

## Current Conversation Flow

At runtime, the flow is:

1. The client connects over WebRTC.
2. The transport receives live microphone audio.
3. Deepgram converts audio to text.
4. The user aggregator groups transcripts into a complete user turn.
5. The LLM reads the current context and generates a spoken reply.
6. MiniMax TTS converts the reply text into PCM audio chunks.
7. The transport streams audio back to the client.
8. The assistant aggregator writes the assistant turn back into context.

This architecture is modular and already suitable for extension because each layer is replaceable.

## Current Strengths

- Clear separation between transport, STT, LLM, and TTS.
- Low coupling between voice infrastructure and business behavior.
- Existing turn management through VAD and context aggregators.
- Easy to swap providers without changing overall pipeline shape.
- Suitable for adding tool calling and custom business state on top.

## Current Gaps For Structured Interview Use Cases

The current bot is still a general conversational assistant and lacks:

- explicit interview session state
- question bank management
- deterministic question selection
- per-question answer recording
- scoring and evaluation logic
- report generation
- role or topic based interview orchestration

At this stage, the framework is a strong voice runtime foundation, but the business layer for interviews has not been introduced yet.

## Interview Assistant Recommended Framework

### Recommendation Summary

The recommended direction is:

- keep the current cascade voice pipeline
- add an interview orchestration layer above the LLM
- use tool calling for question selection, answer recording, and scoring
- treat the LLM as the interviewer, not as the system of record

In other words:

- voice infrastructure remains in Pipecat processors
- interview state becomes explicit server-side business logic
- prompt and tools cooperate, but interview progress is controlled by code

### Why Keep The Current Cascade Pipeline

For an interview assistant, the current `STT -> turn aggregation -> LLM -> TTS` pipeline is preferable to a fully provider-managed speech-to-speech path.

Reasons:

- easier to inspect transcripts and answers
- easier to persist per-question results
- easier to insert deterministic business logic
- easier to swap scoring or question selection strategies later
- better control over interruptions, retries, and end-of-question behavior

This is more important than shaving off a small amount of latency.

## Proposed Target Architecture

### 1. Voice Runtime Layer

Keep these parts mostly unchanged:

- transport
- STT
- VAD and turn aggregation
- LLM service
- TTS
- pipeline task and runner

This layer should stay responsible only for real-time media and conversation turn handling.

### 2. Interview Session Layer

Add a dedicated server-side session object, for example:

- `InterviewSession`

Suggested responsibilities:

- session id
- candidate metadata
- target role
- target level
- interview mode
- current question id
- asked question ids
- answer history
- scores
- interview status

This object should be the source of truth for interview progress.

Do not rely on prompt memory alone to track:

- which question is current
- whether an answer is complete
- whether a follow-up has already been asked
- whether the interview should move to the next topic

### 3. Question Bank Layer

Introduce a structured question bank instead of hardcoding prompts in the system instruction.

Suggested model:

- `Question`
- `QuestionBank`
- `QuestionSelector`

Each question should ideally contain:

- `id`
- `role`
- `level`
- `topic`
- `difficulty`
- `question_text`
- `expected_points`
- `follow_up_prompts`
- `scoring_rubric`

Initial storage can be:

- local JSON or YAML file

Later it can move to:

- SQLite
- Postgres
- external CMS or admin backend

### 4. Interview Orchestrator Layer

Add an orchestration component between business state and the LLM, for example:

- `InterviewOrchestrator`

Suggested responsibilities:

- start interview
- select next question
- decide whether to ask a follow-up
- decide whether the current answer is sufficient
- end interview
- build summary and report

This layer should own the interview state machine.

Suggested states:

- `idle`
- `intro`
- `asking_question`
- `waiting_for_answer`
- `asking_follow_up`
- `evaluating_answer`
- `completed`

### 5. Tool Calling Layer

Use LLM tool calling as the bridge between the conversational interviewer and deterministic interview logic.

Recommended first tools:

- `draw_question(role, level, topic, asked_ids)`
- `save_answer(question_id, transcript, metadata)`
- `score_answer(question_id, transcript)`
- `complete_interview(session_id)`

Recommended behavior:

- the LLM asks for a question through a tool
- the backend returns a concrete question payload
- the LLM presents the question naturally in speech
- after the candidate answer is complete, the backend saves and optionally scores it

This allows the assistant to sound natural while the system remains controllable.

### 6. Transcript And Evaluation Layer

Reuse Pipecat turn events for logging and persistence.

Useful hooks already supported by the framework:

- `on_user_turn_stopped`
- `on_assistant_turn_stopped`

These should be used to:

- record the user answer transcript
- track interviewer prompts
- measure response duration
- trigger scoring after a completed answer

For scoring, keep two modes separated:

- real-time light scoring for follow-up decisions
- deferred heavier scoring for final report generation

This avoids slowing down spoken interaction.

### 7. Persistence Layer

Introduce explicit persistence for interview data.

Suggested entities:

- interview sessions
- questions asked
- raw transcripts
- normalized answers
- score records
- final reports

For the first version, SQLite is enough.

### 8. Reporting Layer

Add a report builder after interview completion.

Suggested outputs:

- per-question answer summary
- per-topic score
- strengths
- weaknesses
- follow-up learning suggestions
- final recommendation

This layer can run after the live session ends and does not need to sit on the critical audio path.

## Recommended Changes To The Current Quickstart Bot

### Minimal Refactor

The current `bot.py` can evolve with limited structural change.

Suggested near-term additions:

- add a local `interview/` package under `pipecat-quickstart/server/`
- keep `bot.py` as the wiring layer
- move business logic into dedicated modules

Suggested module split:

- `server/interview/session.py`
- `server/interview/question_bank.py`
- `server/interview/orchestrator.py`
- `server/interview/tools.py`
- `server/interview/scoring.py`
- `server/interview/reporting.py`

### Prompt Strategy

The system instruction should stop behaving like a generic assistant and instead define a narrow interviewer role.

Suggested prompt characteristics:

- concise spoken Chinese
- one question at a time
- do not reveal rubric
- do not jump topics without tool result
- confirm and transition briefly
- ask follow-up only when instructed by business logic or tool output

### Turn Management Strategy

The interview scenario benefits from stronger turn control than casual chat.

Recommended options:

- keep current Silero VAD
- consider enabling incomplete-turn filtering
- use idle prompts when the candidate stays silent too long
- distinguish between brief pause and true answer completion

This is important because candidate answers are usually longer than normal chat turns.

## Suggested Implementation Phases

### Phase 1: POC

Goal:

- run a 5-question mock interview for one role

Scope:

- local question bank file
- fixed interview flow
- save transcripts
- no advanced scoring, only summary

### Phase 2: Structured Interview

Goal:

- make the interview reproducible and reviewable

Scope:

- explicit session state
- deterministic question selection
- follow-up logic
- per-question scoring
- interview completion report

### Phase 3: Productized Interview Assistant

Goal:

- support multiple interview styles and personalization

Scope:

- role-based question sets
- level-based rubrics
- resume or JD driven question adaptation
- reviewer dashboard or export
- analytics on sessions and scores

## Recommended First Implementation Path

If the goal is to move quickly with low risk, the first practical step is:

1. keep the current voice stack unchanged
2. add a local question bank
3. add interview session state
4. add tool calling for `draw_question` and `save_answer`
5. log each user answer from turn events
6. generate a simple summary at the end

This path preserves the current real-time conversation experience while introducing the minimum business architecture needed for an interview assistant.
