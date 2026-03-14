# Frontend Module Spec

React/TypeScript web UI with Tauri desktop support, Zustand state management, and SSE streaming.

## Store (`lib/store.ts`)

### REQ-frontend.store.conversations: Conversation management
Zustand store with `loadConversations()`, `createConversation()`, `selectConversation()`, `deleteConversation()`. Persists to localStorage.

### REQ-frontend.store.messages: Message management
`addMessage()`, `updateLastAssistant()`, `loadMessages()`. Messages belong to conversations.

### REQ-frontend.store.settings: User settings
Settings: theme (light/dark/system), apiUrl, fontSize, defaultModel, defaultAgent, temperature, maxTokens, speechEnabled.

### REQ-frontend.store.agents: Managed agent state
`managedAgents`, `selectedAgentId`, `agentEvents` (max 100, FIFO).

### REQ-frontend.store.streaming: Stream state
`StreamState` with `isStreaming`, `phase`, `elapsedMs`, `activeToolCalls`, `content`.

## API Client (`lib/api.ts`)

### REQ-frontend.api.chat: Chat API
OpenAI-compatible chat completions via `POST /v1/chat/completions`.

### REQ-frontend.api.models: Model listing
`fetchModels()`, `fetchServerInfo()`.

### REQ-frontend.api.agents: Managed agent API
Full CRUD: `createManagedAgent()`, `fetchManagedAgents()`, `updateManagedAgent()`, `deleteManagedAgent()`, plus pause/resume/run/recover.

### REQ-frontend.api.speech: Speech API
`transcribeAudio(blob)`, `fetchSpeechHealth()`.

### REQ-frontend.api.tauri: Desktop IPC
`isTauri()` detection, `tauriInvoke()` for desktop-specific APIs.

## SSE Streaming (`lib/sse.ts`)

### REQ-frontend.sse.stream: SSE stream handler
`streamChat(request, signal?) -> AsyncGenerator<SSEEvent>` parses Server-Sent Events with buffered line reading.

## Hooks

### REQ-frontend.hooks.speech: Speech hook
`useSpeech()` hook with `startRecording()`, `stopRecording()`, MediaRecorder integration, and transcription via API.

## Utilities

### REQ-frontend.profanity: Content filter
`isProfane(text) -> boolean` checks against 60+ blocked words with normalization.

## Pages

### REQ-frontend.pages.chat: Chat page
### REQ-frontend.pages.dashboard: Dashboard page
### REQ-frontend.pages.agents: Agent management page
### REQ-frontend.pages.settings: Settings page

## Components

### REQ-frontend.components.message: Message rendering
`MessageBubble` with markdown, code highlighting, tool call display.

### REQ-frontend.components.error: Error boundary
`ErrorBoundary` for graceful error handling.

## Tests

- `frontend/src/**/*.test.ts(x)` - Frontend test files (to be created)
