//! Claude Code CLI provider — invokes `claude --print` as a subprocess.
//!
//! This provider delegates to the Claude Code CLI, which uses the user's
//! subscription credentials and provides its own tool set (Bash, Edit, Read,
//! Grep, etc.). Moltis injects conversation context via `--append-system-prompt`
//! so Claude Code's built-in system prompt and tools remain available.
//!
//! ## When to use
//!
//! Use this provider when you want to leverage a Claude subscription (no API key
//! needed) and Claude Code's built-in agent capabilities. It is **not** suitable
//! for Moltis-native tool calling since the subprocess runs its own agent loop.

use std::pin::Pin;

use {async_trait::async_trait, tokio_stream::Stream};

use tracing::{debug, trace, warn};

use moltis_agents::model::{
    ChatMessage, CompletionResponse, LlmProvider, StreamEvent, Usage, UserContent,
};

/// A provider that delegates to `claude --print` as a subprocess.
pub struct ClaudeCliProvider {
    /// Model to request (e.g. "claude-sonnet-4-6", "claude-opus-4-6").
    model: String,
    /// Optional system prompt to append via `--append-system-prompt`.
    system_prompt: Option<String>,
    /// Path to the `claude` binary. Defaults to "claude" (resolved via PATH).
    claude_binary: String,
}

impl ClaudeCliProvider {
    /// Create a new provider for the given model.
    pub fn new(model: String) -> Self {
        Self {
            model,
            system_prompt: None,
            claude_binary: "claude".into(),
        }
    }

    /// Set an optional system prompt that will be appended to Claude Code's
    /// default system prompt via `--append-system-prompt`.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    /// Override the path to the `claude` binary.
    #[must_use]
    pub fn with_binary(mut self, path: String) -> Self {
        self.claude_binary = path;
        self
    }

    /// Build the CLI arguments for a `claude --print` invocation.
    fn build_args(&self, prompt: &str) -> Vec<String> {
        let mut args = vec![
            "--print".into(),
            "--output-format".into(),
            "stream-json".into(),
            "--model".into(),
            self.model.clone(),
            "--no-session-persistence".into(),
            "--verbose".into(),
        ];

        if let Some(ref sys) = self.system_prompt {
            args.push("--append-system-prompt".into());
            args.push(sys.clone());
        }

        args.push(prompt.into());
        args
    }
}

/// Serialize a message list into a single prompt string for `claude --print`.
///
/// Claude CLI takes a single text prompt. We flatten the conversation:
/// - System messages are collected into `--append-system-prompt` (handled at call site).
/// - The last user message becomes the main prompt argument.
/// - Preceding messages are formatted as conversation context prepended to the prompt.
fn messages_to_prompt(messages: &[ChatMessage]) -> (Option<String>, String) {
    let mut system_parts = Vec::new();
    let mut conversation = Vec::new();
    let mut last_user_text: Option<String> = None;

    for msg in messages {
        match msg {
            ChatMessage::System { content } => {
                system_parts.push(content.clone());
            },
            ChatMessage::User { content } => {
                // If there was a previous user message, push it to conversation history.
                if let Some(prev) = last_user_text.take() {
                    conversation.push(format!("User: {prev}"));
                }
                last_user_text = Some(match content {
                    UserContent::Text(text) => text.clone(),
                    UserContent::Multimodal(parts) => parts
                        .iter()
                        .filter_map(|p| match p {
                            moltis_agents::model::ContentPart::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                });
            },
            ChatMessage::Assistant {
                content,
                tool_calls: _,
            } => {
                if let Some(prev_user) = last_user_text.take() {
                    conversation.push(format!("User: {prev_user}"));
                }
                if let Some(text) = content {
                    conversation.push(format!("Assistant: {text}"));
                }
            },
            ChatMessage::Tool {
                tool_call_id: _,
                content,
            } => {
                conversation.push(format!("Tool result: {content}"));
            },
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    let prompt = if conversation.is_empty() {
        last_user_text.unwrap_or_default()
    } else {
        let history = conversation.join("\n\n");
        let user_msg = last_user_text.unwrap_or_default();
        format!("<conversation_history>\n{history}\n</conversation_history>\n\n{user_msg}")
    };

    (system, prompt)
}

/// Parse a stream-json line from Claude Code and extract a text delta.
///
/// Claude Code stream-json format (NDJSON):
/// - `{"type":"system","subtype":"init",...}` — session start
/// - `{"type":"assistant","message":{"content":[{"type":"text","text":"..."},...]},...}` — complete message
/// - `{"type":"result","subtype":"success","total_cost_usd":...,"duration_ms":...}` — session end
fn parse_stream_event(line: &str) -> Option<StreamEvent> {
    let event: serde_json::Value = serde_json::from_str(line).ok()?;
    let event_type = event["type"].as_str()?;

    match event_type {
        "assistant" => {
            // Complete assistant message — extract text content.
            let content = event["message"]["content"].as_array()?;
            let text: String = content
                .iter()
                .filter_map(|block| {
                    if block["type"].as_str() == Some("text") {
                        block["text"].as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                None
            } else {
                Some(StreamEvent::Delta(text))
            }
        },
        "result" => {
            let is_success = event["subtype"].as_str() == Some("success");
            if is_success {
                Some(StreamEvent::Done(Usage::default()))
            } else {
                let subtype = event["subtype"].as_str().unwrap_or("unknown");
                Some(StreamEvent::Error(format!("Claude CLI session ended: {subtype}")))
            }
        },
        _ => None,
    }
}

#[async_trait]
impl LlmProvider for ClaudeCliProvider {
    fn name(&self) -> &str {
        "claude-cli"
    }

    fn id(&self) -> &str {
        &self.model
    }

    fn supports_tools(&self) -> bool {
        // Claude Code handles tools internally, but Moltis cannot inject
        // its own tool schemas into the subprocess, so we report false.
        false
    }

    fn context_window(&self) -> u32 {
        super::context_window_for_model(&self.model)
    }

    fn supports_vision(&self) -> bool {
        false
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        _tools: &[serde_json::Value],
    ) -> anyhow::Result<CompletionResponse> {
        let (extra_system, prompt) = messages_to_prompt(messages);

        // Build a merged system prompt if messages contained system content.
        let merged_system = match (&self.system_prompt, &extra_system) {
            (Some(base), Some(extra)) => Some(format!("{base}\n\n{extra}")),
            (Some(base), None) => Some(base.clone()),
            (None, Some(extra)) => Some(extra.clone()),
            (None, None) => None,
        };

        let mut args = vec![
            "--print".into(),
            "--output-format".into(),
            "json".to_string(),
            "--model".into(),
            self.model.clone(),
            "--no-session-persistence".into(),
        ];

        if let Some(ref sys) = merged_system {
            args.push("--append-system-prompt".into());
            args.push(sys.clone());
        }

        args.push(prompt.clone());

        debug!(
            model = %self.model,
            prompt_len = prompt.len(),
            has_system = merged_system.is_some(),
            "claude-cli complete request"
        );
        trace!(prompt = %prompt, "claude-cli prompt");

        let output = tokio::process::Command::new(&self.claude_binary)
            .args(&args)
            .env_remove("CLAUDECODE")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?
            .wait_with_output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code = output.status.code().unwrap_or(-1);
            anyhow::bail!("claude CLI exited with code {code}: {stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| anyhow::anyhow!("failed to parse claude CLI JSON output: {e}"))?;

        let text = result["result"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| {
                // Fallback: try raw text output.
                let s = stdout.trim().to_string();
                if s.is_empty() { None } else { Some(s) }
            });

        let cost = result["cost_usd"].as_f64().unwrap_or(0.0);
        debug!(
            model = %self.model,
            cost_usd = cost,
            text_len = text.as_ref().map_or(0, |t| t.len()),
            "claude-cli complete response"
        );

        Ok(CompletionResponse {
            text,
            tool_calls: vec![],
            usage: Usage::default(),
        })
    }

    fn stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        self.stream_with_tools(messages, vec![])
    }

    fn stream_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        _tools: Vec<serde_json::Value>,
    ) -> Pin<Box<dyn Stream<Item = StreamEvent> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let (extra_system, prompt) = messages_to_prompt(&messages);

            // Build a merged system prompt if messages contained system content.
            let merged_system = match (&self.system_prompt, &extra_system) {
                (Some(base), Some(extra)) => Some(format!("{base}\n\n{extra}")),
                (Some(base), None) => Some(base.clone()),
                (None, Some(extra)) => Some(extra.clone()),
                (None, None) => None,
            };

            let provider = ClaudeCliProvider {
                model: self.model.clone(),
                system_prompt: merged_system,
                claude_binary: self.claude_binary.clone(),
            };

            let args = provider.build_args(&prompt);

            debug!(
                model = %self.model,
                prompt_len = prompt.len(),
                has_system = provider.system_prompt.is_some(),
                "claude-cli stream request"
            );
            trace!(prompt = %prompt, "claude-cli stream prompt");

            let child = match tokio::process::Command::new(&self.claude_binary)
                .args(&args)
                .env_remove("CLAUDECODE")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    yield StreamEvent::Error(format!("failed to spawn claude CLI: {e}"));
                    return;
                }
            };

            let stdout = match child.stdout {
                Some(stdout) => stdout,
                None => {
                    yield StreamEvent::Error("claude CLI stdout not captured".into());
                    return;
                }
            };

            let reader = tokio::io::BufReader::new(stdout);
            use tokio::io::AsyncBufReadExt;
            let mut lines = reader.lines();

            let mut got_done = false;

            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                trace!(line = %line, "claude-cli stream line");

                if let Some(event) = parse_stream_event(&line) {
                    match &event {
                        StreamEvent::Done(_) | StreamEvent::Error(_) => {
                            got_done = true;
                            yield event;
                            return;
                        },
                        _ => yield event,
                    }
                }
            }

            // If we got here without a Done event, the process may have exited abnormally.
            if !got_done {
                warn!("claude CLI stream ended without result event");
                yield StreamEvent::Done(Usage::default());
            }
        })
    }
}

#[allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(test)]
mod tests {
    use super::*;

    // ── messages_to_prompt ──────────────────────────────────────────

    #[test]
    fn single_user_message() {
        let messages = vec![ChatMessage::user("Hello")];
        let (system, prompt) = messages_to_prompt(&messages);
        assert!(system.is_none());
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn system_plus_user() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let (system, prompt) = messages_to_prompt(&messages);
        assert_eq!(system.as_deref(), Some("You are helpful."));
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn multi_turn_conversation() {
        let messages = vec![
            ChatMessage::system("Be concise."),
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
            ChatMessage::user("And 3+3?"),
        ];
        let (system, prompt) = messages_to_prompt(&messages);
        assert_eq!(system.as_deref(), Some("Be concise."));
        assert!(prompt.contains("User: What is 2+2?"));
        assert!(prompt.contains("Assistant: 4"));
        assert!(prompt.contains("And 3+3?"));
        // The last user message should NOT be prefixed with "User:"
        assert!(!prompt.ends_with("User: And 3+3?"));
    }

    #[test]
    fn multiple_system_messages_concatenated() {
        let messages = vec![
            ChatMessage::system("Rule 1"),
            ChatMessage::system("Rule 2"),
            ChatMessage::user("Hello"),
        ];
        let (system, prompt) = messages_to_prompt(&messages);
        assert_eq!(system.as_deref(), Some("Rule 1\n\nRule 2"));
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn tool_results_in_history() {
        let messages = vec![
            ChatMessage::user("Run ls"),
            ChatMessage::assistant_with_tools(
                Some("Let me run that.".into()),
                vec![moltis_agents::model::ToolCall {
                    id: "call_1".into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({"cmd": "ls"}),
                }],
            ),
            ChatMessage::tool("call_1", "file.txt"),
            ChatMessage::user("What files are there?"),
        ];
        let (_, prompt) = messages_to_prompt(&messages);
        assert!(prompt.contains("User: Run ls"));
        assert!(prompt.contains("Assistant: Let me run that."));
        assert!(prompt.contains("Tool result: file.txt"));
        assert!(prompt.contains("What files are there?"));
    }

    #[test]
    fn empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let (system, prompt) = messages_to_prompt(&messages);
        assert!(system.is_none());
        assert_eq!(prompt, "");
    }

    // ── parse_stream_event ──────────────────────────────────────────

    #[test]
    fn parse_assistant_event() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hello world"}]}}"#;
        let event = parse_stream_event(line);
        assert!(matches!(event, Some(StreamEvent::Delta(ref t)) if t == "Hello world"));
    }

    #[test]
    fn parse_result_success() {
        let line = r#"{"type":"result","subtype":"success","total_cost_usd":0.01,"duration_ms":5000,"num_turns":3}"#;
        let event = parse_stream_event(line);
        assert!(matches!(event, Some(StreamEvent::Done(_))));
    }

    #[test]
    fn parse_result_error() {
        let line = r#"{"type":"result","subtype":"error_max_budget_usd"}"#;
        let event = parse_stream_event(line);
        assert!(
            matches!(event, Some(StreamEvent::Error(ref msg)) if msg.contains("error_max_budget_usd"))
        );
    }

    #[test]
    fn parse_system_event_ignored() {
        let line = r#"{"type":"system","subtype":"init","session_id":"abc","model":"claude-sonnet-4-6"}"#;
        let event = parse_stream_event(line);
        assert!(event.is_none());
    }

    #[test]
    fn parse_invalid_json() {
        let event = parse_stream_event("not json at all");
        assert!(event.is_none());
    }

    #[test]
    fn parse_assistant_with_tool_use_blocks() {
        // When Claude uses tools, the assistant message has tool_use blocks.
        // We extract only text content and skip tool_use blocks.
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls"}}]}}"#;
        let event = parse_stream_event(line);
        assert!(matches!(event, Some(StreamEvent::Delta(ref t)) if t == "Let me check."));
    }

    #[test]
    fn parse_assistant_empty_text() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":""}]}}"#;
        let event = parse_stream_event(line);
        assert!(event.is_none());
    }

    // ── Provider metadata ───────────────────────────────────────────

    #[test]
    fn provider_metadata() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        assert_eq!(provider.name(), "claude-cli");
        assert_eq!(provider.id(), "claude-sonnet-4-6");
        assert!(!provider.supports_tools());
        assert!(!provider.supports_vision());
        assert_eq!(provider.context_window(), 200_000);
    }

    #[test]
    fn build_args_basic() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let args = provider.build_args("Hello");
        assert!(args.contains(&"--print".into()));
        assert!(args.contains(&"stream-json".into()));
        assert!(args.contains(&"claude-sonnet-4-6".into()));
        assert!(args.contains(&"Hello".into()));
        assert!(!args.contains(&"--append-system-prompt".into()));
    }

    #[test]
    fn build_args_with_system_prompt() {
        let provider =
            ClaudeCliProvider::new("claude-opus-4-6".into()).with_system_prompt("Be helpful".into());
        let args = provider.build_args("Hello");
        let sys_idx = args
            .iter()
            .position(|a| a == "--append-system-prompt")
            .unwrap();
        assert_eq!(args[sys_idx + 1], "Be helpful");
    }

    #[test]
    fn with_binary_overrides_path() {
        let provider =
            ClaudeCliProvider::new("claude-sonnet-4-6".into()).with_binary("/usr/local/bin/claude".into());
        assert_eq!(provider.claude_binary, "/usr/local/bin/claude");
    }
}
