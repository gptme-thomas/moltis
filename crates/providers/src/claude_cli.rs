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

use std::{pin::Pin, sync::Mutex};

use {async_trait::async_trait, tokio_stream::Stream, uuid::Uuid};

use tracing::{debug, info, trace, warn};

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
    /// Working directory for spawned `claude` processes.
    /// When set, the subprocess starts in this directory.
    working_dir: Option<String>,
    /// Command to run before each fresh session to generate additional context.
    /// The command's stdout is appended to the system prompt.
    /// Runs in `working_dir` if set, otherwise inherits the gateway's cwd.
    context_command: Option<String>,
    /// Tracks the active session UUID for `--resume` across multi-turn tool loops.
    /// `None` means no active session (next call starts fresh with `--session-id`).
    active_session: Mutex<Option<String>>,
    /// Number of messages in the Moltis history when this provider last ran.
    /// Used to detect when other providers have added messages (model switching)
    /// so we can clear the stale Claude CLI session and start fresh.
    last_seen_msg_count: Mutex<usize>,
}

impl ClaudeCliProvider {
    /// Create a new provider for the given model.
    pub fn new(model: String) -> Self {
        Self {
            model,
            system_prompt: None,
            claude_binary: "claude".into(),
            working_dir: None,
            context_command: None,
            active_session: Mutex::new(None),
            last_seen_msg_count: Mutex::new(0),
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

    /// Set the working directory for spawned `claude` processes.
    #[must_use]
    pub fn with_working_dir(mut self, dir: String) -> Self {
        self.working_dir = Some(dir);
        self
    }

    /// Set a command to run before each fresh session to generate context.
    /// The command's stdout is appended to the system prompt.
    #[must_use]
    pub fn with_context_command(mut self, cmd: String) -> Self {
        self.context_command = Some(cmd);
        self
    }

    /// Run the context command (if configured) and return its stdout.
    fn run_context_command(&self) -> Option<String> {
        let cmd = self.context_command.as_ref()?;

        let mut command = std::process::Command::new("bash");
        command.args(["-c", cmd]);
        if let Some(ref dir) = self.working_dir {
            command.current_dir(dir);
        }

        match command.output() {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout).to_string();
                if text.is_empty() {
                    warn!("context_command produced no output");
                    None
                } else {
                    info!(len = text.len(), "context_command produced dynamic context");
                    Some(text)
                }
            },
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                warn!(
                    exit_code = output.status.code(),
                    stderr = %stderr,
                    "context_command failed"
                );
                None
            },
            Err(e) => {
                warn!(error = %e, "failed to run context_command");
                None
            },
        }
    }

    /// Build CLI arguments and prompt for a `claude --print` invocation,
    /// handling session resume for multi-turn conversations and tool loops.
    ///
    /// Returns `(args, prompt_text)` where the prompt is **not** included
    /// in `args` — callers must deliver it via stdin to avoid hitting
    /// Linux's 128 KB per-argument limit (MAX_ARG_STRLEN / E2BIG).
    ///
    /// - **Resume** (active session exists): uses `--resume`, sends only the
    ///   new content — tool results or the latest user message.
    /// - **Fresh** (no active session): generates a new session UUID, uses
    ///   `--session-id`, sends the full flattened prompt.
    fn build_session_args(
        &self,
        messages: &[ChatMessage],
        output_format: &str,
    ) -> (Vec<String>, String) {
        let mut args = vec![
            "--print".into(),
            "--output-format".into(),
            output_format.into(),
            "--model".into(),
            self.model.clone(),
            "--verbose".into(),
            "--dangerously-skip-permissions".into(),
            "--disallowedTools".into(),
            DISALLOWED_TOOLS.join(","),
        ];

        // Enable token-level streaming for stream-json output.
        if output_format == "stream-json" {
            args.push("--include-partial-messages".into());
        }

        // Detect fresh conversation: if there are no assistant messages in the
        // history, this is a new Moltis session (e.g. after /new). Clear any
        // stale Claude CLI session so we don't resume into old context.
        let is_fresh_conversation = !messages
            .iter()
            .any(|m| matches!(m, ChatMessage::Assistant { .. }));

        let existing = self
            .active_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        if is_fresh_conversation && existing.is_some() {
            info!("claude-cli: fresh conversation detected, clearing stale session");
            self.clear_session();
        }

        let existing = self
            .active_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let last_count = *self
            .last_seen_msg_count
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Detect model-switch divergence: if the Moltis history grew by more
        // than 2 messages (user + assistant) since our last run, another
        // provider handled turns we haven't seen. Clear the CLI session so we
        // start fresh with the full conversation history.
        if existing.is_some() && last_count > 0 && messages.len() > last_count + 2 {
            info!(
                last_count,
                current_count = messages.len(),
                "claude-cli: history diverged (model switch detected), starting fresh session"
            );
            self.clear_session();
        }

        let existing = self
            .active_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(sid) = existing {
            args.push("--resume".into());
            args.push(sid.clone());
            let prompt = new_content_for_resume(messages);
            info!(
                session_id = %sid,
                prompt_len = prompt.len(),
                "claude-cli: resuming session"
            );
            *self
                .last_seen_msg_count
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = messages.len();
            return (args, prompt);
        }

        // No active session — start fresh with the full flattened prompt.
        let (extra_system, prompt) = messages_to_prompt(messages);
        let session_id = Uuid::new_v4().to_string();

        args.push("--session-id".into());
        args.push(session_id.clone());

        *self
            .active_session
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(session_id.clone());

        // Record how many messages are in the history at session creation,
        // so we can detect model-switch divergence on subsequent calls.
        *self
            .last_seen_msg_count
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = messages.len();

        // Merge system prompts from struct config and from messages.
        let merged_system = match (&self.system_prompt, &extra_system) {
            (Some(base), Some(extra)) => Some(format!("{base}\n\n{extra}")),
            (Some(base), None) => Some(base.clone()),
            (None, Some(extra)) => Some(extra.clone()),
            (None, None) => None,
        };

        let merged_system = merged_system.map(|s| adapt_system_prompt_for_cli(&s));

        // Append dynamic context from context_command (if configured).
        let dynamic_context = self.run_context_command();
        let final_system = match (merged_system, dynamic_context) {
            (Some(sys), Some(ctx)) => Some(format!("{sys}\n\n{ctx}")),
            (Some(sys), None) => Some(sys),
            (None, Some(ctx)) => Some(ctx),
            (None, None) => None,
        };

        let has_system_prompt = final_system.is_some();
        let system_prompt_len = final_system.as_ref().map_or(0, |s| s.len());
        if let Some(sys) = final_system {
            args.push("--append-system-prompt".into());
            args.push(sys);
        }

        info!(
            session_id = %session_id,
            has_system_prompt,
            system_prompt_len,
            prompt_len = prompt.len(),
            msg_count = messages.len(),
            "claude-cli: starting fresh session"
        );

        (args, prompt)
    }

    /// Clear the active session (e.g. on error, so the next call starts fresh).
    fn clear_session(&self) {
        *self
            .active_session
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self
            .last_seen_msg_count
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = 0;
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

/// Concrete example appended after the Moltis tool list so the model sees a
/// realistic call/response pair and is more likely to follow the format.
const MOLTIS_TOOL_EXAMPLE: &str = "\
### Example: using a Moltis tool\n\
\n\
User: What do you know about my health data?\n\
Assistant: Let me search your memory for health-related information.\n\
```tool_call\n\
{\"tool\": \"memory_search\", \"arguments\": {\"query\": \"health\"}}\n\
```\n\
\n\
*(Moltis executes the tool and returns the result in your next turn.)*\n\
\n";

/// Claude Code built-in tools that overlap with Moltis tools or are otherwise
/// inappropriate when running inside Moltis (e.g. interactive-only tools).
const DISALLOWED_TOOLS: &[&str] = &["Bash", "AskUserQuestion", "EnterPlanMode", "ExitPlanMode"];

/// Adapt the generic Moltis system prompt for use as a Claude CLI addendum.
///
/// Claude CLI already has its own system prompt and built-in tools (Bash, Read,
/// Write, etc.). The Moltis prompt is appended via `--append-system-prompt`, so
/// we transform it to:
/// - Remove the generic "You are a helpful assistant" intro (Claude CLI has its own)
/// - Reframe tool descriptions as *additional* Moltis platform tools
/// - Replace generic `tool_call` guidance with Claude-CLI-specific instructions
///   that explain these are extra tools whose results will be returned by the runtime
fn adapt_system_prompt_for_cli(prompt: &str) -> String {
    let mut out = String::with_capacity(prompt.len() + 512);

    // Replace the generic intro.
    out.push_str(
        "The following is additional context from the Moltis platform that hosts this conversation.\n\n",
    );

    let mut lines = prompt.lines().peekable();
    let mut in_how_to_call = false;
    let mut saw_tools_section = false;

    while let Some(line) = lines.next() {
        // Skip the generic assistant identity line.
        if line.starts_with("You are a helpful assistant") {
            // Also skip the blank line after it.
            if lines.peek().is_some_and(|l| l.is_empty()) {
                lines.next();
            }
            continue;
        }

        // Replace "## Available Tools" heading with Moltis-specific framing.
        if line == "## Available Tools" {
            saw_tools_section = true;
            out.push_str("## Additional Moltis Tools\n\n");
            out.push_str(
                "In addition to your built-in Claude Code tools, the Moltis platform provides \
                 these additional tools. To call one, output a fenced `tool_call` code block:\n\n",
            );
            out.push_str("```tool_call\n");
            out.push_str("{\"tool\": \"<tool_name>\", \"arguments\": {<arguments>}}\n");
            out.push_str("```\n\n");
            out.push_str(
                "The Moltis runtime will execute the tool and return the result to you in your \
                 next turn. You can then continue your response. One tool call per block; you may \
                 include multiple blocks.\n\n",
            );
            // Skip the blank line after the original heading.
            if lines.peek().is_some_and(|l| l.is_empty()) {
                lines.next();
            }
            continue;
        }

        // Skip the generic "## How to call tools" section entirely
        // (we already included guidance above).
        if line == "## How to call tools" {
            in_how_to_call = true;
            continue;
        }
        if in_how_to_call {
            // End of section: next top-level heading.
            if line.starts_with("## ") {
                in_how_to_call = false;
                // Fall through to emit this line normally.
            } else {
                continue;
            }
        }

        // Insert a concrete example right before the Guidelines section ends
        // the tools block, so Claude sees it immediately after the tool list.
        if saw_tools_section && line == "## Guidelines" {
            out.push_str(MOLTIS_TOOL_EXAMPLE);
            saw_tools_section = false;
        }

        out.push_str(line);
        out.push('\n');
    }

    // If there was no Guidelines section, append the example at the end.
    if saw_tools_section {
        out.push_str(MOLTIS_TOOL_EXAMPLE);
    }

    out
}

/// Extract only the new content to send on a `--resume` call.
///
/// - If the last message is a `Tool` result: sends tool results since the last
///   assistant message.
/// - If the last message is a `User` message: sends just the user text.
/// - Otherwise: sends "Continue."
fn new_content_for_resume(messages: &[ChatMessage]) -> String {
    match messages.last() {
        Some(ChatMessage::Tool { .. }) => tool_results_since_last_assistant(messages),
        Some(ChatMessage::User { content }) => match content {
            UserContent::Text(text) => text.clone(),
            UserContent::Multimodal(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    moltis_agents::model::ContentPart::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        },
        _ => "Continue.".into(),
    }
}

/// Extract only the tool results since the last assistant message.
///
/// Walks backward from the end of the message list, collecting `Tool` messages
/// until an `Assistant` message is found. Returns a formatted prompt containing
/// only these new results, suitable for a `--resume` call.
fn tool_results_since_last_assistant(messages: &[ChatMessage]) -> String {
    let mut results = Vec::new();
    for msg in messages.iter().rev() {
        match msg {
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                results.push(format!("Tool result for {tool_call_id}:\n{content}"));
            },
            ChatMessage::Assistant { .. } => break,
            _ => {},
        }
    }
    results.reverse();
    if results.is_empty() {
        "Continue.".into()
    } else {
        let joined = results.join("\n\n");
        format!("{joined}\n\nContinue.")
    }
}

/// Parse a stream-json line from Claude Code into `StreamEvent`(s).
///
/// Returns a `Vec` because a single `assistant` snapshot may contain multiple
/// tool_use blocks, each producing its own `ObservedToolStart` event.
///
/// With `--include-partial-messages`, Claude Code emits granular NDJSON events:
///
/// - `stream_event` → `content_block_delta` → `text_delta` — token-level text delta
/// - `stream_event` → `content_block_delta` → `thinking_delta` — reasoning delta
/// - `assistant` — tool_use blocks → `ObservedToolStart` events
/// - `user` — tool_use_result → `ObservedToolEnd` events
/// - `result` — session end with usage
fn parse_stream_events(line: &str) -> Vec<StreamEvent> {
    let event: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let event_type = match event["type"].as_str() {
        Some(t) => t,
        None => return vec![],
    };

    match event_type {
        "stream_event" => {
            let inner = &event["event"];
            let inner_type = match inner["type"].as_str() {
                Some(t) => t,
                None => return vec![],
            };

            match inner_type {
                "content_block_delta" => {
                    let delta = &inner["delta"];
                    match delta["type"].as_str() {
                        Some("text_delta") => match delta["text"].as_str() {
                            Some(t) if !t.is_empty() => vec![StreamEvent::Delta(t.to_string())],
                            _ => vec![],
                        },
                        Some("thinking_delta") => match delta["thinking"].as_str() {
                            Some(t) if !t.is_empty() => {
                                vec![StreamEvent::ReasoningDelta(t.to_string())]
                            },
                            _ => vec![],
                        },
                        _ => vec![],
                    }
                },
                _ => vec![],
            }
        },
        // Complete assistant snapshot — emit ObservedToolStart for each tool_use block.
        "assistant" => {
            let content = match event["message"]["content"].as_array() {
                Some(c) => c,
                None => return vec![],
            };
            content
                .iter()
                .filter_map(|block| {
                    if block["type"].as_str() == Some("tool_use") {
                        let id = block["id"].as_str().unwrap_or("unknown").to_string();
                        let name = block["name"].as_str().unwrap_or("unknown").to_string();
                        let arguments = block["input"].clone();
                        Some(StreamEvent::ObservedToolStart {
                            id,
                            name,
                            arguments,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        },
        // Tool result — emit ObservedToolEnd for each tool_result in the message.
        "user" => {
            let content = match event["message"]["content"].as_array() {
                Some(c) => c,
                None => return vec![],
            };
            // Top-level tool_use_result has the output; content[] has the IDs.
            let top_result = &event["tool_use_result"];
            let result_text = if let Some(s) = top_result.as_str() {
                Some(s.to_string())
            } else if !top_result.is_null() {
                Some(top_result.to_string())
            } else {
                None
            };
            // Truncate large results for UI.
            let result_text = result_text.map(|t| {
                if t.len() > 2000 {
                    format!("{}…", &t[..2000])
                } else {
                    t
                }
            });
            content
                .iter()
                .filter_map(|block| {
                    if block["type"].as_str() == Some("tool_result") {
                        let id = block["tool_use_id"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string();
                        let is_error = block["is_error"].as_bool().unwrap_or(false);
                        Some(StreamEvent::ObservedToolEnd {
                            id,
                            result: result_text.clone(),
                            is_error,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        },
        "result" => {
            let is_success = event["subtype"].as_str() == Some("success");
            if is_success {
                let usage = Usage {
                    input_tokens: event["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                    output_tokens: event["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                    ..Usage::default()
                };
                vec![StreamEvent::Done(usage)]
            } else {
                let subtype = event["subtype"].as_str().unwrap_or("unknown");
                vec![StreamEvent::Error(format!(
                    "Claude CLI session ended: {subtype}"
                ))]
            }
        },
        _ => vec![],
    }
}

fn friendly_spawn_error(e: &std::io::Error, prompt_len: usize) -> anyhow::Error {
    if e.raw_os_error() == Some(7) || e.to_string().contains("Argument list too long") {
        anyhow::anyhow!(
            "Conversation too large to send ({:.0} KB in argv). \
             Run /compact to shrink context, or /new to start fresh. \
             (OS error: {e})",
            prompt_len as f64 / 1024.0,
        )
    } else {
        anyhow::anyhow!("failed to spawn claude CLI: {e}")
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
        let (args, prompt) = self.build_session_args(messages, "json");

        debug!(
            model = %self.model,
            prompt_len = prompt.len(),
            "claude-cli complete request"
        );
        trace!(prompt = %prompt, "claude-cli prompt");

        let mut cmd = tokio::process::Command::new(&self.claude_binary);
        cmd.args(&args)
            .env_remove("CLAUDECODE")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }
        let mut child = cmd.spawn().map_err(|e| {
            self.clear_session();
            friendly_spawn_error(&e, prompt.len())
        })?;

        // Deliver the prompt via stdin to avoid Linux's 128 KB
        // per-argument limit (MAX_ARG_STRLEN).
        {
            use tokio::io::AsyncWriteExt;
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("stdin was not piped on spawned claude CLI"))?;
            stdin.write_all(prompt.as_bytes()).await.map_err(|e| {
                self.clear_session();
                anyhow::anyhow!("failed to write prompt to claude CLI stdin: {e}")
            })?;
        }

        let output = child.wait_with_output().await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code = output.status.code().unwrap_or(-1);
            self.clear_session();
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
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
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
            let (args, prompt) = self.build_session_args(&messages, "stream-json");

            debug!(
                model = %self.model,
                prompt_len = prompt.len(),
                "claude-cli stream request"
            );
            trace!(prompt = %prompt, "claude-cli stream prompt");

            let mut cmd = tokio::process::Command::new(&self.claude_binary);
            cmd.args(&args)
                .env_remove("CLAUDECODE")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            if let Some(ref dir) = self.working_dir {
                cmd.current_dir(dir);
            }
            let mut child = match cmd.spawn()
            {
                Ok(child) => child,
                Err(e) => {
                    self.clear_session();
                    yield StreamEvent::Error(friendly_spawn_error(&e, prompt.len()).to_string());
                    return;
                }
            };

            // Deliver the prompt via stdin to avoid Linux's 128 KB
            // per-argument limit (MAX_ARG_STRLEN).
            {
                use tokio::io::AsyncWriteExt;
                let mut stdin = match child.stdin.take() {
                    Some(s) => s,
                    None => {
                        self.clear_session();
                        yield StreamEvent::Error("claude CLI stdin not captured".into());
                        return;
                    }
                };
                if let Err(e) = stdin.write_all(prompt.as_bytes()).await {
                    self.clear_session();
                    yield StreamEvent::Error(format!("failed to write prompt to claude CLI stdin: {e}"));
                    return;
                }
            }

            let stdout = match child.stdout {
                Some(stdout) => stdout,
                None => {
                    self.clear_session();
                    yield StreamEvent::Error("claude CLI stdout not captured".into());
                    return;
                }
            };

            let reader = tokio::io::BufReader::new(stdout);
            use tokio::io::AsyncBufReadExt;
            let mut lines = reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                trace!(line = %line, "claude-cli stream line");

                for event in parse_stream_events(&line) {
                    match &event {
                        StreamEvent::Error(_) => {
                            self.clear_session();
                            yield event;
                            return;
                        },
                        StreamEvent::Done(_) => {
                            yield event;
                            return;
                        },
                        _ => yield event,
                    }
                }
            }

            // If we got here without a Done event, the process may have exited abnormally.
            self.clear_session();
            warn!("claude CLI stream ended without result event");
            yield StreamEvent::Done(Usage::default());
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
            ChatMessage::assistant_with_tools(Some("Let me run that.".into()), vec![
                moltis_agents::model::ToolCall {
                    id: "call_1".into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({"cmd": "ls"}),
                },
            ]),
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

    // ── parse_stream_events ─────────────────────────────────────────

    #[test]
    fn parse_text_delta() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Delta(t) if t == "hello"));
    }

    #[test]
    fn parse_thinking_delta() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me check"}}}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::ReasoningDelta(t) if t == "Let me check"));
    }

    #[test]
    fn parse_observed_tool_start_from_assistant() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls -la"}}]}}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ObservedToolStart {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "Bash");
                assert_eq!(arguments["command"], "ls -la");
            },
            other => panic!("expected ObservedToolStart, got {other:?}"),
        }
    }

    #[test]
    fn parse_multiple_tool_uses_from_assistant() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/a.rs"}},{"type":"tool_use","id":"t2","name":"Grep","input":{"pattern":"TODO"}}]}}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], StreamEvent::ObservedToolStart { name, .. } if name == "Read")
        );
        assert!(
            matches!(&events[1], StreamEvent::ObservedToolStart { name, .. } if name == "Grep")
        );
    }

    #[test]
    fn parse_observed_tool_end_from_user() {
        let line = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"file1.txt\nfile2.txt"}]},"tool_use_result":"file1.txt\nfile2.txt"}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ObservedToolEnd {
                id,
                result,
                is_error,
            } => {
                assert_eq!(id, "toolu_01");
                assert!(!is_error);
                assert!(result.as_ref().unwrap().contains("file1.txt"));
            },
            other => panic!("expected ObservedToolEnd, got {other:?}"),
        }
    }

    #[test]
    fn parse_observed_tool_end_error() {
        let line = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":"Permission denied"}]},"tool_use_result":"Permission denied"}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ObservedToolEnd { is_error, .. } => assert!(is_error),
            other => panic!("expected ObservedToolEnd, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_block_start_tool_ignored() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"Read","input":{}}}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_input_json_delta_ignored() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_assistant_text_only_ignored() {
        let line =
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hello world"}]}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_result_success() {
        let line = r#"{"type":"result","subtype":"success","total_cost_usd":0.01,"duration_ms":5000,"num_turns":3,"usage":{"input_tokens":100,"output_tokens":50}}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::Done(usage) => {
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
            },
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn parse_result_error() {
        let line = r#"{"type":"result","subtype":"error_max_budget_usd"}"#;
        let events = parse_stream_events(line);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::Error(msg) if msg.contains("error_max_budget_usd"))
        );
    }

    #[test]
    fn parse_system_event_ignored() {
        let line =
            r#"{"type":"system","subtype":"init","session_id":"abc","model":"claude-sonnet-4-6"}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_invalid_json() {
        assert!(parse_stream_events("not json at all").is_empty());
    }

    #[test]
    fn parse_empty_text_delta_ignored() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_message_start_ignored() {
        let line = r#"{"type":"stream_event","event":{"type":"message_start","message":{"model":"claude-sonnet-4-6","id":"msg_01"}}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_content_block_start_text_ignored() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_user_no_tool_result_ignored() {
        let line = r#"{"type":"user","message":{"content":[]},"tool_use_result":null}"#;
        assert!(parse_stream_events(line).is_empty());
    }

    #[test]
    fn parse_user_tool_result_truncated() {
        let long_result = "x".repeat(3000);
        let line = format!(
            r#"{{"type":"user","message":{{"content":[{{"type":"tool_result","tool_use_id":"t1"}}]}},"tool_use_result":"{long_result}"}}"#
        );
        let events = parse_stream_events(&line);
        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ObservedToolEnd { result, .. } => {
                let r = result.as_ref().unwrap();
                assert!(r.len() <= 2010);
                assert!(r.ends_with('…'));
            },
            other => panic!("expected ObservedToolEnd, got {other:?}"),
        }
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
    fn with_binary_overrides_path() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into())
            .with_binary("/usr/local/bin/claude".into());
        assert_eq!(provider.claude_binary, "/usr/local/bin/claude");
    }

    // ── new_content_for_resume / tool_results_since_last_assistant ─────

    #[test]
    fn extracts_tool_results_after_assistant() {
        let messages = vec![
            ChatMessage::user("search memory"),
            ChatMessage::assistant_with_tools(Some("Searching.".into()), vec![
                moltis_agents::model::ToolCall {
                    id: "call_1".into(),
                    name: "memory_search".into(),
                    arguments: serde_json::json!({}),
                },
            ]),
            ChatMessage::tool("call_1", "result data"),
        ];
        let prompt = tool_results_since_last_assistant(&messages);
        assert!(prompt.contains("Tool result for call_1:"));
        assert!(prompt.contains("result data"));
        assert!(prompt.ends_with("Continue."));
    }

    #[test]
    fn extracts_multiple_tool_results() {
        let messages = vec![
            ChatMessage::user("do two things"),
            ChatMessage::assistant_with_tools(None, vec![
                moltis_agents::model::ToolCall {
                    id: "call_1".into(),
                    name: "tool_a".into(),
                    arguments: serde_json::json!({}),
                },
                moltis_agents::model::ToolCall {
                    id: "call_2".into(),
                    name: "tool_b".into(),
                    arguments: serde_json::json!({}),
                },
            ]),
            ChatMessage::tool("call_1", "result A"),
            ChatMessage::tool("call_2", "result B"),
        ];
        let prompt = tool_results_since_last_assistant(&messages);
        assert!(prompt.contains("call_1"));
        assert!(prompt.contains("result A"));
        assert!(prompt.contains("call_2"));
        assert!(prompt.contains("result B"));
        // Results should be in order (call_1 before call_2).
        let pos_a = prompt.find("call_1").unwrap();
        let pos_b = prompt.find("call_2").unwrap();
        assert!(pos_a < pos_b);
    }

    #[test]
    fn returns_continue_when_no_tool_results() {
        let messages = vec![ChatMessage::assistant("done")];
        assert_eq!(tool_results_since_last_assistant(&messages), "Continue.");
    }

    #[test]
    fn resume_content_for_user_message() {
        let messages = vec![
            ChatMessage::user("first"),
            ChatMessage::assistant("reply"),
            ChatMessage::user("second"),
        ];
        assert_eq!(new_content_for_resume(&messages), "second");
    }

    #[test]
    fn resume_content_for_tool_result() {
        let messages = vec![
            ChatMessage::user("search"),
            ChatMessage::assistant_with_tools(None, vec![moltis_agents::model::ToolCall {
                id: "c1".into(),
                name: "search".into(),
                arguments: serde_json::json!({}),
            }]),
            ChatMessage::tool("c1", "found it"),
        ];
        let content = new_content_for_resume(&messages);
        assert!(content.contains("c1"));
        assert!(content.contains("found it"));
    }

    // ── build_session_args (session tracking) ─────────────────────────

    #[test]
    fn fresh_call_uses_session_id() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let messages = vec![ChatMessage::user("Hello")];
        let (args, prompt) = provider.build_session_args(&messages, "stream-json");

        assert!(args.contains(&"--session-id".into()));
        assert!(!args.contains(&"--resume".into()));
        assert!(!args.contains(&"--no-session-persistence".into()));
        assert!(args.contains(&"--print".into()));
        assert!(args.contains(&"stream-json".into()));
        assert!(args.contains(&"--include-partial-messages".into()));
        assert_eq!(prompt, "Hello");
        // Session should now be stored.
        assert!(provider.active_session.lock().unwrap().is_some());
    }

    #[test]
    fn json_format_omits_partial_messages_flag() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let messages = vec![ChatMessage::user("Hello")];
        let (args, _) = provider.build_session_args(&messages, "json");

        assert!(!args.contains(&"--include-partial-messages".into()));
    }

    #[test]
    fn continuation_uses_resume() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // First call: fresh session.
        let messages_1 = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("search memory for health"),
        ];
        let (args_1, _) = provider.build_session_args(&messages_1, "stream-json");
        assert!(args_1.contains(&"--session-id".into()));
        let session_id = provider.active_session.lock().unwrap().clone().unwrap();

        // Second call: continuation with tool results.
        let messages_2 = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("search memory for health"),
            ChatMessage::assistant_with_tools(Some("Searching.".into()), vec![
                moltis_agents::model::ToolCall {
                    id: "call_1".into(),
                    name: "memory_search".into(),
                    arguments: serde_json::json!({"query": "health"}),
                },
            ]),
            ChatMessage::tool("call_1", "Found 3 results about health."),
        ];
        let (args_2, prompt_2) = provider.build_session_args(&messages_2, "stream-json");

        assert!(args_2.contains(&"--resume".into()));
        assert!(!args_2.contains(&"--session-id".into()));
        assert!(args_2.contains(&session_id));
        // Prompt should contain only the tool result, not full history.
        assert!(prompt_2.contains("call_1"));
        assert!(prompt_2.contains("Found 3 results"));
        assert!(!prompt_2.contains("search memory for health"));
        // System prompt should NOT be included in resume args.
        assert!(!args_2.contains(&"--append-system-prompt".into()));
    }

    #[test]
    fn fresh_call_with_system_prompt() {
        let provider = ClaudeCliProvider::new("claude-opus-4-6".into())
            .with_system_prompt("Be helpful".into());
        let messages = vec![ChatMessage::user("Hello")];
        let (args, _) = provider.build_session_args(&messages, "json");

        let sys_idx = args
            .iter()
            .position(|a| a == "--append-system-prompt")
            .unwrap();
        // The system prompt is adapted for CLI — should contain the original
        // text wrapped in the Moltis addendum header.
        assert!(args[sys_idx + 1].contains("Be helpful"));
        assert!(args[sys_idx + 1].contains("Moltis platform"));
        assert!(args.contains(&"--session-id".into()));
    }

    #[test]
    fn clear_session_resets_state() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let messages = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages, "json");
        assert!(provider.active_session.lock().unwrap().is_some());

        provider.clear_session();
        assert!(provider.active_session.lock().unwrap().is_none());
    }

    #[test]
    fn no_active_session_falls_back_to_fresh() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        // Don't set up a session first — simulate edge case.
        let messages = vec![
            ChatMessage::user("search"),
            ChatMessage::assistant_with_tools(None, vec![moltis_agents::model::ToolCall {
                id: "c1".into(),
                name: "search".into(),
                arguments: serde_json::json!({}),
            }]),
            ChatMessage::tool("c1", "result"),
        ];
        let (args, _) = provider.build_session_args(&messages, "stream-json");
        // Should fall back to --session-id since there's no active session.
        assert!(args.contains(&"--session-id".into()));
        assert!(!args.contains(&"--resume".into()));
    }

    #[test]
    fn second_user_message_resumes_session() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // First call: fresh session.
        let messages_1 = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages_1, "json");
        let session_id = provider.active_session.lock().unwrap().clone().unwrap();

        // Second call: new user message in same conversation.
        let messages_2 = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("Follow-up question"),
        ];
        let (args, prompt) = provider.build_session_args(&messages_2, "json");

        // Should resume, not start fresh.
        assert!(args.contains(&"--resume".into()));
        assert!(!args.contains(&"--session-id".into()));
        assert!(args.contains(&session_id));
        // Prompt should be just the new user message.
        assert_eq!(prompt, "Follow-up question");
    }

    #[test]
    fn cleared_session_starts_fresh() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // First call: fresh session.
        let messages_1 = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages_1, "json");
        let session_1 = provider.active_session.lock().unwrap().clone().unwrap();

        // Simulate error → session cleared.
        provider.clear_session();

        // Next call starts fresh with a new session.
        let messages_2 = vec![ChatMessage::user("New topic")];
        provider.build_session_args(&messages_2, "json");
        let session_2 = provider.active_session.lock().unwrap().clone().unwrap();

        assert_ne!(session_1, session_2);
    }

    #[test]
    fn fresh_conversation_clears_stale_session() {
        // Regression: /new in Moltis should start a fresh Claude CLI session,
        // not resume the old one. A fresh conversation has no assistant messages.
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // First session: user asks something, gets a reply.
        let messages_1 = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("What is Rust?"),
        ];
        provider.build_session_args(&messages_1, "json");
        let session_1 = provider.active_session.lock().unwrap().clone().unwrap();

        // Simulate /new: fresh conversation with no assistant messages.
        let messages_new = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("What tools do you have?"),
        ];
        let (args, _) = provider.build_session_args(&messages_new, "json");

        // Should NOT resume — should start a fresh session.
        assert!(args.contains(&"--session-id".into()));
        assert!(!args.contains(&"--resume".into()));
        let session_2 = provider.active_session.lock().unwrap().clone().unwrap();
        assert_ne!(session_1, session_2);
    }

    #[test]
    fn continuation_with_assistant_still_resumes() {
        // Verify that legitimate continuations (with assistant messages) still resume.
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // First call.
        let messages_1 = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages_1, "json");
        let session_id = provider.active_session.lock().unwrap().clone().unwrap();

        // Second call with prior assistant message — should resume.
        let messages_2 = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
            ChatMessage::user("What next?"),
        ];
        let (args, _) = provider.build_session_args(&messages_2, "json");
        assert!(args.contains(&"--resume".into()));
        assert!(args.contains(&session_id));
    }

    #[test]
    fn resume_extracts_user_text_for_multimodal() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        // Set up active session.
        let messages_1 = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages_1, "json");

        // Resume with multimodal user message.
        let messages_2 = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
            ChatMessage::user_multimodal(vec![moltis_agents::model::ContentPart::Text(
                "Describe this".into(),
            )]),
        ];
        let (args, prompt) = provider.build_session_args(&messages_2, "json");

        assert!(args.contains(&"--resume".into()));
        assert_eq!(prompt, "Describe this");
    }

    // ── stdin prompt delivery (argv stays small) ──────────────────────

    #[test]
    fn prompt_not_in_argv() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let messages = vec![ChatMessage::user("Hello world")];
        let (args, prompt) = provider.build_session_args(&messages, "json");

        assert_eq!(prompt, "Hello world");
        assert!(
            !args.contains(&"Hello world".to_string()),
            "prompt must not appear in argv — it goes via stdin"
        );
    }

    #[test]
    fn large_prompt_not_in_argv() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());
        let big = "x".repeat(200_000);
        let messages = vec![ChatMessage::user(&big)];
        let (args, prompt) = provider.build_session_args(&messages, "stream-json");

        assert_eq!(prompt.len(), 200_000);
        let argv_bytes: usize = args.iter().map(|a| a.len()).sum();
        assert!(
            argv_bytes < 10_000,
            "argv should be small (flags only), got {argv_bytes} bytes"
        );
    }

    #[test]
    fn resume_prompt_not_in_argv() {
        let provider = ClaudeCliProvider::new("claude-sonnet-4-6".into());

        let messages_1 = vec![ChatMessage::user("Hello")];
        provider.build_session_args(&messages_1, "json");

        let messages_2 = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
            ChatMessage::user("Follow-up question with details"),
        ];
        let (args, prompt) = provider.build_session_args(&messages_2, "json");

        assert_eq!(prompt, "Follow-up question with details");
        assert!(
            !args.contains(&prompt),
            "resume prompt must not appear in argv"
        );
    }

    // ── friendly_spawn_error ────────────────────────────────────────────

    #[test]
    fn e2big_error_gives_friendly_message() {
        let io_err = std::io::Error::from_raw_os_error(7); // E2BIG
        let friendly = friendly_spawn_error(&io_err, 140_000);
        let msg = friendly.to_string();
        assert!(msg.contains("/compact"), "should suggest /compact: {msg}");
        assert!(msg.contains("/new"), "should suggest /new: {msg}");
        assert!(msg.contains("137"), "should show size in KB: {msg}");
    }

    #[test]
    fn non_e2big_error_uses_generic_message() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "No such file");
        let friendly = friendly_spawn_error(&io_err, 1000);
        let msg = friendly.to_string();
        assert!(
            msg.contains("failed to spawn claude CLI"),
            "should use generic prefix: {msg}"
        );
        assert!(
            !msg.contains("/compact"),
            "should not suggest /compact: {msg}"
        );
    }

    // ── adapt_system_prompt_for_cli ───────────────────────────────────

    #[test]
    fn adapt_strips_generic_intro() {
        let input =
            "You are a helpful assistant. You can use tools when needed.\n\nSome context.\n";
        let adapted = adapt_system_prompt_for_cli(input);
        assert!(!adapted.contains("You are a helpful assistant"));
        assert!(adapted.contains("Moltis platform"));
        assert!(adapted.contains("Some context."));
    }

    #[test]
    fn adapt_replaces_tools_heading() {
        let input = "## Available Tools\n\n### memory_search\nSearch memory.\n";
        let adapted = adapt_system_prompt_for_cli(input);
        assert!(adapted.contains("## Additional Moltis Tools"));
        assert!(adapted.contains("built-in Claude Code tools"));
        assert!(adapted.contains("tool_call"));
        // Tool schema still present.
        assert!(adapted.contains("### memory_search"));
    }

    #[test]
    fn adapt_removes_how_to_call_section() {
        let input = "\
## Available Tools\n\
\n\
### exec\n\
Run commands.\n\
\n\
## How to call tools\n\
\n\
When you need to use a tool, output EXACTLY this fenced block:\n\
\n\
```tool_call\n\
{\"tool\": \"exec\", \"arguments\": {\"command\": \"ls\"}}\n\
```\n\
\n\
**Rules:**\n\
- The JSON must be valid.\n\
\n\
## Guidelines\n\
\n\
Be concise.\n";
        let adapted = adapt_system_prompt_for_cli(input);
        // "How to call tools" section removed.
        assert!(!adapted.contains("## How to call tools"));
        assert!(!adapted.contains("**Rules:**"));
        // But "Guidelines" section preserved.
        assert!(adapted.contains("## Guidelines"));
        assert!(adapted.contains("Be concise."));
        // Tool schema preserved.
        assert!(adapted.contains("### exec"));
    }

    #[test]
    fn adapt_preserves_passthrough_content() {
        let input =
            "## Runtime\n\nHost: data_dir=/home/user/.moltis\n\n## Guidelines\n\nBe helpful.\n";
        let adapted = adapt_system_prompt_for_cli(input);
        assert!(adapted.contains("## Runtime"));
        assert!(adapted.contains("Host: data_dir=/home/user/.moltis"));
        assert!(adapted.contains("## Guidelines"));
    }

    /// Dump a realistic adapted system prompt to `/tmp/claude-cli-system-prompt.txt`.
    ///
    /// Run with: `cargo test -p moltis-providers --features provider-claude-cli -- dump_adapted_prompt --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_adapted_prompt() {
        let input = r#"You are a helpful assistant. You can use tools when needed.

Your name is Thomas 🤖.
Your theme: A helpful AI assistant running on a home server.

## Soul

You are a knowledgeable, thoughtful assistant. Be concise and direct.

The user's name is Michael.

## Runtime

Host: data_dir=/home/thomas/.moltis, os=linux, hostname=thomas
Sandbox(exec): enabled=false

Execution routing:
- `exec` runs inside sandbox when `Sandbox(exec): enabled=true`.
- When sandbox is disabled, `exec` runs on the host and may require approval.

## Memory

You have access to a persistent memory system. Use it to store and recall important information.

**Memory bootstrap (from MEMORY.md):**
- Michael lives in Zug, Switzerland.
- Home server hostname: thomas.

## Available Tools

### memory_search
Search semantic memory for relevant stored information.
Params: query (string, required), limit (integer)

### memory_save
Save information to persistent semantic memory.
Params: content (string, required), tags (array)

### memory_get
Retrieve a specific memory entry by ID.
Params: id (string, required)

### exec
Execute a shell command.
Params: command (string, required), timeout (integer), sandbox (boolean)

### speak
Convert text to speech and send as voice message.
Params: text (string, required), voice (string)

### web_fetch
Fetch content from a URL.
Params: url (string, required), prompt (string)

### browser
Launch a headless browser to interact with web pages.
Params: url (string, required), action (string)

### calc
Evaluate a mathematical expression.
Params: expression (string, required)

### send_message
Send a message to a channel (Telegram, etc.).
Params: text (string, required), channel (string)

## How to call tools

When you need to use a tool, output EXACTLY this fenced block:

```tool_call
{"tool": "<tool_name>", "arguments": {<arguments>}}
```

**Rules:**
- The JSON must be valid. No comments, no trailing commas.
- One tool call per fenced block. You may include multiple blocks.
- Wait for the tool result before continuing.
- You may include brief reasoning text before the block.

**Example:**
User: What files are in the current directory?
Assistant: I'll list the files for you.
```tool_call
{"tool": "exec", "arguments": {"command": "ls -la"}}
```

## Guidelines

- Start with a normal conversational response. Do not call tools for greetings, small talk, or questions you can answer directly.
- Use the calc tool for arithmetic and expressions.
- Use the exec tool for shell/system tasks.
- Before tool calls, briefly state what you are about to do.
- The UI already shows raw tool output (stdout/stderr/exit). Summarize outcomes instead.

## Silent Replies

When you have nothing meaningful to add after a tool call, return an empty response.

The current date and time is Saturday, 2026-03-08 05:30 UTC.
"#;
        let adapted = adapt_system_prompt_for_cli(input);
        std::fs::write("/tmp/claude-cli-system-prompt.txt", &adapted).unwrap();
        println!(
            "Wrote {} bytes to /tmp/claude-cli-system-prompt.txt",
            adapted.len()
        );
    }
}
