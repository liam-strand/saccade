//! Blocking HTTP client for OpenAI-compatible chat endpoints (Ollama, OpenRouter, etc.).

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Errors that can arise when sending a request to the inference server.
#[derive(Debug)]
pub enum LlmError {
    /// The HTTP request failed (connection refused, timeout, non-2xx status, etc.).
    Http(String),
    /// Serializing the request body to JSON failed.
    Json(serde_json::Error),
    /// The response body arrived but could not be parsed as an expected chat response.
    BadResponse(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Http(e) => write!(f, "HTTP error: {e}"),
            LlmError::Json(e) => write!(f, "JSON error: {e}"),
            LlmError::BadResponse(s) => write!(f, "bad LLM response: {s}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// A single message in a chat conversation.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Conversation role: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

/// Wire type for the `POST /v1/chat/completions` request body.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    response_format: ResponseFormat<'a>,
}

#[derive(Serialize)]
struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchemaBlock<'a>,
}

#[derive(Serialize)]
struct JsonSchemaBlock<'a> {
    name: &'a str,
    strict: bool,
    schema: &'a serde_json::Value,
}

/// Top-level wrapper in the `/v1/chat/completions` response body.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

/// Blocking HTTP client for any OpenAI-compatible inference server.
///
/// Works with Ollama (`base_url = "http://host:11434"`) and OpenRouter
/// (`base_url = "https://openrouter.ai/api"`, `api_key = Some("sk-or-...")`).
#[derive(Clone)]
pub struct LlmClient {
    /// Server root URL, trailing slash stripped.
    base_url: String,
    /// Model name passed to the server with every request.
    model: String,
    /// Bearer token sent as `Authorization: Bearer {key}` when present.
    api_key: Option<String>,
    /// Underlying HTTP agent with pre-configured connect and response timeouts.
    agent: ureq::Agent,
}

impl LlmClient {
    /// Creates a client targeting `base_url` with a 10 s connect timeout and 120 s response timeout.
    ///
    /// Pass `api_key = Some("sk-or-...")` for services that require Bearer authentication
    /// (e.g. OpenRouter). Pass `None` for unauthenticated servers (e.g. local Ollama).
    pub fn new(base_url: &str, model: &str, api_key: Option<&str>) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_recv_response(Some(Duration::from_secs(600)))
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.map(str::to_string),
            agent: config.into(),
        }
    }

    /// Returns the model name this client was configured with.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Blocking chat round-trip to `POST /v1/chat/completions` with structured output enforcement.
    ///
    /// `schema_name` and `schema` are forwarded as `response_format.json_schema` with `strict: true`
    /// so the server guarantees the reply matches the schema.
    /// `call_type` is logged with the latency for downstream analysis.
    /// `latency_override_ms`, when `Some`, replaces the measured wall-clock latency in the log
    /// so simulations can inject a pre-profiled distribution instead of live server timing.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        schema_name: &str,
        schema: &serde_json::Value,
        call_type: &str,
        latency_override_ms: Option<u64>,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::to_vec(&ChatRequest {
            model: &self.model,
            messages,
            stream: false,
            response_format: ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchemaBlock {
                    name: schema_name,
                    strict: true,
                    schema,
                },
            },
        })
        .map_err(LlmError::Json)?;

        let t0 = std::time::Instant::now();

        let mut req = self
            .agent
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(key) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }

        let mut response = req
            .send(body)
            .map_err(|e: ureq::Error| LlmError::Http(e.to_string()))?;

        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|e: ureq::Error| LlmError::Http(e.to_string()))?;

        let actual_ms = t0.elapsed().as_millis() as u64;
        let effective_ms = latency_override_ms.unwrap_or(actual_ms);
        tracing::info!(
            latency_ms = effective_ms,
            model = ?self.model,
            call_type = ?call_type,
            "llm_call"
        );

        serde_json::from_str::<ChatResponse>(&text)
            .ok()
            .and_then(|r| r.choices.into_iter().next())
            .map(|c| c.message.content)
            .ok_or(LlmError::BadResponse(text))
    }
}
