//! Blocking HTTP client for OpenAI-compatible chat endpoints (Ollama, OpenRouter, etc.).

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Total attempts (initial + retries) for a single LLM round-trip before giving up.
const MAX_ATTEMPTS: u32 = 4;
/// Base backoff before the first retry; doubles each subsequent attempt.
const RETRY_BACKOFF_BASE: Duration = Duration::from_secs(2);
/// Upper bound on a single backoff sleep (before jitter).
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(30);

/// Returns `true` when a `ureq` transport error is worth retrying: timeouts, transient
/// connection/IO failures, HTTP 429 (rate limit), and 5xx server errors. Deterministic
/// failures (4xx other than 429, malformed URI, DNS, …) return `false`.
fn is_retryable(err: &ureq::Error) -> bool {
    match err {
        ureq::Error::Timeout(_) | ureq::Error::Io(_) | ureq::Error::ConnectionFailed => true,
        ureq::Error::StatusCode(code) => *code == 429 || *code >= 500,
        _ => false,
    }
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat<'a>>,
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
        let body = serde_json::to_vec(&ChatRequest {
            model: &self.model,
            messages,
            stream: false,
            response_format: Some(ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchemaBlock {
                    name: schema_name,
                    strict: true,
                    schema,
                },
            }),
        })
        .map_err(LlmError::Json)?;

        self.post_and_log(body, call_type, latency_override_ms)
    }

    /// Blocking free-form chat round-trip to `POST /v1/chat/completions` with no response schema.
    ///
    /// Unlike [`chat`](Self::chat), this method omits `response_format` entirely, allowing the
    /// model to reason freely in prose before a structured call is made.
    /// `call_type` is logged with the latency for downstream analysis.
    /// `latency_override_ms`, when `Some`, replaces the measured wall-clock latency in the log.
    pub fn chat_freeform(
        &self,
        messages: &[ChatMessage],
        call_type: &str,
        latency_override_ms: Option<u64>,
    ) -> Result<String, LlmError> {
        let body = serde_json::to_vec(&ChatRequest {
            model: &self.model,
            messages,
            stream: false,
            response_format: None,
        })
        .map_err(LlmError::Json)?;

        self.post_and_log(body, call_type, latency_override_ms)
    }

    /// Send the request body once and return the raw response text, surfacing the underlying
    /// `ureq::Error` so the caller can decide whether the failure is retryable.
    fn send_once(&self, url: &str, body: &[u8]) -> Result<String, ureq::Error> {
        let mut req = self
            .agent
            .post(url)
            .header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = req.send(body)?;
        response.body_mut().read_to_string()
    }

    /// Internal helper: POST a pre-serialized body, log latency, extract the reply text.
    ///
    /// Transient transport failures (timeouts, connection drops, 429, 5xx) are retried with
    /// exponential backoff and jitter up to [`MAX_ATTEMPTS`] times; deterministic failures and a
    /// final exhausted retry surface as [`LlmError::Http`]. Retrying here, below `chat`/
    /// `chat_freeform`, keeps the parse-level retry loop in `llm_common` unaware of transport.
    fn post_and_log(
        &self,
        body: Vec<u8>,
        call_type: &str,
        latency_override_ms: Option<u64>,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        for attempt in 1..=MAX_ATTEMPTS {
            let t0 = std::time::Instant::now();
            match self.send_once(&url, &body) {
                Ok(text) => {
                    let actual_ms = t0.elapsed().as_millis() as u64;
                    let effective_ms = latency_override_ms.unwrap_or(actual_ms);
                    tracing::info!(
                        latency_ms = effective_ms,
                        model = ?self.model,
                        call_type = ?call_type,
                        "llm_call"
                    );

                    return serde_json::from_str::<ChatResponse>(&text)
                        .ok()
                        .and_then(|r| r.choices.into_iter().next())
                        .map(|c| c.message.content)
                        .ok_or(LlmError::BadResponse(text));
                }
                Err(e) => {
                    if attempt < MAX_ATTEMPTS && is_retryable(&e) {
                        let backoff = retry_backoff(attempt);
                        tracing::warn!(
                            attempt,
                            max_attempts = MAX_ATTEMPTS,
                            call_type,
                            backoff_ms = backoff.as_millis() as u64,
                            "LLM call failed, retrying: {e}"
                        );
                        std::thread::sleep(backoff);
                        continue;
                    }
                    return Err(LlmError::Http(e.to_string()));
                }
            }
        }
        unreachable!("loop returns on the final attempt")
    }
}

/// Exponential backoff for retry `attempt` (1-based): `base * 2^(attempt-1)`, capped, plus up to
/// 25% random jitter to avoid synchronized retries across concurrent batch combos.
fn retry_backoff(attempt: u32) -> Duration {
    let scaled = RETRY_BACKOFF_BASE
        .saturating_mul(1u32 << (attempt - 1))
        .min(RETRY_BACKOFF_CAP);
    let jitter = rand::rng().random_range(0.0..0.25) * scaled.as_secs_f64();
    scaled + Duration::from_secs_f64(jitter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_msg() -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }
    }

    #[test]
    fn is_retryable_classifies_status_codes() {
        assert!(is_retryable(&ureq::Error::StatusCode(429)));
        assert!(is_retryable(&ureq::Error::StatusCode(500)));
        assert!(is_retryable(&ureq::Error::StatusCode(503)));
        assert!(!is_retryable(&ureq::Error::StatusCode(404)));
        assert!(!is_retryable(&ureq::Error::StatusCode(400)));
        assert!(!is_retryable(&ureq::Error::HostNotFound));
    }

    #[test]
    fn retry_backoff_grows_and_caps() {
        // Lower bound (no jitter) doubles per attempt and never exceeds cap + 25% jitter.
        assert!(retry_backoff(1) >= RETRY_BACKOFF_BASE);
        assert!(retry_backoff(2) >= RETRY_BACKOFF_BASE * 2);
        let max = RETRY_BACKOFF_CAP.as_secs_f64() * 1.25;
        for attempt in 1..=8 {
            assert!(retry_backoff(attempt).as_secs_f64() <= max);
        }
    }

    #[test]
    fn chat_request_with_schema_serializes_response_format() {
        let schema = serde_json::json!({"type": "object"});
        let req = ChatRequest {
            model: "test-model",
            messages: &[dummy_msg()],
            stream: false,
            response_format: Some(ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchemaBlock {
                    name: "schedule",
                    strict: true,
                    schema: &schema,
                },
            }),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            json.contains("response_format"),
            "structured call must include response_format"
        );
        assert!(json.contains("json_schema"));
    }

    #[test]
    fn chat_request_without_schema_omits_response_format() {
        let req = ChatRequest {
            model: "test-model",
            messages: &[dummy_msg()],
            stream: false,
            response_format: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("response_format"),
            "freeform call must NOT include response_format; got: {json}"
        );
    }
}
