use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum LlmError {
    Http(String),
    Json(serde_json::Error),
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

#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

pub struct LlmClient {
    base_url: String,
    model: String,
}

impl LlmClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Blocking chat round-trip. Returns the assistant's reply text.
    pub fn chat(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = serde_json::to_vec(&ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        })
        .map_err(LlmError::Json)?;

        let mut response = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(body)
            .map_err(|e: ureq::Error| LlmError::Http(e.to_string()))?;

        let text = response
            .body_mut()
            .read_to_string()
            .map_err(|e: ureq::Error| LlmError::Http(e.to_string()))?;

        let parsed: ChatResponse = serde_json::from_str(&text).map_err(LlmError::Json)?;

        Ok(parsed.message.content)
    }
}
