use crate::llm::client::ChatMessage;

/// Fluent builder for constructing a sequence of chat messages.
pub struct PromptBuilder {
    messages: Vec<ChatMessage>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "system".into(),
            content: content.into(),
        });
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "user".into(),
            content: content.into(),
        });
        self
    }

    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "assistant".into(),
            content: content.into(),
        });
        self
    }

    pub fn build(&self) -> &[ChatMessage] {
        &self.messages
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}
