//! Fluent builder for assembling ordered [`ChatMessage`] sequences.

use crate::llm::client::ChatMessage;

/// Fluent builder for constructing a sequence of chat messages.
pub struct PromptBuilder {
    /// Accumulated messages in the order they were appended.
    messages: Vec<ChatMessage>,
}

impl PromptBuilder {
    /// Creates an empty builder with no messages.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Appends a `system`-role message.
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "system".into(),
            content: content.into(),
        });
        self
    }

    /// Appends a `user`-role message.
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "user".into(),
            content: content.into(),
        });
        self
    }

    /// Appends an `assistant`-role message, useful for providing few-shot examples.
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage {
            role: "assistant".into(),
            content: content.into(),
        });
        self
    }

    /// Returns a slice of all accumulated messages in insertion order.
    pub fn build(&self) -> &[ChatMessage] {
        &self.messages
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}
