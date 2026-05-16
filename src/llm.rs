//! HTTP client and prompt-building utilities for LLM-backed scheduler policies.
//!
//! Schedulers that need language-model guidance construct a [`crate::llm::prompt::PromptBuilder`] message
//! sequence and send it to an Ollama-compatible inference server via [`crate::llm::client::LlmClient`].

pub mod client;
pub mod prompt;

pub use client::{ChatMessage, LlmClient, LlmError};
pub use prompt::PromptBuilder;
