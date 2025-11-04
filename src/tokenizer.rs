use once_cell::sync::Lazy;
use tiktoken_rs::CoreBPE;

use crate::{Content, Message};

static TOKENIZER: Lazy<CoreBPE> =
    Lazy::new(|| tiktoken_rs::cl100k_base().expect("failed to initialize tokenizer"));

pub fn count_text_tokens(text: &str) -> i64 {
    TOKENIZER.encode_with_special_tokens(text).len() as i64
}

pub fn count_messages_tokens(messages: &[Message]) -> i64 {
    messages
        .iter()
        .map(|message| {
            message
                .content
                .as_ref()
                .map(|content| match content {
                    Content::String(text) => count_text_tokens(text),
                    Content::Array(items) => items.iter().map(|s| count_text_tokens(s)).sum(),
                })
                .unwrap_or(0)
        })
        .sum()
}
