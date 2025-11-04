use crate::{Content, Message};

const AVG_CHARS_PER_TOKEN: f32 = 4.0;

pub fn estimate_text_tokens(text: &str) -> i64 {
    let char_count = text.chars().count() as f32;
    ((char_count / AVG_CHARS_PER_TOKEN).ceil().max(1.0)) as i64
}

pub fn estimate_messages_tokens(messages: &[Message]) -> i64 {
    let mut total_chars = 0usize;
    for message in messages {
        if let Some(content) = &message.content {
            match content {
                Content::String(text) => total_chars += text.chars().count(),
                Content::Array(parts) => {
                    total_chars += parts.iter().map(|p| p.chars().count()).sum::<usize>();
                }
            }
        }
    }
    ((total_chars as f32 / AVG_CHARS_PER_TOKEN).ceil().max(1.0)) as i64
}
