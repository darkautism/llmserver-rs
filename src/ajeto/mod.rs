use std::fmt;
use std::time::{Duration, SystemTime};

use actix::Recipient;
use actix_web::rt::time::timeout;
use futures::StreamExt;
use rand::seq::IndexedRandom;
use serde::de::DeserializeOwned;

use crate::{Content, Message, ProcessMessages, Role};

const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(Clone)]
pub struct AjetoEngine {
    llm_pool: Vec<Recipient<ProcessMessages>>,
    timeout: Duration,
}

impl AjetoEngine {
    pub fn new(llm_pool: Vec<Recipient<ProcessMessages>>, timeout: Duration) -> Self {
        Self { llm_pool, timeout }
    }

    pub fn with_default_timeout(llm_pool: Vec<Recipient<ProcessMessages>>) -> Self {
        Self::new(llm_pool, Duration::from_secs(DEFAULT_TIMEOUT_SECS))
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub async fn invoke<T>(
        &self,
        agent_name: &str,
        messages: Vec<Message>,
    ) -> Result<AjetoInvocation<T>, AjetoError>
    where
        T: DeserializeOwned + Clone + Send + 'static,
    {
        let mut rng = rand::rng();
        let Some(recipient) = self.llm_pool.choose(&mut rng) else {
            return Err(AjetoError::new("Model pool is empty"));
        };

        let send_future = recipient.send(ProcessMessages { messages });
        match timeout(self.timeout, send_future).await {
            Ok(Ok(Ok(stream))) => {
                let chunks = stream.collect::<Vec<_>>().await;
                let raw = chunks.join("");
                let parsed = parse_agent_json::<T>(&raw);
                Ok(AjetoInvocation { raw, parsed })
            }
            Ok(Ok(Err(_))) => Err(AjetoError::new(format!(
                "{} agent returned an empty stream",
                agent_name
            ))),
            Ok(Err(e)) => Err(AjetoError::new(format!(
                "{} agent mailbox error: {}",
                agent_name, e
            ))),
            Err(_) => Err(AjetoError::new(format!(
                "{} agent timed out after {} seconds",
                agent_name,
                self.timeout.as_secs()
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AjetoInvocation<T> {
    pub raw: String,
    pub parsed: Option<T>,
}

#[derive(Debug)]
pub struct AjetoError {
    message: String,
}

impl AjetoError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for AjetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AjetoError {}

pub fn build_messages(
    system_prompt: Option<&str>,
    developer_prompt: Option<&str>,
    user_prompt: &str,
) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(Message {
            role: Some(Role::System),
            content: Some(Content::String(system.to_string())),
        });
    }
    if let Some(developer) = developer_prompt {
        messages.push(Message {
            role: Some(Role::Developer),
            content: Some(Content::String(developer.to_string())),
        });
    }
    messages.push(Message {
        role: Some(Role::User),
        content: Some(Content::String(user_prompt.to_string())),
    });
    messages
}

pub fn parse_agent_json<T: DeserializeOwned>(raw: &str) -> Option<T> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let candidate = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        Some(trimmed.to_string())
    } else if let Some(start) = trimmed.find('{') {
        let end = trimmed.rfind('}')?;
        Some(trimmed[start..=end].to_string())
    } else if let Some(start) = trimmed.find('[') {
        let end = trimmed.rfind(']')?;
        Some(trimmed[start..=end].to_string())
    } else {
        None
    }?;

    let sanitized = strip_json_code_fence(&candidate);
    serde_json::from_str::<T>(&sanitized).ok()
}

pub fn strip_json_code_fence(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.starts_with("```") {
        let mut lines = trimmed.lines();
        lines.next();
        let mut collected = Vec::new();
        for line in lines {
            if line.trim_start().starts_with("```") {
                break;
            }
            collected.push(line);
        }
        collected.join("\n")
    } else {
        trimmed.to_string()
    }
}

pub fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}
