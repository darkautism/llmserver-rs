pub mod asr;
pub mod audio;
pub mod chat;
pub mod llm;
pub mod ollama;
pub mod openai;
pub mod utils;

use std::{io::Read, pin::Pin};

use actix::{Actor, Handler};
use hf_hub::api::Progress;
pub use rkllm_rs::prelude::RkllmCallbackHandler;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAiError {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: String,
}

pub trait AIModel {
    type Config: DeserializeOwned;
    fn init_with_progress<P: Progress + ModelProgress + Clone>(
        config: &Self::Config,
        p: Option<P>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized;

    fn init(config: &Self::Config) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized,
    {
        Self::init_with_progress(config, None::<()>)
    }
}

#[derive(Debug, Clone, utoipa::ToSchema)]
pub enum Content {
    Parts(Vec<ContentPart>),
    String(String),
    Array(Vec<String>),
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(Content::String(s)),
            serde_json::Value::Array(arr) => {
                if arr.is_empty() {
                    Ok(Content::Array(vec![]))
                } else if arr[0].is_string() {
                    let strings: Vec<String> = serde_json::from_value(serde_json::Value::Array(arr))
                        .map_err(serde::de::Error::custom)?;
                    Ok(Content::Array(strings))
                } else {
                    let parts: Vec<ContentPart> = serde_json::from_value(serde_json::Value::Array(arr))
                        .map_err(serde::de::Error::custom)?;
                    Ok(Content::Parts(parts))
                }
            }
            _ => Err(serde::de::Error::custom("expected string or array")),
        }
    }
}

impl Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Content::String(s) => serializer.serialize_str(s),
            Content::Array(arr) => arr.serialize(serializer),
            Content::Parts(parts) => parts.serialize(serializer),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ContentPart {
    pub r#type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default, deserialize_with = "deserialize_null_image_url")]
    pub image_url: Option<ImageUrl>,
}

fn deserialize_null_image_url<'de, D>(deserializer: D) -> Result<Option<ImageUrl>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<serde_json::Value>::deserialize(deserializer)?;
    match opt {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => {
            let url: ImageUrl = serde_json::from_value(v).map_err(serde::de::Error::custom)?;
            Ok(Some(url))
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ImageUrl {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, utoipa::ToSchema)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "developer")]
    Developer,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Message {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Role)]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Content)]
    pub content: Option<Content>,
}

#[derive(actix::Message)]
#[rtype(result = "Result<Pin<Box<dyn futures::Stream<Item = String> + Send + 'static>>, ()>")]
pub struct ProcessMessages {
    pub messages: Vec<Message>,
}

#[derive(actix::Message)]
#[rtype(result = "Result<Pin<Box<dyn futures::Stream<Item = AsrText> + Send + 'static>>, ()>")]
pub enum ProcessAudio {
    FilePath(String),
    Buffer(Box<dyn Read + Send>),
}

pub enum AsrText {
    SenseVoice(sensevoice_rs::VoiceText),
}

#[derive(actix::Message)]
#[rtype(result = "Result<(), ()>")]
pub struct ShutdownMessages;

pub trait ASR: Actor + Handler<ProcessAudio> + Handler<ShutdownMessages> + AIModel {}
pub trait LLM: Actor + Handler<ProcessMessages> + Handler<ShutdownMessages> + AIModel {}

pub trait ModelProgress {
    fn model_load(&mut self, size: usize, filename: &str, start: std::time::Instant);
    fn model_finished(&mut self);
}

impl ModelProgress for () {
    fn model_load(&mut self, _size: usize, _filename: &str, _start: std::time::Instant) {}
    fn model_finished(&mut self) {}
}
