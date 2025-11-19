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

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
#[serde(untagged)]
pub enum Content {
    String(String),
    Array(Vec<String>),
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
