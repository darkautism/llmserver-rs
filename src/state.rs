use std::{collections::HashMap, path::PathBuf, sync::Arc};

use actix::Recipient;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use tokio::sync::RwLock;

use crate::{encryption::EncryptionService, ProcessAudio, ProcessMessages, ShutdownMessages};

use sqlx::SqlitePool;

#[derive(Clone)]
pub struct LlmHandle {
    pub processor: Recipient<ProcessMessages>,
    pub shutdown: Recipient<ShutdownMessages>,
}

#[derive(Clone)]
pub struct AppState {
    pub pool: SqlitePool,
    pub model_manager: ModelManager,
    pub encryption: Arc<EncryptionService>,
    pub http_client: reqwest::Client,
    pub huggingface_cache: PathBuf,
}

#[derive(Clone)]
pub struct AsrHandle {
    pub processor: Recipient<ProcessAudio>,
    pub shutdown: Recipient<ShutdownMessages>,
}

#[derive(Clone, Default)]
pub struct ModelManager {
    llm_handles: Arc<RwLock<HashMap<String, Vec<LlmHandle>>>>,
    asr_handles: Arc<RwLock<HashMap<String, Vec<AsrHandle>>>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register_llm(&self, model_name: &str, handles: Vec<LlmHandle>) {
        let mut guard = self.llm_handles.write().await;
        guard.insert(model_name.to_owned(), handles);
    }

    pub async fn add_llm_instances(&self, model_name: &str, mut handles: Vec<LlmHandle>) {
        let mut guard = self.llm_handles.write().await;
        guard
            .entry(model_name.to_owned())
            .and_modify(|existing| existing.append(&mut handles))
            .or_insert(handles);
    }

    pub async fn register_asr(&self, model_name: &str, handles: Vec<AsrHandle>) {
        let mut guard = self.asr_handles.write().await;
        guard.insert(model_name.to_owned(), handles);
    }

    pub async fn add_asr_instances(&self, model_name: &str, mut handles: Vec<AsrHandle>) {
        let mut guard = self.asr_handles.write().await;
        guard
            .entry(model_name.to_owned())
            .and_modify(|existing| existing.append(&mut handles))
            .or_insert(handles);
    }

    pub async fn remove_llm(&self, model_name: &str) -> Vec<LlmHandle> {
        let mut guard = self.llm_handles.write().await;
        guard.remove(model_name).unwrap_or_default()
    }

    pub async fn remove_asr(&self, model_name: &str) -> Vec<AsrHandle> {
        let mut guard = self.asr_handles.write().await;
        guard.remove(model_name).unwrap_or_default()
    }

    pub async fn choose_llm(&self, model_name: &str) -> Option<Recipient<ProcessMessages>> {
        let guard = self.llm_handles.read().await;
        guard.get(model_name).and_then(|handles| {
            let mut rng = StdRng::from_entropy();
            handles.choose(&mut rng).map(|h| h.processor.clone())
        })
    }

    pub async fn choose_asr(&self, model_name: &str) -> Option<Recipient<ProcessAudio>> {
        let guard = self.asr_handles.read().await;
        guard.get(model_name).and_then(|handles| {
            let mut rng = StdRng::from_entropy();
            handles.choose(&mut rng).map(|h| h.processor.clone())
        })
    }

    pub async fn list_llm_models(&self) -> Vec<String> {
        self.llm_handles.read().await.keys().cloned().collect()
    }

    pub async fn list_asr_models(&self) -> Vec<String> {
        self.asr_handles.read().await.keys().cloned().collect()
    }
}
