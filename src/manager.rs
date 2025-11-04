use std::{collections::HashMap, sync::Arc};

use actix::Recipient;
use actix_web::rt;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    asr::simple::{SimpleASR, SimpleASRConfig},
    llm::simple::SimpleRkLLM,
    tts::simple::{SimpleToneTts, SimpleTtsConfig},
    utils::{ModelConfig, ModelType},
    ProcessAudio, ProcessMessages, ProcessTts, ShutdownMessages,
};

#[derive(Clone)]
pub struct ModelManager {
    inner: Arc<RwLock<ModelRegistry>>,
}

struct ModelRegistry {
    configs: HashMap<String, ModelConfig>,
    llm: HashMap<String, Vec<ModelInstance>>, // model_name -> instances
    asr: HashMap<String, Vec<ModelInstance>>,
    tts: HashMap<String, Vec<ModelInstance>>,
    lookup: HashMap<Uuid, (ModelType, String)>,
}

#[derive(Clone)]
pub struct ModelInstance {
    pub id: Uuid,
    pub repo_id: String,
    pub model_name: String,
    pub model_type: ModelType,
    pub created_at: DateTime<Utc>,
    recipient: InstanceRecipient,
    shutdown: Recipient<ShutdownMessages>,
}

impl ModelInstance {
    pub fn llm_recipient(&self) -> Option<Recipient<ProcessMessages>> {
        match &self.recipient {
            InstanceRecipient::Llm(r) => Some(r.clone()),
            _ => None,
        }
    }

    pub fn asr_recipient(&self) -> Option<Recipient<ProcessAudio>> {
        match &self.recipient {
            InstanceRecipient::Asr(r) => Some(r.clone()),
            _ => None,
        }
    }

    pub fn tts_recipient(&self) -> Option<Recipient<ProcessTts>> {
        match &self.recipient {
            InstanceRecipient::Tts(r) => Some(r.clone()),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum InstanceRecipient {
    Llm(Recipient<ProcessMessages>),
    Asr(Recipient<ProcessAudio>),
    Tts(Recipient<ProcessTts>),
}

#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("unknown model repo: {0}")]
    UnknownModel(String),
    #[error("actor init error: {0}")]
    Actor(String),
    #[error("model instance not found")]
    InstanceNotFound,
}

impl ModelManager {
    pub fn new(configs: HashMap<String, ModelConfig>) -> Self {
        let registry = ModelRegistry {
            configs,
            llm: HashMap::new(),
            asr: HashMap::new(),
            tts: HashMap::new(),
            lookup: HashMap::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(registry)),
        }
    }

    pub async fn start_instances(
        &self,
        repo_id: &str,
        instances: usize,
    ) -> Result<Vec<ModelInstance>, ModelManagerError> {
        let config = {
            let registry = self.inner.read().await;
            registry
                .configs
                .get(repo_id)
                .cloned()
                .ok_or_else(|| ModelManagerError::UnknownModel(repo_id.to_string()))?
        };

        let mut started = Vec::new();
        for _ in 0..instances.max(1) {
            let model_instance = Self::spawn_instance(&config)?;
            let mut registry = self.inner.write().await;
            registry.lookup.insert(
                model_instance.id,
                (config.model_type.clone(), config.model_name.clone()),
            );
            match config.model_type {
                ModelType::LLM => registry
                    .llm
                    .entry(config.model_name.clone())
                    .or_default()
                    .push(model_instance.clone()),
                ModelType::ASR => registry
                    .asr
                    .entry(config.model_name.clone())
                    .or_default()
                    .push(model_instance.clone()),
                ModelType::TTS => registry
                    .tts
                    .entry(config.model_name.clone())
                    .or_default()
                    .push(model_instance.clone()),
            }
            started.push(model_instance);
        }
        Ok(started)
    }

    fn spawn_instance(config: &ModelConfig) -> Result<ModelInstance, ModelManagerError> {
        let created_at = Utc::now();
        match config.model_type {
            ModelType::LLM => {
                let llm = SimpleRkLLM::init(config)
                    .map_err(|e| ModelManagerError::Actor(e.to_string()))?;
                let addr = llm.start();
                let instance = ModelInstance {
                    id: Uuid::new_v4(),
                    repo_id: config.model_repo.clone(),
                    model_name: config.model_name.clone(),
                    model_type: ModelType::LLM,
                    created_at,
                    recipient: InstanceRecipient::Llm(addr.clone().recipient()),
                    shutdown: addr.recipient(),
                };
                Ok(instance)
            }
            ModelType::ASR => {
                let asr_config = SimpleASRConfig {
                    model_repo: config.model_repo.clone(),
                    model_name: config.model_name.clone(),
                };
                let asr = SimpleASR::init(&asr_config)
                    .map_err(|e| ModelManagerError::Actor(e.to_string()))?;
                let addr = asr.start();
                let instance = ModelInstance {
                    id: Uuid::new_v4(),
                    repo_id: config.model_repo.clone(),
                    model_name: config.model_name.clone(),
                    model_type: ModelType::ASR,
                    created_at,
                    recipient: InstanceRecipient::Asr(addr.clone().recipient()),
                    shutdown: addr.recipient(),
                };
                Ok(instance)
            }
            ModelType::TTS => {
                let tts_config = SimpleTtsConfig {
                    model_name: config.model_name.clone(),
                };
                let tts = SimpleToneTts::init(&tts_config)
                    .map_err(|e| ModelManagerError::Actor(e.to_string()))?;
                let addr = tts.start();
                let instance = ModelInstance {
                    id: Uuid::new_v4(),
                    repo_id: config.model_repo.clone(),
                    model_name: config.model_name.clone(),
                    model_type: ModelType::TTS,
                    created_at,
                    recipient: InstanceRecipient::Tts(addr.clone().recipient()),
                    shutdown: addr.recipient(),
                };
                Ok(instance)
            }
        }
    }

    pub async fn stop_instance(&self, instance_id: Uuid) -> Result<(), ModelManagerError> {
        let (model_type, model_name) = {
            let mut registry = self.inner.write().await;
            let Some((model_type, model_name)) = registry.lookup.remove(&instance_id) else {
                return Err(ModelManagerError::InstanceNotFound);
            };
            let list = match model_type {
                ModelType::LLM => registry.llm.get_mut(&model_name),
                ModelType::ASR => registry.asr.get_mut(&model_name),
                ModelType::TTS => registry.tts.get_mut(&model_name),
            };
            if let Some(instances) = list {
                if let Some(pos) = instances.iter().position(|i| i.id == instance_id) {
                    let instance = instances.remove(pos);
                    let shutdown = instance.shutdown.clone();
                    rt::spawn(async move {
                        let _ = shutdown.send(ShutdownMessages).await;
                    });
                }
            }
            (model_type, model_name)
        };
        // remove empty groups
        let mut registry = self.inner.write().await;
        let empty = match model_type {
            ModelType::LLM => registry
                .llm
                .get(&model_name)
                .map(|v| v.is_empty())
                .unwrap_or(false),
            ModelType::ASR => registry
                .asr
                .get(&model_name)
                .map(|v| v.is_empty())
                .unwrap_or(false),
            ModelType::TTS => registry
                .tts
                .get(&model_name)
                .map(|v| v.is_empty())
                .unwrap_or(false),
        };
        if empty {
            match model_type {
                ModelType::LLM => {
                    registry.llm.remove(&model_name);
                }
                ModelType::ASR => {
                    registry.asr.remove(&model_name);
                }
                ModelType::TTS => {
                    registry.tts.remove(&model_name);
                }
            }
        }
        Ok(())
    }

    pub async fn llm_pool(&self) -> HashMap<String, Vec<Recipient<ProcessMessages>>> {
        let registry = self.inner.read().await;
        registry
            .llm
            .iter()
            .map(|(name, instances)| {
                (
                    name.clone(),
                    instances.iter().filter_map(|i| i.llm_recipient()).collect(),
                )
            })
            .collect()
    }

    pub async fn asr_pool(&self) -> HashMap<String, Vec<Recipient<ProcessAudio>>> {
        let registry = self.inner.read().await;
        registry
            .asr
            .iter()
            .map(|(name, instances)| {
                (
                    name.clone(),
                    instances.iter().filter_map(|i| i.asr_recipient()).collect(),
                )
            })
            .collect()
    }

    pub async fn tts_pool(&self) -> HashMap<String, Vec<Recipient<ProcessTts>>> {
        let registry = self.inner.read().await;
        registry
            .tts
            .iter()
            .map(|(name, instances)| {
                (
                    name.clone(),
                    instances.iter().filter_map(|i| i.tts_recipient()).collect(),
                )
            })
            .collect()
    }

    pub async fn list_instances(&self) -> Vec<ModelInstance> {
        let registry = self.inner.read().await;
        registry
            .llm
            .values()
            .chain(registry.asr.values())
            .chain(registry.tts.values())
            .flat_map(|instances| instances.iter().cloned())
            .collect()
    }

    pub async fn configs(&self) -> HashMap<String, ModelConfig> {
        let registry = self.inner.read().await;
        registry.configs.clone()
    }

    pub async fn shutdown_all(&self) {
        let instances = self.list_instances().await;
        for instance in instances {
            let shutdown = instance.shutdown.clone();
            rt::spawn(async move {
                let _ = shutdown.send(ShutdownMessages).await;
            });
        }
        let mut registry = self.inner.write().await;
        registry.llm.clear();
        registry.asr.clear();
        registry.tts.clear();
        registry.lookup.clear();
    }
}
