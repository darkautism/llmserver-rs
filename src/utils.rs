use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    #[default]
    LLM,
    ASR,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelConfig {
    pub model_repo: String,
    pub model_name: String,
    pub model_type: ModelType,
    pub model_path: Option<String>,
    pub tokenizer_repo: Option<String>,
    #[serde(skip_deserializing)]
    pub _asserts_path: String,
    pub cache_path: Option<String>,
    pub think: Option<bool>,
}
