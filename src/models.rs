use std::{fs, path::PathBuf, str::FromStr};

use actix_session::Session;
use actix_web::{web, HttpResponse, Responder};
use derive_more::Display;
use futures::future::join_all;
use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Row};

use crate::{
    auth::require_admin,
    huggingface::{require_token, HuggingFaceError},
    state::{AppState, LlmHandle},
    utils::{ModelConfig, ModelType},
    ProcessMessages, ShutdownMessages,
};

#[derive(Debug, Display)]
pub enum ModelError {
    #[display("database error: {0}")]
    Database(String),
    #[display("model not found")]
    NotFound,
    #[display("invalid model type")]
    InvalidType,
    #[display("i/o error: {0}")]
    Io(String),
    #[display("serde error: {0}")]
    Serde(String),
    #[display("huggingface error: {0}")]
    HuggingFace(String),
    #[display("model runtime error: {0}")]
    Runtime(String),
}

impl std::error::Error for ModelError {}

impl From<sqlx::Error> for ModelError {
    fn from(value: sqlx::Error) -> Self {
        ModelError::Database(value.to_string())
    }
}

impl From<std::io::Error> for ModelError {
    fn from(value: std::io::Error) -> Self {
        ModelError::Io(value.to_string())
    }
}

impl From<serde_json::Error> for ModelError {
    fn from(value: serde_json::Error) -> Self {
        ModelError::Serde(value.to_string())
    }
}

impl From<HuggingFaceError> for ModelError {
    fn from(value: HuggingFaceError) -> Self {
        ModelError::HuggingFace(value.to_string())
    }
}

#[derive(Debug, FromRow)]
struct ModelRow {
    pub id: i64,
    pub name: String,
    pub repo_id: String,
    pub revision: Option<String>,
    pub model_type: String,
    pub config_path: Option<String>,
    pub local_path: Option<String>,
    pub status: String,
    pub last_started_at: Option<String>,
    pub last_stopped_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct ModelDto {
    pub id: i64,
    pub name: String,
    pub repo_id: String,
    pub revision: Option<String>,
    pub model_type: ModelType,
    pub config_path: Option<String>,
    pub local_path: Option<String>,
    pub status: String,
    pub last_started_at: Option<String>,
    pub last_stopped_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl TryFrom<ModelRow> for ModelDto {
    type Error = ModelError;

    fn try_from(value: ModelRow) -> Result<Self, Self::Error> {
        let model_type =
            ModelType::from_str(&value.model_type).map_err(|_| ModelError::InvalidType)?;
        Ok(ModelDto {
            id: value.id,
            name: value.name,
            repo_id: value.repo_id,
            revision: value.revision,
            model_type,
            config_path: value.config_path,
            local_path: value.local_path,
            status: value.status,
            last_started_at: value.last_started_at,
            last_stopped_at: value.last_stopped_at,
            created_at: value.created_at,
            updated_at: value.updated_at,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    pub repo_id: String,
    pub revision: Option<String>,
    pub model_type: ModelType,
    #[serde(default)]
    pub auto_download: bool,
    pub files: Option<Vec<String>>,
    pub think: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct DownloadModelRequest {
    pub files: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct StartModelRequest {
    #[serde(default = "default_instances")]
    pub instances: usize,
}

fn default_instances() -> usize {
    1
}

pub async fn list_models(
    state: web::Data<AppState>,
    session: Session,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let rows = sqlx::query_as::<_, ModelRow>(
        "SELECT id, name, repo_id, revision, model_type, config_path, local_path, status, last_started_at, last_stopped_at, created_at, updated_at FROM models ORDER BY created_at DESC"
    )
    .fetch_all(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let models: Result<Vec<ModelDto>, ModelError> =
        rows.into_iter().map(ModelDto::try_from).collect();
    let models = models.map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::Ok().json(models))
}

fn sanitize_model_name(name: &str) -> Result<(), ModelError> {
    let valid = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'));
    if valid {
        Ok(())
    } else {
        Err(ModelError::Runtime(
            "model name may only contain letters, numbers, '-' and '_'".to_string(),
        ))
    }
}

pub async fn create_model(
    state: web::Data<AppState>,
    session: Session,
    payload: web::Json<CreateModelRequest>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let request = payload.into_inner();
    sanitize_model_name(&request.name)
        .map_err(|e| actix_web::error::ErrorBadRequest(e.to_string()))?;

    let existing = sqlx::query("SELECT id FROM models WHERE name = ?1")
        .bind(&request.name)
        .fetch_optional(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    if existing.is_some() {
        return Ok(HttpResponse::Conflict().body("model name already exists"));
    }

    let mut config = ModelConfig {
        model_repo: request.repo_id.clone(),
        model_name: request.name.clone(),
        model_type: request.model_type.clone(),
        model_path: None,
        _asserts_path: String::new(),
        cache_path: Some(state.huggingface_cache.to_string_lossy().to_string()),
        think: request.think,
    };

    let mut config_path = PathBuf::from("assets/config");
    fs::create_dir_all(&config_path)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    config_path.push(format!("{}.json", request.name));
    config._asserts_path = config_path.to_string_lossy().to_string();
    let serialized = serde_json::to_string_pretty(&config)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    fs::write(&config_path, serialized)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let result = sqlx::query(
        "INSERT INTO models (name, repo_id, revision, model_type, config_path, status) VALUES (?1, ?2, ?3, ?4, ?5, 'stopped')"
    )
    .bind(&request.name)
    .bind(&request.repo_id)
    .bind(&request.revision)
    .bind(request.model_type.to_string())
    .bind(config_path.to_string_lossy().to_string())
    .execute(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    let model_id = result.last_insert_rowid();

    let mut dto = fetch_model_dto(&state, model_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    if request.auto_download {
        dto = download_model_internal(&state, dto.id, request.files)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    }

    Ok(HttpResponse::Created().json(dto))
}

async fn fetch_model_row(state: &AppState, id: i64) -> Result<ModelRow, ModelError> {
    sqlx::query_as::<_, ModelRow>(
        "SELECT id, name, repo_id, revision, model_type, config_path, local_path, status, last_started_at, last_stopped_at, created_at, updated_at FROM models WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(&state.pool)
    .await?
    .ok_or(ModelError::NotFound)
}

async fn fetch_model_dto(state: &AppState, id: i64) -> Result<ModelDto, ModelError> {
    let row = fetch_model_row(state, id).await?;
    row.try_into()
}

async fn download_model_internal(
    state: &AppState,
    model_id: i64,
    files: Option<Vec<String>>,
) -> Result<ModelDto, ModelError> {
    let row = fetch_model_row(state, model_id).await?;
    let revision = row.revision.clone().unwrap_or_else(|| "main".to_string());

    let token = match require_token(state, "default").await {
        Ok(token) => Some(token),
        Err(HuggingFaceError::MissingToken) => None,
        Err(e) => return Err(e.into()),
    };

    let mut api_builder = ApiBuilder::new()
        .with_progress(false)
        .with_cache_dir(state.huggingface_cache.clone());
    if let Some(token) = token {
        api_builder = api_builder.with_token(Some(token));
    }
    let api = api_builder
        .build()
        .map_err(|e| ModelError::HuggingFace(e.to_string()))?;

    let repo = Repo::with_revision(row.repo_id.clone(), RepoType::Model, revision.clone());
    let repo_handle = api.repo(repo.clone());
    let info = repo_handle
        .info()
        .await
        .map_err(|e| ModelError::HuggingFace(e.to_string()))?;

    let files_to_download = if let Some(specified) = files {
        specified
    } else {
        info.siblings.into_iter().map(|s| s.rfilename).collect()
    };

    let mut snapshot_root: Option<PathBuf> = None;
    for filename in files_to_download {
        let path = repo_handle
            .download(&filename)
            .await
            .map_err(|e| ModelError::HuggingFace(e.to_string()))?;
        if snapshot_root.is_none() {
            snapshot_root = path.parent().map(|p| p.to_path_buf());
        }
    }

    if let Some(root) = snapshot_root {
        sqlx::query(
            "UPDATE models SET local_path = ?1, status = 'downloaded', updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?2"
        )
        .bind(root.to_string_lossy().to_string())
        .bind(row.id)
        .execute(&state.pool)
        .await?;
    }

    fetch_model_dto(state, row.id).await
}

pub async fn download_model(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<i64>,
    payload: Option<web::Json<DownloadModelRequest>>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let model_id = path.into_inner();
    let files = payload.map(|p| p.into_inner().files).flatten();
    let dto = download_model_internal(&state, model_id, files)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::Ok().json(dto))
}

pub async fn delete_model(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<i64>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let model_id = path.into_inner();
    sqlx::query("DELETE FROM models WHERE id = ?1")
        .bind(model_id)
        .execute(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::NoContent().finish())
}

fn load_model_config(path: &str) -> Result<ModelConfig, ModelError> {
    let data = fs::read_to_string(path)?;
    let mut config: ModelConfig = serde_json::from_str(&data)?;
    config._asserts_path = path.to_string();
    Ok(config)
}

async fn stop_llm_instances(state: &AppState, name: &str) {
    let handles = state.model_manager.remove_llm(name).await;
    let futures = handles
        .into_iter()
        .map(|handle| handle.shutdown.send(ShutdownMessages));
    join_all(futures).await;
}

pub async fn start_model(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<i64>,
    payload: Option<web::Json<StartModelRequest>>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let model_id = path.into_inner();
    let request = payload
        .map(|p| p.into_inner())
        .unwrap_or(StartModelRequest { instances: 1 });

    let dto = start_model_internal(&state, model_id, request.instances)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::Ok().json(dto))
}

pub(crate) async fn start_model_internal(
    state: &AppState,
    model_id: i64,
    instances: usize,
) -> Result<ModelDto, ModelError> {
    let row = fetch_model_row(state, model_id).await?;
    let model_type = ModelType::from_str(&row.model_type).map_err(|_| ModelError::InvalidType)?;
    match model_type {
        ModelType::LLM => {
            let config_path = row
                .config_path
                .clone()
                .ok_or_else(|| ModelError::Runtime("model is missing config path".into()))?;
            let config = load_model_config(&config_path)?;

            stop_llm_instances(state, &row.name).await;

            let total_instances = if instances == 0 { 1 } else { instances };
            let mut handles = Vec::new();
            for _ in 0..total_instances {
                let actor = crate::llm::simple::SimpleRkLLM::init(&config)
                    .map_err(|e| ModelError::Runtime(e.to_string()))?;
                let addr = actor.start();
                handles.push(LlmHandle {
                    processor: addr.clone().recipient::<ProcessMessages>(),
                    shutdown: addr.recipient::<ShutdownMessages>(),
                });
            }
            state.model_manager.register_llm(&row.name, handles).await;

            sqlx::query(
                "UPDATE models SET status = 'running', last_started_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?1"
            )
            .bind(row.id)
            .execute(&state.pool)
            .await?;
        }
        ModelType::TTS => {
            sqlx::query(
                "UPDATE models SET status = 'running', last_started_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?1"
            )
            .bind(row.id)
            .execute(&state.pool)
            .await?;
        }
        ModelType::ASR => {
            return Err(ModelError::Runtime(
                "ASR model lifecycle management is not yet supported in the admin API".to_string(),
            ));
        }
    }

    fetch_model_dto(state, row.id).await
}

pub async fn stop_model(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<i64>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let model_id = path.into_inner();
    let row = fetch_model_row(&state, model_id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    let model_type = ModelType::from_str(&row.model_type)
        .map_err(|_| actix_web::error::ErrorInternalServerError("invalid model type"))?;
    match model_type {
        ModelType::LLM => {
            stop_llm_instances(&state, &row.name).await;
        }
        ModelType::TTS => {}
        ModelType::ASR => {
            return Ok(HttpResponse::BadRequest()
                .body("ASR model lifecycle management is not yet supported in the admin API"));
        }
    }

    sqlx::query(
        "UPDATE models SET status = 'stopped', last_stopped_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?1"
    )
    .bind(row.id)
    .execute(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let dto = fetch_model_dto(&state, row.id)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::Ok().json(dto))
}

pub async fn get_model(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<i64>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let dto = fetch_model_dto(&state, path.into_inner())
        .await
        .map_err(|e| match e {
            ModelError::NotFound => actix_web::error::ErrorNotFound("model not found"),
            other => actix_web::error::ErrorInternalServerError(other.to_string()),
        })?;
    Ok(HttpResponse::Ok().json(dto))
}

pub async fn resume_running_models(state: &AppState) -> Result<(), ModelError> {
    let rows = sqlx::query("SELECT id FROM models WHERE status = 'running'")
        .fetch_all(&state.pool)
        .await?;

    for row in rows {
        let id: i64 = row.try_get("id").unwrap_or_default();
        let _ = start_model_internal(state, id, 1).await?;
    }

    Ok(())
}
