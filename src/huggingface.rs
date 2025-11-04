use actix_session::Session;
use actix_web::{web, HttpResponse, Responder};
use derive_more::Display;
use serde::{Deserialize, Serialize};

use crate::{auth::require_admin, encryption::EncryptionError, state::AppState};
use sqlx::Row;

#[derive(Debug, Display)]
pub enum HuggingFaceError {
    #[display("token not configured")]
    MissingToken,
    #[display("encryption error: {0}")]
    Encryption(String),
    #[display("database error: {0}")]
    Database(String),
}

impl std::error::Error for HuggingFaceError {}

impl From<EncryptionError> for HuggingFaceError {
    fn from(value: EncryptionError) -> Self {
        HuggingFaceError::Encryption(value.to_string())
    }
}

#[derive(Debug, Deserialize)]
pub struct SetTokenRequest {
    #[serde(default = "default_token_name")]
    pub name: String,
    pub token: String,
}

#[derive(Debug, Serialize)]
pub struct TokenMetadata {
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
}

fn default_token_name() -> String {
    "default".to_string()
}

pub async fn upsert_token(
    state: web::Data<AppState>,
    session: Session,
    payload: web::Json<SetTokenRequest>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let request = payload.into_inner();

    let encrypted = state
        .encryption
        .encrypt(request.token.as_bytes())
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    sqlx::query(
        "INSERT INTO huggingface_tokens (name, encrypted_token) VALUES (?1, ?2)
         ON CONFLICT(name) DO UPDATE SET encrypted_token = excluded.encrypted_token, updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')"
    )
    .bind(&request.name)
    .bind(encrypted)
    .execute(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    Ok(HttpResponse::NoContent().finish())
}

#[derive(Debug, Deserialize)]
pub struct TokenNameRequest {
    #[serde(default = "default_token_name")]
    pub name: String,
}

pub async fn delete_token(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<String>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let name = path.into_inner();
    sqlx::query("DELETE FROM huggingface_tokens WHERE name = ?1")
        .bind(name)
        .execute(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::NoContent().finish())
}

pub async fn list_tokens(
    state: web::Data<AppState>,
    session: Session,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let rows =
        sqlx::query("SELECT name, created_at, updated_at FROM huggingface_tokens ORDER BY name")
            .fetch_all(&state.pool)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let metadata: Vec<TokenMetadata> = rows
        .into_iter()
        .map(|row| TokenMetadata {
            name: row.try_get::<String, _>("name").unwrap_or_default(),
            created_at: row.try_get::<String, _>("created_at").unwrap_or_default(),
            updated_at: row.try_get::<String, _>("updated_at").unwrap_or_default(),
        })
        .collect();

    Ok(HttpResponse::Ok().json(metadata))
}

pub async fn get_token(state: &AppState, name: &str) -> Result<Option<String>, HuggingFaceError> {
    let record = sqlx::query("SELECT encrypted_token FROM huggingface_tokens WHERE name = ?1")
        .bind(name)
        .fetch_optional(&state.pool)
        .await
        .map_err(|e| HuggingFaceError::Database(e.to_string()))?;

    let Some(record) = record else {
        return Ok(None);
    };

    let decrypted = state
        .encryption
        .decrypt(
            record
                .try_get("encrypted_token")
                .map_err(|e| HuggingFaceError::Database(e.to_string()))?,
        )
        .map_err(HuggingFaceError::from)?;

    let token =
        String::from_utf8(decrypted).map_err(|e| HuggingFaceError::Encryption(e.to_string()))?;

    Ok(Some(token))
}

pub async fn require_token(state: &AppState, name: &str) -> Result<String, HuggingFaceError> {
    match get_token(state, name).await? {
        Some(token) => Ok(token),
        None => Err(HuggingFaceError::MissingToken),
    }
}
