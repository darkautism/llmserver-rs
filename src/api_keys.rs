use std::time::SystemTime;

use actix_session::Session;
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::{auth::require_admin, state::AppState};

#[derive(Debug, Serialize)]
pub struct ApiKeyDto {
    pub id: String,
    pub label: String,
    pub last_four: String,
    pub is_active: bool,
    pub token_limit: Option<i64>,
    pub prompt_tokens_used: i64,
    pub completion_tokens_used: i64,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

pub fn extract_api_key(req: &HttpRequest) -> Option<String> {
    if let Some(header) = req.headers().get("Authorization") {
        if let Ok(value) = header.to_str() {
            if let Some(token) = value.strip_prefix("Bearer ") {
                return Some(token.trim().to_owned());
            }
        }
    }
    if let Some(header) = req.headers().get("x-api-key") {
        if let Ok(value) = header.to_str() {
            return Some(value.trim().to_owned());
        }
    }
    None
}

fn hash_secret(secret: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut rand::thread_rng());
    Argon2::default()
        .hash_password(secret.as_bytes(), &salt)
        .map(|hash| hash.to_string())
}

pub async fn create_api_key(
    state: web::Data<AppState>,
    session: Session,
    payload: web::Json<CreateApiKeyRequest>,
) -> Result<impl Responder, actix_web::Error> {
    let admin_id = require_admin(&session)?;

    let secret: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(48)
        .map(char::from)
        .collect();

    let key_id = Uuid::new_v4().to_string();
    let raw_key = format!("rk-{key_id}-{secret}");
    let last_four = raw_key
        .chars()
        .rev()
        .take(4)
        .collect::<String>()
        .chars()
        .rev()
        .collect();

    let hashed_secret = hash_secret(&secret)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    sqlx::query(
        "INSERT INTO api_keys (id, label, hashed_key, last_four, user_id, token_limit) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
    )
    .bind(&key_id)
    .bind(&request.label)
    .bind(&hashed_secret)
    .bind(&last_four)
    .bind(admin_id)
    .bind(request.token_limit)
    .execute(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let created_at_row = sqlx::query("SELECT created_at FROM api_keys WHERE id = ?1")
        .bind(&key_id)
        .fetch_one(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    let created_at: String = created_at_row
        .try_get("created_at")
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let response = CreatedApiKeyResponse {
        id: key_id,
        api_key: raw_key,
        created_at,
    };

    Ok(HttpResponse::Created().json(response))
}

#[derive(Debug, Deserialize)]
pub struct CreateApiKeyRequest {
    pub label: String,
    pub token_limit: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CreatedApiKeyResponse {
    pub id: String,
    pub api_key: String,
    pub created_at: String,
}

pub async fn list_api_keys(
    state: web::Data<AppState>,
    session: Session,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;

    let rows = sqlx::query(
        "SELECT id, label, last_four, is_active, token_limit, prompt_tokens_used, completion_tokens_used, created_at, last_used_at FROM api_keys"
    )
    .fetch_all(&state.pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let token_limit: Option<i64> = row.try_get::<Option<i64>, _>("token_limit").unwrap_or(None);
        results.push(ApiKeyDto {
            id: row.try_get::<String, _>("id").unwrap_or_default(),
            label: row.try_get::<String, _>("label").unwrap_or_default(),
            last_four: row.try_get::<String, _>("last_four").unwrap_or_default(),
            is_active: row.try_get::<i64, _>("is_active").unwrap_or(1) != 0,
            token_limit,
            prompt_tokens_used: row.try_get::<i64, _>("prompt_tokens_used").unwrap_or(0),
            completion_tokens_used: row.try_get::<i64, _>("completion_tokens_used").unwrap_or(0),
            created_at: row.try_get::<String, _>("created_at").unwrap_or_default(),
            last_used_at: row
                .try_get::<Option<String>, _>("last_used_at")
                .unwrap_or(None),
        });
    }

    Ok(HttpResponse::Ok().json(results))
}

#[derive(Debug, Deserialize)]
pub struct UpdateApiKeyRequest {
    pub label: Option<String>,
    pub is_active: Option<bool>,
    pub token_limit: Option<Option<i64>>,
}

pub async fn update_api_key(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<String>,
    payload: web::Json<UpdateApiKeyRequest>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;

    let key_id = path.into_inner();
    let mut updated = false;

    if let Some(ref new_label) = payload.label {
        sqlx::query("UPDATE api_keys SET label = ?1 WHERE id = ?2")
            .bind(new_label)
            .bind(&key_id)
            .execute(&state.pool)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
        updated = true;
    }

    if let Some(flag) = payload.is_active {
        sqlx::query("UPDATE api_keys SET is_active = ?1 WHERE id = ?2")
            .bind(if flag { 1 } else { 0 })
            .bind(&key_id)
            .execute(&state.pool)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
        updated = true;
    }

    if let Some(limit_option) = &payload.token_limit {
        let limit_value: Option<i64> = *limit_option;
        sqlx::query("UPDATE api_keys SET token_limit = ?1 WHERE id = ?2")
            .bind(limit_value)
            .bind(&key_id)
            .execute(&state.pool)
            .await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
        updated = true;
    }

    if updated {
        sqlx::query(
            "UPDATE api_keys SET updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?1",
        )
        .bind(&key_id)
        .execute(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    }

    Ok(HttpResponse::NoContent().finish())
}

pub async fn delete_api_key(
    state: web::Data<AppState>,
    session: Session,
    path: web::Path<String>,
) -> Result<impl Responder, actix_web::Error> {
    require_admin(&session)?;
    let key_id = path.into_inner();
    sqlx::query("DELETE FROM api_keys WHERE id = ?1")
        .bind(key_id)
        .execute(&state.pool)
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    Ok(HttpResponse::NoContent().finish())
}

pub struct ApiKeyValidation {
    pub id: String,
    pub label: String,
    pub token_limit: Option<i64>,
    pub prompt_tokens_used: i64,
    pub completion_tokens_used: i64,
}

pub async fn validate_api_key(
    pool: &SqlitePool,
    key: &str,
) -> Result<Option<ApiKeyValidation>, actix_web::Error> {
    let parts: Vec<&str> = key.splitn(3, '-').collect();
    if parts.len() != 3 {
        return Ok(None);
    }
    let id = parts[1];
    let secret = parts[2];

    let record = sqlx::query(
        "SELECT id, label, hashed_key, token_limit, is_active, prompt_tokens_used, completion_tokens_used FROM api_keys WHERE id = ?1"
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let Some(record) = record else {
        return Ok(None);
    };

    let is_active: i64 = record.try_get("is_active").unwrap_or(0);
    if is_active == 0 {
        return Ok(None);
    }

    let hashed_key: String = record
        .try_get("hashed_key")
        .map_err(|_| actix_web::error::ErrorInternalServerError("invalid hash"))?;
    let hash = PasswordHash::new(&hashed_key)
        .map_err(|_| actix_web::error::ErrorInternalServerError("invalid hash"))?;
    if Argon2::default()
        .verify_password(secret.as_bytes(), &hash)
        .is_err()
    {
        return Ok(None);
    }

    Ok(Some(ApiKeyValidation {
        id: record.try_get("id").unwrap_or_else(|_| id.to_string()),
        label: record.try_get("label").unwrap_or_default(),
        token_limit: record
            .try_get::<Option<i64>, _>("token_limit")
            .unwrap_or(None),
        prompt_tokens_used: record.try_get("prompt_tokens_used").unwrap_or(0),
        completion_tokens_used: record.try_get("completion_tokens_used").unwrap_or(0),
    }))
}

pub async fn record_token_usage(
    pool: &SqlitePool,
    api_key_id: &str,
    model_name: &str,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> Result<(), actix_web::Error> {
    let total_tokens = prompt_tokens + completion_tokens;

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    sqlx::query(
        "INSERT INTO token_usage_events (api_key_id, model_name, prompt_tokens, completion_tokens, total_tokens) VALUES (?1, ?2, ?3, ?4, ?5)"
    )
    .bind(api_key_id)
    .bind(model_name)
    .bind(prompt_tokens)
    .bind(completion_tokens)
    .bind(total_tokens)
    .execute(&mut *tx)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    sqlx::query(
        "UPDATE api_keys SET prompt_tokens_used = prompt_tokens_used + ?1, completion_tokens_used = completion_tokens_used + ?2, last_used_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?3"
    )
    .bind(prompt_tokens)
    .bind(completion_tokens)
    .bind(api_key_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    tx.commit()
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))
}
