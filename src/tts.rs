use actix_web::{post, web, HttpRequest, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

use crate::state::AppState;
use crate::{
    api_keys::{extract_api_key, record_token_usage, validate_api_key},
    huggingface::{require_token, HuggingFaceError},
    tokenizer::count_text_tokens,
    OpenAiError,
};
use sqlx::Row;

#[derive(Debug, Deserialize, Serialize, utoipa::ToSchema)]
pub struct TextToSpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
}

#[post("/audio/speech")]
pub async fn text_to_speech(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<TextToSpeechRequest>,
) -> impl Responder {
    let Some(api_key) = extract_api_key(&req) else {
        return HttpResponse::Unauthorized().finish();
    };

    let api_key_info = match validate_api_key(&state.pool, &api_key).await {
        Ok(Some(info)) => info,
        Ok(None) => return HttpResponse::Unauthorized().finish(),
        Err(err) => return err,
    };

    let prompt_tokens = count_text_tokens(&body.input);
    if let Some(limit) = api_key_info.token_limit {
        let current_total = api_key_info.prompt_tokens_used + api_key_info.completion_tokens_used;
        if current_total + prompt_tokens >= limit {
            return HttpResponse::TooManyRequests().json(OpenAiError {
                message: "token quota exceeded".to_owned(),
                code: "quota_exceeded".to_owned(),
                r#type: "rate_limit_exceeded".to_owned(),
                param: None,
            });
        }
    }

    let row = match sqlx::query(
        "SELECT repo_id, revision FROM models WHERE name = ?1 AND model_type = 'TTS'",
    )
    .bind(&body.model)
    .fetch_optional(&state.pool)
    .await
    {
        Ok(Some(row)) => row,
        Ok(None) => {
            return HttpResponse::BadRequest().json(OpenAiError {
                message: format!(
                    "The TTS model {} does not exist or is not configured.",
                    body.model
                ),
                code: "model_not_found".to_owned(),
                r#type: "invalid_request_error".to_owned(),
                param: None,
            })
        }
        Err(err) => return HttpResponse::InternalServerError().body(err.to_string()),
    };

    let repo_id: String = row
        .try_get("repo_id")
        .unwrap_or_else(|_| body.model.clone());
    let _revision: Option<String> = row.try_get("revision").unwrap_or(None);

    let token = match require_token(&state, "default").await {
        Ok(token) => Some(token),
        Err(HuggingFaceError::MissingToken) => None,
        Err(e) => return HttpResponse::InternalServerError().body(e.to_string()),
    };

    let url = format!("https://api-inference.huggingface.co/models/{}", repo_id);

    let mut request_builder = state.http_client.post(url);
    if let Some(token) = token {
        request_builder = request_builder.bearer_auth(token);
    }

    let mut payload = serde_json::json!({
        "inputs": body.input,
    });
    if body.voice.is_some() || body.format.is_some() {
        let mut parameters = serde_json::Map::new();
        if let Some(voice) = &body.voice {
            parameters.insert(
                "voice".to_string(),
                serde_json::Value::String(voice.clone()),
            );
        }
        if let Some(format) = &body.format {
            parameters.insert(
                "format".to_string(),
                serde_json::Value::String(format.clone()),
            );
        }
        payload.as_object_mut().unwrap().insert(
            "parameters".to_string(),
            serde_json::Value::Object(parameters),
        );
    }

    let response = match request_builder.json(&payload).send().await {
        Ok(response) => response,
        Err(err) => return HttpResponse::BadGateway().body(err.to_string()),
    };

    if !response.status().is_success() {
        let status = response.status();
        let message = response
            .text()
            .await
            .unwrap_or_else(|_| "failed to decode error".to_owned());
        return HttpResponse::build(status).body(message);
    }

    let audio_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => return HttpResponse::BadGateway().body(err.to_string()),
    };

    if let Err(err) =
        record_token_usage(&state.pool, &api_key_info.id, &body.model, prompt_tokens, 0).await
    {
        return err;
    }

    let content_type = body
        .format
        .as_deref()
        .map(|fmt| format!("audio/{fmt}"))
        .unwrap_or_else(|| "audio/wav".to_string());

    HttpResponse::Ok()
        .content_type(content_type)
        .body(audio_bytes)
}
