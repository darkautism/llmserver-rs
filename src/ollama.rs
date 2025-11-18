use std::collections::HashMap;

use actix::Recipient;
use actix_web::{
    get, post,
    web::{self, Json},
    HttpResponse, Responder,
};
use serde::{Deserialize, Serialize};

use crate::{ProcessMessages, utils::ModelConfig};

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Version {
    pub version: String,
}

#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = Version, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[get("/version")]
pub async fn version() -> impl Responder {
    HttpResponse::Ok().json(Version {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Status {
    pub status: String,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema, Default)]
#[schema(
    example = json!({
        "model": "DeepSeek-R1-Distill-Qwen-1.5B",
        "insecure": false,
    })
)]
#[derive(Debug, Clone)]
pub struct PullPushRequest {
    pub model: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default = "default_true")]
    pub stream: bool,
}

pub fn default_true() -> bool {
    true
}

#[utoipa::path(
    request_body = PullPushRequest,
    responses(
        (status = OK, description = "Success", body = Status, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[post("/push")]
pub async fn push(_body: Json<PullPushRequest>) -> impl Responder {
    HttpResponse::Ok().json(Status {
        status: "not implemented".to_string(),
    })
}

#[utoipa::path(
    request_body = PullPushRequest,
    responses(
        (status = OK, description = "Success", body = Status, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[post("/pull")]
pub async fn pull(_body: Json<PullPushRequest>) -> impl Responder {
    HttpResponse::Ok().json(Status {
        status: "not implemented".to_string(),
    })
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ModelDetail {
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
struct OllamaModel {
    pub name: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub digest: String,
    pub details: Option<ModelDetail>,
}

#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = OllamaModel, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[get("/tags")]
pub async fn tags(all_configs: web::Data<HashMap<String, ModelConfig>>) -> impl Responder {
    HttpResponse::Ok().json(
        all_configs
            .keys()
            .map(|config| OllamaModel {
                name: config.clone(),
                modified_at: "".to_string(),
                size: "".to_string(),
                digest: "".to_string(),
                details: None,
            })
            .collect::<Vec<OllamaModel>>(),
    )
}

#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = OllamaModel, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[get("/ps")]
pub async fn ps(
    llm_pool: web::Data<HashMap<String, Vec<Recipient<ProcessMessages>>>>,
) -> impl Responder {
    HttpResponse::Ok().json(
        llm_pool
            .keys()
            .map(|config| OllamaModel {
                name: config.clone(),
                modified_at: "".to_string(),
                size: "".to_string(),
                digest: "".to_string(),
                details: None,
            })
            .collect::<Vec<OllamaModel>>(),
    )
}
