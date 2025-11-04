use actix_web::{
    delete, get, post,
    web::{self, Json, Path},
    HttpRequest, HttpResponse, Responder,
    HttpResponse, Responder,
};
use log::{error, warn};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::db::{PgmlRepository, RegisterModelPayload, StoredModel};
use subtle::ConstantTimeEq;

const DASHBOARD_HTML: &str = include_str!("../assets/admin-dashboard.html");

pub const ADMIN_API_TOKEN_ENV: &str = "ADMIN_API_TOKEN";

#[derive(Clone, Default)]
pub struct AdminAuthConfig {
    token: Option<Vec<u8>>,
}

impl AdminAuthConfig {
    pub fn from_env() -> Self {
        match std::env::var(ADMIN_API_TOKEN_ENV) {
            Ok(raw) if !raw.trim().is_empty() => Self {
                token: Some(raw.into_bytes()),
            },
            _ => Self { token: None },
        }
    }

    fn unauthorized_response() -> HttpResponse {
        HttpResponse::Unauthorized()
            .insert_header(("WWW-Authenticate", "Bearer"))
            .json(AdminErrorResponse::new(
                "Admin authentication failed. Supply a valid bearer token.",
            ))
    }

    pub fn is_enabled(&self) -> bool {
        self.token.is_some()
    }

    pub fn verify_token(&self, provided: &str) -> bool {
        let Some(expected) = &self.token else {
            return true;
        };

        let provided = provided.trim().as_bytes();
        provided.ct_eq(expected).into()
    }

    pub fn require_request(&self, req: &HttpRequest) -> Result<(), HttpResponse> {
        if self.token.is_none() {
            return Ok(());
        }

        if let Some(header) = req.headers().get(actix_web::http::header::AUTHORIZATION) {
            if let Ok(value) = header.to_str() {
                if let Some(token) = value.strip_prefix("Bearer ") {
                    if self.verify_token(token) {
                        return Ok(());
                    }
                }
            }
        }

        if let Some(header) = req.headers().get("X-Admin-Token") {
            if let Ok(token) = header.to_str() {
                if self.verify_token(token) {
                    return Ok(());
                }
            }
        }

        Err(Self::unauthorized_response())
    }
}


const DASHBOARD_HTML: &str = include_str!("../assets/admin-dashboard.html");

#[derive(Debug, Serialize, ToSchema)]
pub struct RagModelListResponse {
    pub models: Vec<StoredModel>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AdminErrorResponse {
    pub message: String,
}

impl AdminErrorResponse {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct RagModelRequest {
    pub pipeline_name: String,
    pub model_uri: String,
    pub task: String,
    pub collection_name: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct AdminSessionRequest {
    pub token: String,
}

#[utoipa::path(
    post,
    path = "/admin/api/session",
    request_body = AdminSessionRequest,
    responses(
        (status = OK, description = "Admin session established"),
        (status = UNAUTHORIZED, description = "Invalid admin token", body = AdminErrorResponse)
    ),
    tag = "Admin",
)]
#[post("/session")]
pub async fn create_session(
    auth: web::Data<AdminAuthConfig>,
    payload: Json<AdminSessionRequest>,
) -> impl Responder {
    if auth.verify_token(&payload.token) {
        HttpResponse::Ok().finish()
    } else {
        AdminAuthConfig::unauthorized_response()
    }
}

#[utoipa::path(
    get,
    path = "/admin/api/models",
    responses(
        (status = UNAUTHORIZED, description = "Missing or invalid admin token", body = AdminErrorResponse),
        (status = OK, description = "List all registered PGML pipelines", body = RagModelListResponse),
        (status = SERVICE_UNAVAILABLE, description = "PGML repository is disabled", body = AdminErrorResponse)
    ),
    tag = "Admin"
)]
#[get("/models")]
pub async fn list_models(
    req: HttpRequest,
    repo: web::Data<Option<PgmlRepository>>,
    auth: web::Data<AdminAuthConfig>,
) -> impl Responder {
    if let Err(resp) = auth.require_request(&req) {
        return resp;
    }

pub async fn list_models(repo: web::Data<Option<PgmlRepository>>) -> impl Responder {
    let Some(repository) = repo.get_ref() else {
        return HttpResponse::ServiceUnavailable().json(AdminErrorResponse::new(
            "PGML repository is not configured. Set DATABASE_URL to enable the Admin dashboard.",
        ));
    };

    match repository.list_models().await {
        Ok(models) => HttpResponse::Ok().json(RagModelListResponse { models }),
        Err(err) => {
            error!("Failed to list PGML models: {}", err);
            HttpResponse::InternalServerError().json(AdminErrorResponse::new(
                "Unable to query PGML model registry",
            ))
        }
    }
}

#[utoipa::path(
    post,
    path = "/admin/api/models",
    request_body = RagModelRequest,
    responses(
        (status = UNAUTHORIZED, description = "Missing or invalid admin token", body = AdminErrorResponse),
        (status = OK, description = "Model registered or updated", body = StoredModel),
        (status = SERVICE_UNAVAILABLE, description = "PGML repository is disabled", body = AdminErrorResponse),
        (status = BAD_REQUEST, description = "Invalid payload", body = AdminErrorResponse)
    ),
    tag = "Admin"
)]
#[post("/models")]
pub async fn register_model(
    req: HttpRequest,
    repo: web::Data<Option<PgmlRepository>>,
    auth: web::Data<AdminAuthConfig>,
    payload: Json<RagModelRequest>,
) -> impl Responder {
    if let Err(resp) = auth.require_request(&req) {
        return resp;
    }

    repo: web::Data<Option<PgmlRepository>>,
    payload: Json<RagModelRequest>,
) -> impl Responder {
    let Some(repository) = repo.get_ref() else {
        return HttpResponse::ServiceUnavailable().json(AdminErrorResponse::new(
            "PGML repository is not configured. Set DATABASE_URL to enable the Admin dashboard.",
        ));
    };

    if payload.pipeline_name.trim().is_empty()
        || payload.model_uri.trim().is_empty()
        || payload.task.trim().is_empty()
        || payload.collection_name.trim().is_empty()
    {
        return HttpResponse::BadRequest().json(AdminErrorResponse::new(
            "All fields are required to register a PGML pipeline",
        ));
    }

    match repository
        .register_model(RegisterModelPayload {
            pipeline_name: &payload.pipeline_name,
            model_uri: &payload.model_uri,
            task: &payload.task,
            collection_name: &payload.collection_name,
            metadata: payload.metadata.clone(),
        })
        .await
    {
        Ok(model) => HttpResponse::Ok().json(model),
        Err(err) => {
            error!(
                "Failed to register PGML model {}: {}",
                payload.pipeline_name, err
            );
            HttpResponse::InternalServerError()
                .json(AdminErrorResponse::new("Unable to register PGML model"))
        }
    }
}

#[utoipa::path(
    delete,
    path = "/admin/api/models/{pipeline_name}",
    params(("pipeline_name" = String, Path, description = "Pipeline identifier to drop")),
    responses(
        (status = UNAUTHORIZED, description = "Missing or invalid admin token", body = AdminErrorResponse),
        (status = NO_CONTENT, description = "Pipeline deleted"),
        (status = SERVICE_UNAVAILABLE, description = "PGML repository is disabled", body = AdminErrorResponse)
    ),
    tag = "Admin"
)]
#[delete("/models/{pipeline_name}")]
pub async fn delete_model(
    req: HttpRequest,
    repo: web::Data<Option<PgmlRepository>>,
    auth: web::Data<AdminAuthConfig>,
    pipeline_name: Path<String>,
) -> impl Responder {
    if let Err(resp) = auth.require_request(&req) {
        return resp;
    }

    repo: web::Data<Option<PgmlRepository>>,
    pipeline_name: Path<String>,
) -> impl Responder {
    let Some(repository) = repo.get_ref() else {
        return HttpResponse::ServiceUnavailable().json(AdminErrorResponse::new(
            "PGML repository is not configured. Set DATABASE_URL to enable the Admin dashboard.",
        ));
    };

    let pipeline_name = pipeline_name.into_inner();

    match repository.delete_model(&pipeline_name).await {
        Ok(_) => HttpResponse::NoContent().finish(),
        Err(err) => {
            warn!("Failed to delete PGML model {}: {}", pipeline_name, err);
            HttpResponse::InternalServerError()
                .json(AdminErrorResponse::new("Unable to delete PGML model"))
        }
    }
}

pub async fn dashboard(
    repo: web::Data<Option<PgmlRepository>>,
    auth: web::Data<AdminAuthConfig>,
) -> impl Responder {
pub async fn dashboard(repo: web::Data<Option<PgmlRepository>>) -> impl Responder {
    if repo.get_ref().is_none() {
        return HttpResponse::ServiceUnavailable().body(
            "<html><body><h1>PGML dashboard disabled</h1><p>Set the DATABASE_URL environment variable to enable the admin interface.</p></body></html>",
        );
    }

    let mut body = DASHBOARD_HTML.to_owned();
    if auth.is_enabled() {
        body.push_str(
            "\n<!-- Admin authentication is enabled. Users must supply a bearer token before API calls succeed. -->",
        );
    }

    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(body)
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(DASHBOARD_HTML)
}
