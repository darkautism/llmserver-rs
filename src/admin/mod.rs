use std::{fmt::Write, time::Duration};

use actix_web::{
    get, post,
    web::{self, Data, Form},
    HttpRequest, HttpResponse, Responder,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    auth::{self, session_cookie, AdminSession},
    db::{ApiKeyRecord, Database, DatabaseError, ModelDownloadRecord},
    hf::{DownloaderError, HuggingFaceDownloader},
    manager::{ModelInstance, ModelManager, ModelManagerError},
    utils::{ModelConfig, ModelType},
};

#[derive(Debug, Deserialize)]
pub struct LoginForm {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiKeyForm {
    pub name: String,
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderTokenForm {
    pub name: String,
    pub token: String,
}

#[derive(Debug, Deserialize)]
pub struct DownloadForm {
    pub repo_id: String,
    pub revision: Option<String>,
    pub model_type: ModelType,
    pub credential_name: String,
}

#[derive(Debug, Deserialize)]
pub struct StartModelForm {
    pub repo_id: String,
    pub instances: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct StopModelForm {
    pub instance_id: String,
}

#[get("/admin/login")]
pub async fn login_page() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("templates/login.html"))
}

#[post("/admin/login")]
pub async fn login(
    form: Form<LoginForm>,
    db: Data<Database>,
    req: HttpRequest,
) -> Result<HttpResponse, actix_web::Error> {
    let Some(user) = db
        .verify_user(&form.username, &form.password)
        .await
        .map_err(map_db_error)?
    else {
        return Ok(HttpResponse::Unauthorized()
            .content_type("text/html; charset=utf-8")
            .body("<p>Invalid credentials</p><a href=\"/admin/login\">Back</a>"));
    };
    let token = db
        .create_session(user.id, Duration::from_hours(8))
        .await
        .map_err(map_db_error)?;
    let secure = req.connection_info().scheme() == "https";
    Ok(HttpResponse::Found()
        .append_header((actix_web::http::header::LOCATION, "/admin"))
        .cookie(session_cookie(&token, secure))
        .finish())
}

#[get("/admin/logout")]
pub async fn logout(
    session: AdminSession,
    db: Data<Database>,
) -> Result<HttpResponse, actix_web::Error> {
    db.revoke_session(&session.token)
        .await
        .map_err(map_db_error)?;
    Ok(auth::clear_session_cookie())
}

#[get("/admin")]
pub async fn dashboard(
    _session: AdminSession,
    db: Data<Database>,
    manager: Data<ModelManager>,
) -> Result<HttpResponse, actix_web::Error> {
    let api_keys = db.list_api_keys().await.map_err(map_db_error)?;
    let downloads = db.list_model_downloads().await.map_err(map_db_error)?;
    let instances = manager.list_instances().await;
    let configs = manager.configs().await;

    let mut body = String::new();
    render_dashboard(&mut body, &api_keys, &downloads, &instances, &configs)?;
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(body))
}

#[post("/admin/api-keys")]
pub async fn create_api_key(
    _session: AdminSession,
    db: Data<Database>,
    form: Form<ApiKeyForm>,
) -> Result<HttpResponse, actix_web::Error> {
    let (token, record) = db
        .create_api_key(&form.name, form.scope.as_deref())
        .await
        .map_err(map_db_error)?;
    let body = format!(
        "<p>Created API key <strong>{}</strong>. Store this token securely:</p><pre>{}</pre><p><a href=\"/admin\">Back</a></p>",
        record.name, token
    );
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(body))
}

#[post("/admin/provider-credentials")]
pub async fn store_provider_token(
    _session: AdminSession,
    db: Data<Database>,
    form: Form<ProviderTokenForm>,
) -> Result<HttpResponse, actix_web::Error> {
    db.store_provider_secret("huggingface", &form.name, &form.token)
        .await
        .map_err(map_db_error)?;
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body("<p>Credential stored.</p><a href=\"/admin\">Back</a>"))
}

#[post("/admin/models/download")]
pub async fn download_model(
    _session: AdminSession,
    db: Data<Database>,
    downloader: Data<HuggingFaceDownloader>,
    form: Form<DownloadForm>,
) -> Result<HttpResponse, actix_web::Error> {
    let token = db
        .get_provider_secret("huggingface", &form.credential_name)
        .await
        .map_err(map_db_error)?
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Credential not found"))?;
    let snapshot = downloader
        .download_repo(&form.repo_id, form.revision.as_deref(), Some(&token))
        .map_err(map_download_error)?;
    let size_bytes =
        compute_dir_size(&snapshot).map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    db.upsert_model_download(
        &form.repo_id,
        form.revision.as_deref(),
        snapshot.to_string_lossy().as_ref(),
        match form.model_type {
            ModelType::LLM => "LLM",
            ModelType::ASR => "ASR",
            ModelType::TTS => "TTS",
        },
        size_bytes as i64,
    )
    .await
    .map_err(map_db_error)?;
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body("<p>Model downloaded.</p><a href=\"/admin\">Back</a>"))
}

#[post("/admin/models/start")]
pub async fn start_model(
    _session: AdminSession,
    manager: Data<ModelManager>,
    form: Form<StartModelForm>,
) -> Result<HttpResponse, actix_web::Error> {
    let started = manager
        .start_instances(&form.repo_id, form.instances.unwrap_or(1))
        .await
        .map_err(map_manager_error)?;
    let mut response = String::from("<h2>Started instances</h2><ul>");
    for instance in started {
        let _ = write!(
            &mut response,
            "<li>{} ({:?}) - {}</li>",
            instance.model_name, instance.model_type, instance.id
        );
    }
    response.push_str("</ul><a href=\"/admin\">Back</a>");
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(response))
}

#[post("/admin/models/stop")]
pub async fn stop_model(
    _session: AdminSession,
    manager: Data<ModelManager>,
    form: Form<StopModelForm>,
) -> Result<HttpResponse, actix_web::Error> {
    let id = Uuid::parse_str(&form.instance_id)
        .map_err(|_| actix_web::error::ErrorBadRequest("Invalid instance id"))?;
    manager.stop_instance(id).await.map_err(map_manager_error)?;
    Ok(HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body("<p>Instance stopped.</p><a href=\"/admin\">Back</a>"))
}

fn render_dashboard(
    body: &mut String,
    api_keys: &[ApiKeyRecord],
    downloads: &[ModelDownloadRecord],
    instances: &[ModelInstance],
    configs: &std::collections::HashMap<String, ModelConfig>,
) -> Result<(), std::fmt::Error> {
    body.push_str(include_str!("templates/header.html"));
    body.push_str("<section><h2>Active Model Instances</h2><table><thead><tr><th>ID</th><th>Name</th><th>Type</th><th>Started</th></tr></thead><tbody>");
    for instance in instances {
        write!(
            body,
            "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{}</td></tr>",
            instance.id, instance.model_name, instance.model_type, instance.created_at
        )?;
    }
    if instances.is_empty() {
        body.push_str("<tr><td colspan=4>No running instances</td></tr>");
    }
    body.push_str("</tbody></table></section>");

    body.push_str("<section><h2>Configured Models</h2><ul>");
    for (repo, config) in configs {
        write!(
            body,
            "<li><strong>{}</strong> - {} ({:?})</li>",
            repo, config.model_name, config.model_type
        )?;
    }
    body.push_str("</ul></section>");

    body.push_str("<section><h2>API Keys</h2><ul>");
    for key in api_keys {
        write!(body, "<li>{} (scope: {:?})</li>", key.name, key.scope)?;
    }
    if api_keys.is_empty() {
        body.push_str("<li>No API keys configured</li>");
    }
    body.push_str("</ul></section>");

    body.push_str("<section><h2>Downloads</h2><table><thead><tr><th>Repo</th><th>Revision</th><th>Type</th><th>Size</th><th>Path</th><th>At</th></tr></thead><tbody>");
    for download in downloads {
        write!(
            body,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2} MiB</td><td>{}</td><td>{}</td></tr>",
            download.repo_id,
            download.revision,
            download.model_type,
            download.size_bytes as f64 / (1024.0 * 1024.0),
            download.local_path,
            download.downloaded_at
        )?;
    }
    if downloads.is_empty() {
        body.push_str("<tr><td colspan=6>No downloads yet</td></tr>");
    }
    body.push_str("</tbody></table></section>");

    body.push_str(include_str!("templates/forms.html"));
    body.push_str("</main></body></html>");
    Ok(())
}

fn compute_dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    if path.is_file() {
        return Ok(path.metadata()?.len());
    }
    let mut size = 0;
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            size += entry.metadata()?.len();
        }
    }
    Ok(size)
}

fn map_db_error(err: DatabaseError) -> actix_web::Error {
    actix_web::error::ErrorInternalServerError(err)
}

fn map_manager_error(err: ModelManagerError) -> actix_web::Error {
    actix_web::error::ErrorBadRequest(err)
}

fn map_download_error(err: DownloaderError) -> actix_web::Error {
    match err {
        DownloaderError::MissingToken => actix_web::error::ErrorBadRequest(err),
        DownloaderError::Hub(_) | DownloaderError::Io(_) => {
            actix_web::error::ErrorInternalServerError(err)
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(login_page)
        .service(login)
        .service(logout)
        .service(dashboard)
        .service(create_api_key)
        .service(store_provider_token)
        .service(download_model)
        .service(start_model)
        .service(stop_model);
}
