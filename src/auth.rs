use actix_web::{
    dev::Payload,
    error::{ErrorInternalServerError, ErrorUnauthorized},
    http::header,
    web::Data,
    FromRequest, HttpRequest, HttpResponse,
};

use crate::db::{ApiKeyRecord, Database, DatabaseError, UserRecord};

pub const ADMIN_SESSION_COOKIE: &str = "ADMIN_SESSION";

#[derive(Clone, Debug)]
pub struct ApiKeyIdentity {
    pub record: ApiKeyRecord,
    token: String,
}

impl ApiKeyIdentity {
    pub fn id(&self) -> i64 {
        self.record.id
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

impl FromRequest for ApiKeyIdentity {
    type Error = actix_web::Error;
    type Future = futures::future::LocalBoxFuture<'static, Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        let db = req
            .app_data::<Data<Database>>()
            .cloned()
            .expect("Database must be configured");
        let token = extract_api_key(req);
        Box::pin(async move {
            let token = token.ok_or_else(|| ErrorUnauthorized("missing API key"))?;
            let record = db
                .verify_api_key(&token)
                .await
                .map_err(map_db_error)?
                .ok_or_else(|| ErrorUnauthorized("invalid API key"))?;
            Ok(ApiKeyIdentity { record, token })
        })
    }
}

fn extract_api_key(req: &HttpRequest) -> Option<String> {
    if let Some(header_value) = req.headers().get("x-api-key") {
        if let Ok(value) = header_value.to_str() {
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    if let Some(header_value) = req.headers().get(header::AUTHORIZATION) {
        if let Ok(value) = header_value.to_str() {
            if let Some(rest) = value.strip_prefix("Bearer ") {
                return Some(rest.trim().to_owned());
            }
        }
    }
    req
        .query_string()
        .split('&')
        .find_map(|pair| {
            let mut iter = pair.splitn(2, '=');
            match (iter.next(), iter.next()) {
                (Some("api_key"), Some(value)) if !value.is_empty() => {
                    Some(value.to_string())
                }
                _ => None,
            }
        })
}

#[derive(Clone, Debug)]
pub struct AdminSession {
    pub user: UserRecord,
    pub token: String,
}

impl FromRequest for AdminSession {
    type Error = actix_web::Error;
    type Future = futures::future::LocalBoxFuture<'static, Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        let db = req
            .app_data::<Data<Database>>()
            .cloned()
            .expect("Database must be configured");
        let token = req
            .cookie(ADMIN_SESSION_COOKIE)
            .map(|cookie| cookie.value().to_owned());
        Box::pin(async move {
            let token = token.ok_or_else(|| ErrorUnauthorized("session required"))?;
            let user = db
                .resolve_session(&token)
                .await
                .map_err(map_db_error)?
                .ok_or_else(|| ErrorUnauthorized("session expired"))?;
            Ok(AdminSession { user, token })
        })
    }
}

fn map_db_error(err: DatabaseError) -> actix_web::Error {
    ErrorInternalServerError(err)
}

pub fn clear_session_cookie() -> HttpResponse {
    HttpResponse::Found()
        .append_header((header::LOCATION, "/admin/login"))
        .del_cookie(actix_web::cookie::Cookie::named(ADMIN_SESSION_COOKIE))
        .finish()
}

pub fn session_cookie(token: &str, secure: bool) -> actix_web::cookie::Cookie<'static> {
    let mut cookie = actix_web::cookie::Cookie::build(ADMIN_SESSION_COOKIE, token.to_owned())
        .path("/admin")
        .http_only(true)
        .same_site(actix_web::cookie::SameSite::Lax)
        .finish();
    if secure {
        cookie.set_secure(true);
    }
    cookie
}
