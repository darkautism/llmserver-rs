use std::collections::HashMap;

use actix::Recipient;
use actix_web::{
    get, post,
    web::{self, Json},
    HttpResponse, Responder,
};
use serde::{Deserialize, Serialize};

use crate::{utils::ModelConfig, ProcessMessages};

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
struct ListModel {
    pub object: String,
    pub data: Vec<Model>,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
struct Model {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u32,
    #[serde(default)]
    pub owned_by: String,
}

#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = Model, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[get("/models")]
pub async fn models(all_configs: web::Data<HashMap<String, ModelConfig>>) -> impl Responder {
    
    HttpResponse::Ok().json(ListModel {
        object: "list".to_string(),
        data: all_configs.iter().map(|(_,config)| Model {
                id: config.model_name.clone(),
                object: "model".to_string(),
                created: 0,
                owned_by: "llmserver-rs".to_string(),
            })
            .collect::<Vec<Model>>(),
    })
}
