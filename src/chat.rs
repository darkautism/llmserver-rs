use actix_web::{
    post,
    web::{self, Data, Json},
    HttpResponse, Responder,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::SystemTime};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    auth::ApiKeyIdentity,
    db::Database,
    manager::ModelManager,
    token::{estimate_messages_tokens, estimate_text_tokens},
    Content, Message, OpenAiError, ProcessMessages, Role,
};

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Delta {
    #[schema(value_type = Role)]
    pub role: Role,
    #[schema(value_type = Content)]
    pub content: Content,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub enum Stop {
    String(String),
    Array(Vec<String>),
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ResponseFormat {
    //#[schema(enum = ["json_object", "json_object"])]
    pub r#type: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct Tool {
    pub r#type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub enum ToolChoice {
    Auto,
    None,
    Function { name: String },
}

#[derive(Deserialize, Serialize, utoipa::ToSchema, Default)]
#[schema(
    example = json!({
        "model": "DeepSeek-R1-Distill-Qwen-1.5B",
        "messages": [
            {
                "role": "developer",
                "content": "你是一個愚蠢的智慧音箱。除非使用者特別要求回答盡量短促。"
            },
            {
                "role": "user",
                "content": "你好，請問5+3等於多少!"
            }
        ]
    })
)]
#[derive(Debug, Clone)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<i32>,
    pub stream: Option<bool>,
    pub stop: Option<Stop>,
    pub max_tokens: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<i32, f32>>,
    pub user: Option<String>,
    pub response_format: Option<ResponseFormat>,
    pub seed: Option<i32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub enum FinishReason {
    Stop,
    Length,
    FunctionCall,
    InvalidRequestError,
    ModelError,
    InternalError,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub struct Choice {
    pub index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    #[schema(value_type = String)]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub struct Usage {
    pub completion_tokens: i32,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[utoipa::path(
    request_body = ChatCompletionsRequest,
    responses(
        (status = OK, description = "Success", body = ChatCompletionsResponse, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[post("/chat/completions")]
pub async fn chat_completions(
    api_key: ApiKeyIdentity,
    body: Json<ChatCompletionsRequest>,
    manager: Data<ModelManager>,
    db: Data<Database>,
) -> impl Responder {
    let id = Uuid::new_v4().to_string();
    let created = SystemTime::now();
    let created = created
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    let llm_pool = manager.llm_pool().await;
    let Some(llm_pool) = llm_pool.get(&body.model) else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!(
                "The model {} does not exist or you do not have access to it.",
                body.model
            ),
            code: "model_not_found".to_owned(),
            r#type: "invalid_request_error".to_owned(),
            param: None,
        });
    };

    let send_future = llm.send(ProcessMessages {
        messages: body.messages.clone(),
    });

    match actix_web::rt::time::timeout(std::time::Duration::from_secs(5), send_future).await {
        Ok(Ok(Ok(receiver))) => {
            let prompt_tokens = estimate_messages_tokens(&body.messages);
            if body.stream.unwrap_or(false) {
                let object = "chat.completion.chunk".to_owned();
                let stream_counter = Arc::new(Mutex::new(0usize));
                let completions = Arc::new(Mutex::new(String::new()));
                let db_clone = db.clone();
                let model_name = body.model.clone();
                let api_key_id = api_key.id();
                let sse_stream = receiver.then(move |content| {
                    let db = db_clone.clone();
                    let model_name = model_name.clone();
                    let completions = completions.clone();
                    let stream_counter = stream_counter.clone();
                    let object_clone = object.clone();
                    let id_clone = id.clone();
                    async move {
                        let mut counter = stream_counter.lock().await;
                        let is_first = *counter == 0;
                        *counter += 1;
                        drop(counter);

                        if !content.is_empty() {
                            let mut buffer = completions.lock().await;
                            buffer.push_str(&content);
                        } else {
                            let text = {
                                let mut buffer = completions.lock().await;
                                std::mem::take(&mut *buffer)
                            };
                            let completion_tokens = estimate_text_tokens(&text);
                            let _ = db
                                .record_token_usage(
                                    api_key_id,
                                    &model_name,
                                    prompt_tokens,
                                    completion_tokens,
                                )
                                .await;
                        }

                        let choices = vec![Choice {
                            index: 0,
                            finish_reason: if content.is_empty() {
                                Some(FinishReason::Stop)
                            } else {
                                None
                            },
                            message: Some(Message {
                                role: if is_first {
                                    Some(Role::Assistant)
                                } else {
                                    None
                                },
                                content: if content.is_empty() {
                                    None
                                } else {
                                    Some(Content::String(content.clone()))
                                },
                            }),
                            delta: None,
                        }];

                        let chunk = ChatCompletionsResponse {
                            id: id_clone.clone(),
                            object: object_clone.clone(),
                            created,
                            choices,
                            usage: None,
                        };

                        let sse_data = serde_json::to_string(&chunk).unwrap() + "\n";
                        Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(sse_data))
                    }
                });
                actix_web::HttpResponse::Ok()
                    .content_type("text/event-stream")
                    .streaming(sse_stream)
            } else {
                let a = receiver.collect::<Vec<_>>().await;
                let content = a.join("");

                let object = "chat.completion".to_owned();
                let completion_tokens = estimate_text_tokens(&content);
                let usage = Usage {
                    completion_tokens: completion_tokens as i32,
                    prompt_tokens: prompt_tokens as i32,
                    total_tokens: (prompt_tokens + completion_tokens) as i32,
                };
                let choices = vec![Choice {
                    index: 0,
                    message: Some(Message {
                        role: Some(Role::Assistant),
                        content: Some(Content::String(content)),
                    }),
                    delta: None,
                    finish_reason: Some(FinishReason::Stop),
                }];

                if let Err(err) = db
                    .record_token_usage(api_key.id(), &body.model, prompt_tokens, completion_tokens)
                    .await
                {
                    return HttpResponse::InternalServerError().json(OpenAiError {
                        message: format!("Failed to record usage: {}", err),
                        code: "usage_record_error".to_owned(),
                        r#type: "internal_error".to_owned(),
                        param: None,
                    });
                }

                HttpResponse::Ok().json(ChatCompletionsResponse {
                    id,
                    object,
                    created,
                    choices,
                    usage: Some(usage),
                })
            }

            let object = "chat.completion".to_owned();
            let choices = vec![Choice {
                index: 0,
                message: Some(Message {
                    role: Some(Role::Assistant),
                    content: Some(Content::String(content)),
                }),
                delta: None,
                finish_reason: Some(FinishReason::Stop),
            }];

            HttpResponse::Ok().json(ChatCompletionsResponse {
                id,
                object,
                created,
                choices,
                usage: Some(usage),
            })
        }
        Ok(Ok(Err(e))) => HttpResponse::InternalServerError().json(OpenAiError {
            message: format!("Internal processing error: {:?}", e),
            code: "processing_error".to_owned(),
            r#type: "internal_error".to_owned(),
            param: None,
        }),
        Err(_timeout) => HttpResponse::UnavailableForLegalReasons().json(OpenAiError {
            message: format!("Server Busy."),
            code: "server_".to_owned(),
            r#type: "internal_error".to_owned(),
            param: None,
        }),
        Ok(Err(e)) => HttpResponse::UnavailableForLegalReasons().json(OpenAiError {
            message: format!("Internal server error:{}", e),
            code: "server_".to_owned(),
            r#type: "internal_error".to_owned(),
            param: None,
        }),
    }
}
