use actix::{Actor, Recipient};
use actix_web::{
    post,
    web::{self, Json},
    HttpResponse, Responder,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::SystemTime,
};

use crate::{
    utils::ModelConfig, AIModel, Content, Message, OpenAiError, ProcessMessages, Role,
    ShutdownMessages,
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
    #[serde(default)]
    pub stream: bool,
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
    #[serde(rename = "stop")]
    Stop,
    Length,
    FunctionCall,
    InvalidRequestError,
    ModelError,
    InternalError,
}

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub struct Choice {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    #[schema(value_type = String)]
    pub logprobs: Option<String>,
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
    pub model: String,
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
    body: Json<ChatCompletionsRequest>,
    llm_pool: web::Data<Arc<Mutex<HashMap<String, Recipient<ProcessMessages>>>>>,
    shutdown_pool: web::Data<Arc<Mutex<HashMap<String, Recipient<ShutdownMessages>>>>>,
    all_configs: web::Data<HashMap<String, ModelConfig>>,
) -> impl Responder {
    println!("Received chat completion request: {:?}", body);
    let id = "chatcmpl-123".to_owned(); // Todo: 要改從資料庫拿
    let created = SystemTime::now();
    let created = created
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    let Some(llm_config) = all_configs.get(&body.model) else {
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

    let Ok(mut llm_pool_locked) = llm_pool.try_lock() else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!(
                "There is another instance running, please wait other instance finished."
            ),
            code: "busy".to_owned(),
            r#type: "busy".to_owned(),
            param: None,
        });
    };

    if !llm_pool_locked.contains_key(&body.model) {
        if body.stream {
            let shutdowns_tasks = {
                let mut shutdown_pool_lock = shutdown_pool.lock().unwrap(); // MutexGuard 獲得鎖
                shutdown_pool_lock
                    .drain()
                    .map(|(_, addr)| {
                        // addr 是 Recipient<ShutdownMessages> 的所有權
                        async move {
                            let _ = addr.send(ShutdownMessages).await.unwrap();
                        }
                    })
                    .collect::<Vec<_>>() // 收集成 Vec<impl Future>
            };
            llm_pool_locked.clear();

            if let Err(err) = tokio::spawn(async move {
                futures::future::join_all(shutdowns_tasks).await;
            })
            .await
            {
                println!("Join failed:{}", err);
            };

            // ... 建立新的大模型 ...
            let llm_config_clone = llm_config.clone();
            let llm_init_future = tokio::task::spawn_blocking(move || {
                crate::llm::simple::SimpleRkLLM::init(&llm_config_clone)
            });

            let llm = match llm_init_future.await {
                Ok(Ok(llm)) => llm,
                // 處理阻塞任務失敗或 init 失敗的情況
                Ok(Err(err)) => {
                    return HttpResponse::InternalServerError().json(OpenAiError {
                        message: format!(
                            "LLM init failed: {}", err
                        ),
                        code: "model_init_failed".to_owned(),
                        r#type: "model_init_failed".to_owned(),
                        param: None,
                    })
                }
                Err(join_err) => {
                    return HttpResponse::InternalServerError().json(OpenAiError {
                        message: format!(
                            "Join error: {}", join_err
                        ),
                        code: "join_failed".to_owned(),
                        r#type: "join_failed".to_owned(),
                        param: None,
                    })
                }
            };
            let model_name = llm_config.model_name.clone();

            // TODO: 顯示安裝模型中，而且要定期吐資料避免timeout

            let addr = llm.start(); // 啟動 Actor，一次即可
            llm_pool_locked.insert(
                model_name.clone(),
                addr.clone().recipient::<ProcessMessages>(),
            );
            shutdown_pool
                .lock()
                .unwrap()
                .insert(model_name, addr.clone().recipient::<ShutdownMessages>());
        } else {
            return HttpResponse::BadRequest().json(OpenAiError {
                message: format!(
                    "Model not load, please run stream version api to fix this problem."
                ),
                code: "resource_not_found".to_owned(),
                r#type: "resource_not_found".to_owned(),
                param: None,
            });
        }
    }

    let Some(llm) = llm_pool_locked.get(&body.model) else {
        panic!("");
    };

    let send_future = llm.send(ProcessMessages {
        messages: body.messages.clone(),
    });

    match actix_web::rt::time::timeout(std::time::Duration::from_secs(60), send_future).await {
        Ok(Ok(Ok(receiver))) => {
            if body.stream {
                let object = "chat.completion.chunk".to_owned();
                let mut stream_counter = 0;
                let sse_stream = receiver.map(move |content| {
                    let choices = vec![Choice {
                        index: 0,
                        finish_reason: if &content == "" {
                            Some(FinishReason::Stop)
                        } else {
                            None
                        },
                        delta: Some(Message {
                            role: if stream_counter == 0 {
                                Some(Role::Assistant)
                            } else {
                                None
                            },
                            content: if &content == "" {
                                None
                            } else {
                                Some(Content::String(content))
                            },
                        }),
                        logprobs: None,
                        message: None,
                    }];
                    let chunk = ChatCompletionsResponse {
                        id: id.clone(),
                        object: object.clone(),
                        created,
                        model: body.model.clone(),
                        choices,
                        usage: None,
                    };

                    stream_counter += 1;
                    // 將 JSON 序列化為字串並添加換行符
                    let sse_data =
                        "data: ".to_owned() + &serde_json::to_string(&chunk).unwrap() + "\n\n";
                    Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(sse_data))
                    // 轉為 Bytes 並包裝在 Result 中
                });
                actix_web::HttpResponse::Ok()
                    .content_type("text/event-stream")
                    .streaming(sse_stream)
            } else {
                if !llm_pool_locked.contains_key(&body.model) {
                    return HttpResponse::BadRequest().json(OpenAiError {
                        message: format!(
                            "Your request model is not been load, use stream mode chat to enable this model."
                        ),
                        code: "resource_not_found".to_owned(),
                        r#type: "resource_not_found".to_owned(),
                        param: None,
                    });
                }
                let a = receiver.collect::<Vec<_>>().await;
                let content = a.join("");

                // TODO: 執行完解包
                let object = "chat.completion".to_owned();
                let usage = Usage {
                    // TODO: 要給實際數字
                    completion_tokens: 9,
                    prompt_tokens: 9,
                    total_tokens: 9,
                };
                // TODO
                let choices = vec![Choice {
                    index: 0,
                    message: Some(Message {
                        role: Some(Role::Assistant),
                        content: Some(Content::String(content)),
                    }),
                    delta: None,
                    logprobs: None,
                    finish_reason: Some(FinishReason::Stop),
                }];

                HttpResponse::Ok().json(ChatCompletionsResponse {
                    id,
                    object,
                    created,
                    model: body.model.clone(),
                    choices,
                    usage: Some(usage),
                })
            }
        }
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
        Ok(Ok(Err(e))) => HttpResponse::InternalServerError().json(OpenAiError {
            message: format!("Internal processing error: {:?}", e),
            code: "processing_error".to_owned(),
            r#type: "internal_error".to_owned(),
            param: None,
        }),
    }
}

fn create_sse_chunk_data(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    role: Option<Role>,
    content: Option<Content>,
) -> String {
    let chunk = ChatCompletionsResponse {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![Choice {
            index,
            finish_reason: if content.is_none() {
                Some(FinishReason::Stop)
            } else {
                None
            },
            delta: Some(Message { role, content }),
            logprobs: None,
            message: None,
        }],
        usage: None,
    };
    "data: ".to_owned() + &serde_json::to_string(&chunk).unwrap() + "\n\n"
}
