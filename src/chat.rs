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
    pin::Pin,
    sync::{Arc, Mutex},
    time::SystemTime,
};

use crate::{
    utils::{ModelConfig, OpenWebUIProgress},
    AIModel, Content, Message, OpenAiError, ProcessMessages, Role, ShutdownMessages,
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
    let id = "chatcmpl-123".to_owned(); // Todo: 要改從資料庫拿
    let created = SystemTime::now();
    let created = created
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    let Some(llm_config) = all_configs.get(&body.model) else {
        let msg = format!(
            "The model {} does not exist or you do not have access to it.",
            body.model
        );
        log::warn!("{}", msg);
        return HttpResponse::BadRequest().json(OpenAiError {
            message: msg,
            code: "model_not_found".to_owned(),
            r#type: "invalid_request_error".to_owned(),
            param: None,
        });
    };

    let Ok(mut llm_pool_locked) = llm_pool.try_lock() else {
        let msg =
            format!("There is another instance running, please wait other instance finished.");
        log::warn!("{}", msg);
        return HttpResponse::BadRequest().json(OpenAiError {
            message: msg,
            code: "busy".to_owned(),
            r#type: "busy".to_owned(),
            param: None,
        });
    };

    let model_init_progress_stream = if !llm_pool_locked.contains_key(&body.model) {
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
                log::error!("Join failed:{}", err);
            };

            log::info!("建立進度 Stream");
            let (progress_tx, progress_rx) = tokio::sync::mpsc::channel(64);
            let progress_rx_stream = tokio_stream::wrappers::ReceiverStream::new(progress_rx);
            // 建立進度 Stream
            let modelname = body.model.clone();
            let id = id.clone();
            let progress_sse_stream =
                progress_rx_stream.map(move |msg: crate::utils::ProgressMessage| {
                    let id = id.clone();
                    let created = created.clone();
                    log::info!("ProgressMessage: {}", msg.message);
                    // ProgressMessage 序列化為 SSE 格式 (自定義的 Progress 訊息)
                    Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(create_sse_chunk_data(
                        &id,
                        created,
                        &modelname,
                        Some(Role::System),
                        Some(Content::String(msg.message)),
                    )))
                });

                
            log::info!("啟用大模型");
            // ... 建立新的大模型 ...
            let llm_config_clone = llm_config.clone();
            let progress_tx_clone = progress_tx.clone();
            let llm_init_future = tokio::task::spawn_blocking(move || {
                let progress_instance = OpenWebUIProgress::new(progress_tx_clone);
                crate::llm::simple::SimpleRkLLM::init_with_progress(
                    &llm_config_clone,
                    Some(progress_instance),
                )
            });

            log::info!("處理阻塞任務失敗");
            let model_name = llm_config.model_name.clone();
            let llm = match llm_init_future.await {
                Ok(Ok(llm)) => llm,
                // 處理阻塞任務失敗或 init 失敗的情況
                Ok(Err(err)) => {
                    return HttpResponse::InternalServerError().json(OpenAiError {
                        message: format!("LLM init failed: {}", err),
                        code: "model_init_failed".to_owned(),
                        r#type: "model_init_failed".to_owned(),
                        param: None,
                    })
                }
                Err(join_err) => {
                    return HttpResponse::InternalServerError().json(OpenAiError {
                        message: format!("Join error: {}", join_err),
                        code: "join_failed".to_owned(),
                        r#type: "join_failed".to_owned(),
                        param: None,
                    })
                }
            };

            log::info!("llm.start");

            let addr = llm.start(); // 啟動 Actor，一次即可
            llm_pool_locked.insert(
                model_name.clone(),
                addr.clone().recipient::<ProcessMessages>(),
            );
            shutdown_pool
                .lock()
                .unwrap()
                .insert(model_name, addr.clone().recipient::<ShutdownMessages>());
            Some(progress_sse_stream)
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
    } else {
        None
    };

    let Some(llm) = llm_pool_locked.get(&body.model) else {
        panic!("");
    };

    let send_future = llm.send(ProcessMessages {
        messages: body.messages.clone(),
    });

    log::info!("llm.send");
    match actix_web::rt::time::timeout(std::time::Duration::from_secs(60), send_future).await {
        Ok(Ok(Ok(receiver))) => {
            if body.stream {
                let object = "chat.completion.chunk".to_owned();
                let mut stream_counter = 0;
                let llm_output_stream = receiver.map(move |content| {
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

                // 串聯 Stream
                let final_stream: Pin<
                    Box<
                        dyn futures::stream::Stream<Item = Result<web::Bytes, actix_web::Error>>
                            + Send,
                    >,
                > = if let Some(progress_stream) = model_init_progress_stream {
                    // if: 串聯兩個 Boxed Stream
                    Box::pin(progress_stream.chain(llm_output_stream))
                } else {
                    Box::pin(llm_output_stream)
                };
                log::info!("串聯 Stream");
                actix_web::HttpResponse::Ok()
                    .content_type("text/event-stream")
                    .streaming(final_stream)
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
                    id: id.clone(),
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
    role: Option<Role>,
    content: Option<Content>,
) -> String {
    let chunk = ChatCompletionsResponse {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![Choice {
            index: 0,
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
