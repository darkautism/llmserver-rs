use actix::{Actor, Recipient};
use actix_web::{
    post,
    web::{self, Json},
    HttpResponse, Responder,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
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
    let id = "chatcmpl-123".to_owned();
    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // 1. 檢查模型設定是否存在
    let Some(llm_config) = all_configs.get(&body.model) else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!("Model {} does not exist.", body.model),
            code: "model_not_found".to_owned(),
            r#type: "invalid_request_error".to_owned(),
            param: None,
        });
    };

    // 準備要移入 Stream 的資源 (Clone 指標)
    let llm_pool = llm_pool.clone();
    let shutdown_pool = shutdown_pool.clone();
    let model_name = body.model.clone();
    let messages = body.messages.clone();
    let is_stream_mode = body.stream;
    let llm_config = llm_config.clone();

    // 檢查模型是否已載入 (這裡只做快速檢查，不長時間持有鎖)
    let model_exists = llm_pool
        .try_lock()
        .map(|m| m.contains_key(&model_name))
        .unwrap_or(false);

    // 如果模型不存在且不是 Stream 模式，直接報錯
    if !model_exists && !is_stream_mode {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: "Model not loaded. Use stream mode to load.".to_owned(),
            code: "resource_not_found".to_owned(),
            r#type: "resource_not_found".to_owned(),
            param: None,
        });
    }
    // 定義單一的輸出串流：這是你的主要骨牌鏈
    let outbound_stream: Pin<Box<dyn Stream<Item = Result<web::Bytes, actix_web::Error>>>> =
        Box::pin(async_stream::try_stream! {

            // ==========================================
            // 階段一：取得 LLM Actor (可能是現有的，或是剛載入的)
            // ==========================================

            let recipient = if model_exists {
                // [情況 A] 模型已經在 Pool 裡
                let pool = llm_pool.lock().unwrap(); // 這裡可以 blocking lock，因為只是讀 HashMap 很快
                pool.get(&model_name).cloned().unwrap() // 拿出 Recipient
            } else {
                // [情況 B] 需要載入模型 (長任務)

                // 1. 先進行清理工作 (快速同步操作)
                {
                    let mut pool = llm_pool.lock().unwrap();
                    let mut shutdown = shutdown_pool.lock().unwrap();
                    pool.clear();
                    let tasks: Vec<_> = shutdown.drain().map(|(_, addr)| async move {
                        let _ = addr.send(ShutdownMessages).await;
                    }).collect();
                    // 在背景執行關閉，不等待
                    tokio::spawn(futures::future::join_all(tasks));
                }

                // 2. 建立溝通管道
                log::info!("建立進度 Stream");
                let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel(64);

                // 3. 啟動背景任務
                // 注意：這裡不用 await，讓它開始在另一個執行緒跑
                let join_handle = tokio::task::spawn_blocking(move || {
                    let progress = OpenWebUIProgress::new(progress_tx);
                    // 這會跑很久 (5-10分鐘)
                    crate::llm::simple::SimpleRkLLM::init_with_progress(&llm_config, Some(progress))
                });

                // 4. 進入「讀取進度」迴圈
                // 只要背景任務還在跑，progress_rx 就會一直收到資料
                let mut bar = indicatif::ProgressBar::new(1000);
                let mut first_download_done = true;
                let mut percent = -1_i64;
                let mut count = 0;
                while let Some(msg) = progress_rx.recv().await {
                    let mut edited_msg = msg.message.clone();
                    if !msg.download_done && msg.current == 0 { // 剛開始下載
                        bar = indicatif::ProgressBar::new(msg.total as u64);
                        edited_msg += "<think>"
                    } else if msg.download_done && first_download_done { // 剛結束下載
                        first_download_done=false;
                        bar.finish();
                        bar = indicatif::ProgressBar::new_spinner();
                        bar.enable_steady_tick(Duration::from_millis(100));
                        log::info!("Progress: {}", msg.message);
                        if count == 0 { // 這個case是一開始就從快取拿
                            edited_msg = format!("{}<think>\n", edited_msg);
                        } else {
                            edited_msg = format!("</think>{}<think>下載完成\n", edited_msg);
                        }
                    } else if msg.finished { // 整個結束
                        bar.finish();
                        log::info!("Progress: {}", msg.message);
                        edited_msg = format!("</think>");
                    } else {
                        bar.set_position(msg.current as u64);
                    }

                    if msg.current != 0 && !msg.download_done && percent == (msg.current*100/ msg.total) as i64 {
                        // do not display
                    } else {
                        if !msg.download_done {
                            percent = (msg.current*100/ msg.total) as i64;
                        }
                        // 立即吐出 SSE 給前端
                        let sse = create_sse_chunk_data(
                            &id, created, &model_name,
                            Some(Role::System), Some(Content::String(edited_msg))
                        );
                        yield web::Bytes::from(sse);
                    }
                    count += 1;
                }
                // 當迴圈結束，表示背景任務做完了 (Channel 被 Drop)

                // 因為上面迴圈結束代表任務已停，這裡的 await 會瞬間完成
                let llm_result = join_handle.await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Join err: {}", e)))?;

                let llm = llm_result
                    .map_err(|e| actix_web::error::ErrorInternalServerError(format!("Init err: {}", e)))?;

                // 6. 啟動 Actor 並更新 Pool
                log::info!("模型載入完成，啟動 Actor");
                let addr = llm.start();
                let recipient = addr.clone().recipient::<ProcessMessages>();

                // 更新全域狀態
                llm_pool.lock().unwrap().insert(model_name.clone(), recipient.clone());
                shutdown_pool.lock().unwrap().insert(model_name.clone(), addr.recipient::<ShutdownMessages>());

                recipient
            };

            // ==========================================
            // 階段二：執行對話 (Chat Completions)
            // ==========================================


            // 發送訊息給 Actor
            let send_future = recipient.send(ProcessMessages {
                messages: messages.clone(),
            });


            // 等待 Actor 回應 (設定 60 秒 Timeout)
            // 注意：這裡是等待「開始生成」，而不是等待「生成完畢」
            let actor_response = match actix_web::rt::time::timeout(std::time::Duration::from_secs(60), send_future).await {
                Ok(Ok(res)) => res,
                Ok(Err(e)) => {
                    yield web::Bytes::from(format!("data: {{\"error\": \"Mailbox error: {}\"}}\n\n", e));
                    return;
                },
                Err(_) => {
                    yield web::Bytes::from("data: {\"error\": \"Timeout waiting for model slot\"}\n\n");
                    return;
                }
            };

            // 解開 Result<Stream>
            let mut chat_stream = match actor_response {
                Ok(s) => s,
                Err(_) => {
                    yield web::Bytes::from("data: {\"error\": \"Internal stream error\"}\n\n");
                    return;
                }
            };

            // ==========================================
            // 階段三：串流輸出 Token
            // ==========================================
            let mut stream_counter = 0;
            while let Some(content) = chat_stream.next().await {
                let chunk = ChatCompletionsResponse {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_owned(),
                    created,
                    model: model_name.clone(),
                    choices: vec![Choice {
                        index: 0,
                        finish_reason: if content.is_empty() { Some(FinishReason::Stop) } else { None },
                        delta: Some(Message {
                            role: if stream_counter == 0 { Some(Role::Assistant) } else { None },
                            content: if content.is_empty() { None } else { Some(Content::String(content)) },
                        }),
                        logprobs: None,
                        message: None,
                    }],
                    usage: None,
                };
                stream_counter += 1;
                
                let sse_data = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                yield web::Bytes::from(sse_data);
            }
        });

    // 根據請求模式回傳
    if is_stream_mode {
        HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(outbound_stream)
    } else {
        // 為了相容舊邏輯，如果不是 Stream 模式但還是跑到這裡 (例如本來已存在)
        // 我們還是得把 Stream 收集回來轉成 JSON
        // 但因為上面的 outbound_stream 已經包含了 SSE 格式化，這會變得有點髒。
        // 建議：如果你的前端能支援，統一都用 SSE 回傳最簡單。
        // 若必須支援非 Stream，建議在最外層拆開邏輯，或者這裡做個簡易的 collect

        // 這裡示範直接用 Streaming 回傳，通常現代 LLM API 即使不開 stream 參數，
        // 內部邏輯一致比較好維護，或者你需要重寫一段專門收集 Vec 的邏輯。
        HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(outbound_stream)
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
