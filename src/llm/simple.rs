use actix::Actor;
use hf_hub::api::sync::Api;
use rkllm_rs::prelude::*;
use serde_variant::to_variant_name;
use std::ffi::CString;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;

use autotokenizer::AutoTokenizer;
use autotokenizer::DefaultPromptMessage;

use crate::utils::ModelConfig;
use crate::AIModel;
use crate::ProcessMessages;
use crate::ShutdownMessages;
use crate::LLM;

#[derive(Debug)]
pub struct SimpleRkLLM {
    handle: LLMHandle,
    atoken: AutoTokenizer,
    infer_params: RKLLMInferParam,
    config: ModelConfig,
}

impl Actor for SimpleRkLLM {
    type Context = actix::Context<Self>;
}

impl actix::Handler<ProcessMessages> for SimpleRkLLM {
    type Result = Result<Pin<Box<dyn futures::Stream<Item = String> + Send + 'static>>, ()>;

    fn handle(&mut self, msg: ProcessMessages, _ctx: &mut Self::Context) -> Self::Result {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let atoken = self.atoken.clone();
        let prompt = msg
            .messages
            .iter()
            .map(|a| {
                let content = match &a.content {
                    Some(crate::Content::String(s)) => s,
                    Some(crate::Content::Array(items)) => &items.join(""),
                    None => "", // 老實說不應該發生
                };
                DefaultPromptMessage::new(to_variant_name(&a.role).unwrap(), &content)
            })
            .collect::<Vec<_>>();

        let mut input = match atoken.apply_chat_template(prompt, true) {
            Ok(parsed) => parsed,
            Err(_) => {
                println!("apply_chat_template failed.");
                "".to_owned()
            }
        };
        // TODO: 用參數判斷要不要think
        let think = self.config.think.unwrap_or(false);
        if !think {
            input += "\n\n</think>\n\n";
        }

        let handle = self.handle.clone();
        let infer_params_cloned = self.infer_params.clone();
        actix_web::rt::spawn(async move {
            let cb = CallbackSendSelfChannel { sender: Some(tx) };
            // TODO: Maybe someday should have good error handling
            let _ = handle.run(
                RKLLMInput {
                    input_type: RKLLMInputType::Prompt(input.clone()),
                    enable_thinking: think,
                    role: RKLLMInputRole::User,
                },
                Some(infer_params_cloned),
                cb,
            );
        });

        // 將 Receiver 轉換為 Stream
        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

impl actix::Handler<ShutdownMessages> for SimpleRkLLM {
    type Result = Result<(), ()>;

    fn handle(&mut self, _: ShutdownMessages, _: &mut Self::Context) -> Self::Result {
        // TODO: Maybe someday should have good error handling
        let _ = self.handle.destroy();
        Ok(())
    }
}

impl AIModel for SimpleRkLLM {
    type Config = ModelConfig;
    fn init(config: &ModelConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut param = RKLLMParam {
            ..Default::default()
        };
        let api = Api::new().unwrap();
        let repo = api.model(config.model_repo.clone());
        let binding = repo.get(
            &config
                .model_path
                .clone()
                .unwrap_or("model.rkllm".to_owned()),
        )?;
        let model_path = binding.to_string_lossy();
        let c_str = CString::new(model_path.as_ref()).unwrap();
        param.model_path = c_str.as_ptr();

        let handle = match rkllm_init(&mut param) {
            Ok(handle) => handle,
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error initializing RKLLM: {:?}", e),
                )));
            }
        };
        let atoken = match AutoTokenizer::from_pretrained(config.model_repo.clone(), None) {
            Ok(atoken) => atoken,
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error loading tokenizer: {:?}", e),
                )));
            }
        };

        let infer_params = RKLLMInferParam {
            mode: RKLLMInferMode::InferGenerate,
            lora_params: None,
            prompt_cache_params: if let Some(cache_path) = &config.cache_path {
                Some(RKLLMPromptCacheParam {
                    save_prompt_cache: true,
                    prompt_cache_path: cache_path.to_owned(),
                })
            } else {
                None
            },
            ..Default::default()
        };

        Ok(SimpleRkLLM {
            handle,
            atoken,
            infer_params,
            config: config.clone(),
        })
    }
}

impl LLM for SimpleRkLLM {}

struct CallbackSendSelfChannel {
    sender: Option<tokio::sync::mpsc::Sender<String>>,
}
impl RkllmCallbackHandler for CallbackSendSelfChannel {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        match state {
            LLMCallState::Normal => {
                if let Some(result) = result {
                    if let Some(sender) = &self.sender {
                        while sender.try_send(result.text.clone()).is_err() {
                            std::thread::yield_now();
                        }
                    }
                }
            }
            LLMCallState::Waiting => {}
            LLMCallState::Finish => {
                drop(self.sender.take());
            }
            LLMCallState::Error => {}
            LLMCallState::GetLastHiddenLayer => {}
        }
    }
}
