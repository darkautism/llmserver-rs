use actix::Actor;
use hf_hub::api::sync::Api;
use hf_hub::api::Progress;
use rkllm_rs::prelude::*;
use serde_variant::to_variant_name;
use std::ffi::CString;
use std::fs;
use std::pin::Pin;
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

use autotokenizer::AutoTokenizer;
use autotokenizer::DefaultPromptMessage;

use crate::utils::ModelConfig;
use crate::AIModel;
use crate::ModelProgress;
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

        let input = match atoken.apply_chat_template(prompt, true) {
            Ok(parsed) => parsed,
            Err(err) => {
                log::warn!("apply_chat_template failed. Error: {:?}", err);
                "".to_owned()
            }
        };

        let think = self.config.think.unwrap_or(false);

        let handle = self.handle.clone();
        let infer_params_cloned = self.infer_params.clone();
        actix_web::rt::spawn(async move {
            let cb = CallbackSendSelfChannel {
                sender: Some(tx),
                abort: Box::new(move || {
                    if let Err(err) = handle.abort() {
                        log::info!("Abort rkllm failed: {}", err)
                    }
                }),
            };
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
    fn init_with_progress<P: Progress + ModelProgress + Clone>(
        config: &Self::Config,
        p: Option<P>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut param = RKLLMParam {
            ..Default::default()
        };
        let api = Api::new().unwrap();

        let repo = api.model(config.model_repo.clone());
        let tokenizer_repo = config
            .tokenizer_repo
            .clone()
            .unwrap_or(config.model_repo.clone());
        let filename = &config
            .model_path
            .clone()
            .unwrap_or("model.rkllm".to_owned());
        let (binding, progress) = if let Some(progress) = p {
            let ret = repo.download_with_progress(filename, progress.clone())?;
            (ret, Some(progress))
        } else {
            (repo.get(filename)?, None)
        };
        let model_path = binding.to_string_lossy();
        let c_str = CString::new(model_path.as_ref()).unwrap();
        param.model_path = c_str.as_ptr();
        let progress = if let Some(mut progress) = progress {
            let meta = fs::metadata(&binding)?;
            progress.model_load(meta.len().try_into().unwrap(), filename, Instant::now());
            Some(progress)
        } else {
            None
        };

        let handle = match rkllm_init(&mut param) {
            Ok(handle) => handle,
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error initializing RKLLM: {:?}", e),
                )));
            }
        };
        let atoken = match AutoTokenizer::from_pretrained(tokenizer_repo, None) {
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

        if let Some(mut progress) = progress {
            progress.model_finished();
        }

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
    abort: Box<dyn FnMut() + Send + Sync + 'static>,
}
impl RkllmCallbackHandler for CallbackSendSelfChannel {
    fn handle(&mut self, result: Option<RKLLMResult>, state: LLMCallState) {
        match state {
            LLMCallState::Normal => {
                if let Some(result) = result {
                    if let Some(sender) = &self.sender {
                        let mut err = Ok(());
                        let mut first = true;
                        while first || err.is_err() {
                            first = false;
                            match err {
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                    // 通知模型關閉
                                    (self.abort)();
                                    return;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                    std::thread::yield_now();
                                }
                                Ok(_) => {}
                            }
                            err = sender.try_send(result.text.clone());
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
