use std::{pin::Pin, sync::Arc};

use actix::Actor;
use hound::WavReader;
use sensevoice_rs::{fsmn_vad::VADXOptions, SenseVoiceSmall};
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::{AIModel, AsrText, ProcessAudio, ShutdownMessages, ASR};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SimpleASRConfig {
    pub model_repo: String,
    pub model_name: String,
}

pub struct SimpleASR {
    handle: Arc<SenseVoiceSmall>,
}

impl Actor for SimpleASR {
    type Context = actix::Context<Self>;
}

impl actix::Handler<ProcessAudio> for SimpleASR {
    type Result = Result<Pin<Box<dyn futures::Stream<Item = AsrText> + Send + 'static>>, ()>;
    fn handle(&mut self, msg: ProcessAudio, _ctx: &mut Self::Context) -> Self::Result {
        let (tx, rx) = tokio::sync::mpsc::channel::<AsrText>(64);

        let handle_clone = self.handle.clone();
        actix_web::rt::spawn(async move {
            let allseg = match msg {
                ProcessAudio::FilePath(audio_path) => handle_clone
                    .infer_file(audio_path)
                    .expect("Infer file failed"),
                ProcessAudio::Buffer(read) => {
                    // TODO: sensevoice-rs not support reader now, so read all!
                    let mut wav_reader =
                        WavReader::new(read).expect("Should give me wave reader but not!");
                    let content = wav_reader
                        .samples()
                        .filter_map(|x| x.ok())
                        .collect::<Vec<i16>>();
                    handle_clone
                        .infer_vec(content, 16000)
                        .expect("Infer vec failed")
                }
            };
            for seg in allseg {
                // TODO: Maybe someday should have good error handling
                let _ = tx.send(AsrText::SenseVoice(seg)).await;
            }
        });

        // 將 Receiver 轉換為 Stream
        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

impl actix::Handler<ShutdownMessages> for SimpleASR {
    type Result = Result<(), ()>;

    fn handle(&mut self, _msg: ShutdownMessages, _ctx: &mut Self::Context) -> Self::Result {
        // TODO: Maybe someday should have good error handling
        let _ = self.handle.destroy();
        Ok(())
    }
}

impl ASR for SimpleASR {}

impl AIModel for SimpleASR {
    type Config = SimpleASRConfig;

    fn init(config: &Self::Config) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized,
    {
        let handle = Arc::new(
            SenseVoiceSmall::init(&config.model_repo, VADXOptions::default())
                .map_err(|_| "Load model error")?,
        );
        Ok(SimpleASR { handle })
    }
}
