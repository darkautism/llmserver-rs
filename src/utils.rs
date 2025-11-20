use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use hf_hub::api::Progress;
use indicatif::HumanBytes;
use serde::Deserialize;

use crate::ModelProgress;

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    #[default]
    LLM,
    ASR,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelConfig {
    pub model_repo: String,
    pub model_name: String,
    pub model_type: ModelType,
    pub model_path: Option<String>,
    pub tokenizer_repo: Option<String>,
    #[serde(skip_deserializing)]
    pub _asserts_path: String,
    pub cache_path: Option<String>,
    pub think: Option<bool>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProgressMessage {
    pub current: usize,
    pub total: usize,
    pub download_done: bool,
    pub finished: bool,
    pub message: String,
}

#[derive(Debug)]
pub struct OpenWebUIProgress {
    // 透過 MPSC Sender 將進度發送到 Actix Web Handler
    sender: tokio::sync::mpsc::Sender<ProgressMessage>,
    current: usize,
    total: usize,
    // ModelProgress 相關狀態
    start_time: Option<std::time::Instant>, // 模型載入的開始時間
    stop_flag: Arc<AtomicBool>,
    update_handle: Option<std::thread::JoinHandle<()>>,
}

impl Clone for OpenWebUIProgress {
    fn clone(&self) -> Self {
        // 複製 sender 和基本狀態
        let new_instance = Self {
            sender: self.sender.clone(),
            current: self.current,
            total: self.total,
            start_time: self.start_time,
            stop_flag: Arc::new(AtomicBool::new(false)),
            update_handle: None,
        };
        new_instance
    }
}

impl OpenWebUIProgress {
    // 建立一個靜態方法來產生 (Sender, Receiver) 配對
    pub fn new(sender: tokio::sync::mpsc::Sender<ProgressMessage>) -> Self {
        Self {
            sender,
            total: 0,
            current: 0,
            start_time: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            update_handle: None,
        }
    }
}

impl Progress for OpenWebUIProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total = size;
        self.current = 0;
        let msg = ProgressMessage {
            current: 0,
            total: size,
            download_done: false,
            finished: false,
            message: format!("開始下載模型：{}", filename),
        };
        // 由於我們在同步 Trait 裡，不能 await，我們必須用 try_send 或 blocking_send (如果需要)
        // 這裡我們假設 MPSC 緩衝區夠大，使用 try_send
        let _ = self.sender.try_send(msg);
    }

    fn update(&mut self, size: usize) {
        self.current += size;
        let msg = ProgressMessage {
            current: self.current,
            total: self.total,
            download_done: false,
            finished: false,
            message: format!("下載中... {}/{}\n", self.current, self.total),
        };
        let _ = self.sender.try_send(msg);
    }

    fn finish(&mut self) {
        let msg = ProgressMessage {
            current: self.total,
            total: self.total,
            download_done: true,
            finished: false,
            message: "下載完成，正在初始化模型...".to_owned(),
        };
        let _ = self.sender.try_send(msg);
    }
}

// 實作 ModelProgress
impl ModelProgress for OpenWebUIProgress {
    fn model_load(&mut self, size: usize, filename: &str, start: std::time::Instant) {
        self.start_time = Some(start);
        // 發送載入開始訊息
        let msg = ProgressMessage {
            current: self.current,
            total: self.total,
            download_done: true,
            finished: false,
            message: format!(
                "下載完成，開始載入 RKLLM 核心 {} ({})...",
                filename,
                HumanBytes(size as u64)
            ),
        };
        let _ = self.sender.try_send(msg);

        // 準備給閉包的資料
        let sender_clone = self.sender.clone();
        let stop_clone = self.stop_flag.clone();
        let current = self.current;
        let total = self.current;

        let handle = std::thread::spawn(move || {
            // 這裡不需要複製整個 Progress 實例，因為我們只需要 sender
            while !stop_clone.load(Ordering::Relaxed) {
                // 這裡呼叫一個靜態或輔助函數來使用 sender_clone 發送進度
                let msg = ProgressMessage {
                    current,
                    total,
                    download_done: true,
                    finished: false,
                    message: format!("讀取模型中，已過去{}秒", start.elapsed().as_secs()),
                };
                let _ = sender_clone.try_send(msg);
                std::thread::sleep(Duration::from_secs(1));
            }
        });

        // 儲存 JoinHandle
        self.update_handle = Some(handle);
    }

    fn model_finished(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);

        // 呼叫 join() 阻塞當前線程，等待定時線程退出
        if let Some(handle) = self.update_handle.take() {
            // .take() 取得所有權後，我們就可以 join
            match handle.join() {
                Ok(_) => (),
                Err(e) => log::error!("Update thread panicked: {:?}", e),
            }
        }
        // 發送完全完成訊息
        let msg = ProgressMessage {
            current: self.current,
            total: self.total,
            download_done: true,
            finished: true,
            message: "模型完全初始化完成，正在啟動 Actor。".to_owned(),
        };
        let _ = self.sender.try_send(msg);
    }
}
