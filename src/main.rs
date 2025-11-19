use actix::{Actor, Recipient};
use clap::{Arg, Command};
use log::info;
use std::{
    collections::HashMap,
    fs,
    io::Read,
    net::Ipv4Addr,
    sync::{Arc, Mutex},
    time::Duration,
};

use actix_web::{head, middleware::Logger, App, HttpServer, Result};
use llmserver_rs::{utils::ModelConfig, AIModel, ProcessAudio, ProcessMessages, ShutdownMessages};
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

fn load_model_configs() -> Result<HashMap<String, ModelConfig>, Box<dyn std::error::Error>> {
    let dir_path = "assets/config";
    let entries = fs::read_dir(dir_path).map_err(|e| e.to_string())?;

    let mut configs: HashMap<String, ModelConfig> = HashMap::new();

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
            let mut file = fs::File::open(&path).map_err(|e| e.to_string())?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|e| e.to_string())?;

            let mut config: ModelConfig =
                serde_json::from_str(&contents).map_err(|e| e.to_string())?;
            info!("Loaded model config: {:?}", path.display());
            config._asserts_path = path.to_string_lossy().to_string();
            configs.insert(config.model_repo.clone(), config);
        }
    }

    Ok(configs)
}

/// Get health of the API.
#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = str, content_type = "text/plain")
    )
)]
#[head("/health")]
async fn health() -> &'static str {
    ""
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    const VERSION: &str = env!("CARGO_PKG_VERSION");
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let matches = Command::new("rkllm")
        .about("Stupid webserver ever!")
        .version(VERSION)
        .arg(Arg::new("model_name"))
        .get_matches();

    //初始化模型
    let model_name_opt = matches.get_one::<String>("model_name");

    // Text type LLM
    let llm_recipients = Arc::new(Mutex::new(
        HashMap::<String, Recipient<ProcessMessages>>::new(),
    ));
    let audio_recipients = Arc::new(Mutex::new(HashMap::<String, Recipient<ProcessAudio>>::new()));
    let shutdown_recipients = Arc::new(Mutex::new(
        HashMap::<String, Recipient<ShutdownMessages>>::new(),
    ));

    let model_config_table = load_model_configs()?;

    if let Some(model_name) = model_name_opt {
        if let Some(config) = model_config_table.get(model_name) {
            if config.model_type == llmserver_rs::utils::ModelType::LLM {
                let llm = llmserver_rs::llm::simple::SimpleRkLLM::init(&config);
                let model_name = config.model_name.clone();

                let addr = llm.unwrap().start(); // 啟動 Actor，一次即可
                llm_recipients.lock().unwrap().insert(
                    model_name.clone(),
                    addr.clone().recipient::<ProcessMessages>(),
                );
                shutdown_recipients
                    .clone()
                    .lock()
                    .unwrap()
                    .insert(model_name, addr.clone().recipient::<ShutdownMessages>());
            } else if config.model_type == llmserver_rs::utils::ModelType::ASR {
                // let (llm, model_name) = match (*model_name).as_str() {
                //     "happyme531/SenseVoiceSmall-RKNN2" => {
                //         let config_path = "assets/config/sensevoicesmall.json";
                //         let file = File::open(config_path)
                //             .expect(&format!("Config {} not found!", config_path));
                //         let mut de = serde_json::Deserializer::from_reader(BufReader::new(file));
                //         let config = SimpleASRConfig::deserialize(&mut de)?;
                //         (
                //             llmserver_rs::asr::simple::SimpleASR::init(&config),
                //             config.model_name.clone(),
                //         )
                //     }
                //     _ => {}
                // };
                // let addr = llm.unwrap().start(); // 啟動 Actor，一次即可
                // audio_recipients.insert(model_name, vec![addr.clone().recipient::<ProcessAudio>()]);

                // shutdown_recipients
                //     .lock()
                //     .unwrap()
                //     .insert(model_name, addr.clone().recipient::<ShutdownMessages>());
            }
        } else {
            panic!("Model {} not found in the configuration!", model_name);
        }
    }

    let shutdown_recipients_cloned = shutdown_recipients.clone();
    HttpServer::new(move || {
        let shutdown_for_data = shutdown_recipients_cloned.clone();
        let (app, api) = App::new()
            .app_data(actix_web::web::Data::new(llm_recipients.clone()))
            .app_data(actix_web::web::Data::new(audio_recipients.clone()))
            .app_data(actix_web::web::Data::new(model_config_table.clone()))
            .app_data(actix_web::web::Data::new(shutdown_for_data))
            .into_utoipa_app()
            .map(|app| app.wrap(Logger::default()))
            .service(
                scope::scope("/v1")
                    .service(llmserver_rs::chat::chat_completions)
                    .service(llmserver_rs::openai::models)
                    .service(llmserver_rs::audio::audio_transcriptions),
            )
            .service(
                // Some Ollama compatible APIs
                scope::scope("/api/")
                    .service(llmserver_rs::ollama::version)
                    .service(llmserver_rs::ollama::push)
                    .service(llmserver_rs::ollama::pull)
                    .service(llmserver_rs::ollama::ps),
            )
            .service(health)
            .split_for_parts();

        app.service(SwaggerUi::new("/swagger-ui/{_:.*}").url("/api-docs/openapi.json", api))
    })
    .keep_alive(Some(Duration::from_secs(1800)))
    .client_request_timeout(Duration::from_secs(1800))
    .client_disconnect_timeout(Duration::from_secs(1800))
    .bind((Ipv4Addr::UNSPECIFIED, 8080))?
    .run()
    .await?;

    let shutdowns = {
        let shutdown_arc_clone = shutdown_recipients.clone();
        let mut shutdown_pool_lock = shutdown_arc_clone.lock().unwrap();
        shutdown_pool_lock
            .drain()
            .map(|(_, addr)| async move {
                let _ = addr.send(ShutdownMessages).await.unwrap();
            })
            .collect::<Vec<_>>()
    };

    tokio::spawn(async {
        futures::future::join_all(shutdowns).await;
    })
    .await?;
    Ok(())
}
