use actix::{Actor, Recipient};
use clap::{Arg, ArgAction, Command};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Read},
    net::Ipv4Addr,
};

use actix_web::{head, middleware::Logger, App, HttpServer, Result};
use llmserver_rs::{
    asr::simple::SimpleASRConfig, utils::ModelConfig, AIModel, ProcessAudio, ProcessMessages,
    ShutdownMessages,
};
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
            println!("Loaded model config: {:?}", path.display());
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
        .arg_required_else_help(true)
        .arg(Arg::new("model_name"))
        .arg(
            Arg::new("instances")
                .short('i')
                .help("How many llm instances do you want to create.")
                .action(ArgAction::Set)
                .num_args(1),
        )
        .get_matches();

    //初始化模型
    let mut num_instances = 1; // 根據資源設定

    if let Some(value) = matches.get_one::<usize>("instances") {
        num_instances = *value;
    }
    let model_name = matches.get_one::<String>("model_name").unwrap();

    // Text type LLM
    let mut llm_recipients = HashMap::<String, Vec<Recipient<ProcessMessages>>>::new();
    let mut audio_recipients = HashMap::<String, Vec<Recipient<ProcessAudio>>>::new();
    let mut shutdown_recipients = Vec::new();

    let model_config_table = load_model_configs()?;

    for _ in 0..num_instances {
        if let Some(config) = model_config_table.get(model_name) {
            if config.model_type == llmserver_rs::utils::ModelType::LLM {
                let llm = llmserver_rs::llm::simple::SimpleRkLLM::init(&config);
                let model_name = config.model_name.clone();

                let addr = llm.unwrap().start(); // 啟動 Actor，一次即可
                if let Some(vec) = llm_recipients.get_mut(&model_name) {
                    vec.push(addr.clone().recipient::<ProcessMessages>());
                } else {
                    llm_recipients.insert(
                        model_name,
                        vec![addr.clone().recipient::<ProcessMessages>()],
                    );
                }
                shutdown_recipients.push(addr.clone().recipient::<ShutdownMessages>());
            } else if config.model_type == llmserver_rs::utils::ModelType::ASR {
                let (llm, modelname) = match (*model_name).as_str() {
                    "happyme531/SenseVoiceSmall-RKNN2" => {
                        let config_path = "assets/config/sensevoicesmall.json";
                        let file = File::open(config_path)
                            .expect(&format!("Config {} not found!", config_path));
                        let mut de = serde_json::Deserializer::from_reader(BufReader::new(file));
                        let config = SimpleASRConfig::deserialize(&mut de)?;
                        (
                            llmserver_rs::asr::simple::SimpleASR::init(&config),
                            config.model_name.clone(),
                        )
                    }
                    _ => {
                        continue;
                    }
                };
                let addr = llm.unwrap().start(); // 啟動 Actor，一次即可
                if let Some(vec) = audio_recipients.get_mut(&modelname) {
                    vec.push(addr.clone().recipient::<ProcessAudio>());
                } else {
                    audio_recipients
                        .insert(modelname, vec![addr.clone().recipient::<ProcessAudio>()]);
                }
                shutdown_recipients.push(addr.clone().recipient::<ShutdownMessages>());
            }
        } else {
            panic!("Model {} not found in the configuration!", model_name);
        }
    }
    
    if audio_recipients.len() == 0 && llm_recipients.len() == 0 {
        panic!("You do not load any model");
    }

    HttpServer::new(move || {
        let (app, api) = App::new()
            .app_data(actix_web::web::Data::new(llm_recipients.clone()))
            .app_data(actix_web::web::Data::new(audio_recipients.clone()))
            .app_data(actix_web::web::Data::new(model_config_table.clone()))
            .into_utoipa_app()
            .map(|app| app.wrap(Logger::default()))
            .service(
                scope::scope("/v1")
                    .service(llmserver_rs::chat::chat_completions)
                    .service(llmserver_rs::openai::models)
                    .service(llmserver_rs::audio::audio_transcriptions),
            )
            .service( // Some Ollama compatible APIs
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
    .bind((Ipv4Addr::UNSPECIFIED, 8080))?
    .run()
    .await?;

    let shutdowns = shutdown_recipients.into_iter().map(|addr| async move {
        let _ = addr.send(ShutdownMessages).await.unwrap();
    });

    tokio::spawn(async {
        futures::future::join_all(shutdowns).await;
    })
    .await?;
    Ok(())
}
