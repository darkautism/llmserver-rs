use clap::{Arg, ArgAction, Command};
use std::{
    fs,
    net::Ipv4Addr,
    path::{Path, PathBuf},
};

use actix_web::{head, middleware::Logger, web, App, HttpServer, Result};
use llmserver_rs::{
    admin, asr::simple::SimpleASRConfig, db::PgmlRepository, utils::ModelConfig, AIModel,
    ProcessAudio, ProcessMessages, ShutdownMessages,
};
use log::{info, warn};
use actix_web::{head, middleware::Logger, web::Data, App, HttpServer, Result};
use llmserver_rs::{
    admin, audio, chat,
    crypto::{CryptoError, SecretCipher},
    db::Database,
    hf::HuggingFaceDownloader,
    manager::ModelManager,
    utils::ModelConfig,
};
use log::warn;
use utoipa_actix_web::{scope, AppExt};
use utoipa_swagger_ui::SwaggerUi;

fn load_model_configs(
) -> Result<std::collections::HashMap<String, ModelConfig>, Box<dyn std::error::Error>> {
    let dir_path = "assets/config";
    let entries = fs::read_dir(dir_path).map_err(|e| e.to_string())?;

    let mut configs: std::collections::HashMap<String, ModelConfig> =
        std::collections::HashMap::new();

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

#[get("/health")]
async fn health() -> &'static str {
    ""
}

fn decode_hex_key(value: &str, expected_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = hex::decode(value)?;
    if bytes.len() != expected_len {
        return Err(format!("expected {expected_len} bytes but decoded {}", bytes.len()).into());
    }
    Ok(bytes)
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "RUST_LOG",
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
    );
    env_logger::init();

    let matches = Command::new("llmserver")
        .about("Unified management and inference server for text and speech models")
        .version(VERSION)
        .arg(
            Arg::new("model_name")
                .long("model-name")
                .value_name("MODEL_REPO")
                .default_value("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                .help("Model repository identifier as defined in assets/config/*.json"),
            Arg::new("model")
                .short('m')
                .long("model")
                .num_args(1)
                .action(ArgAction::Append)
                .help("Preload a model by its repository identifier"),
        )
        .arg(
            Arg::new("instances")
                .short('i')
                .long("instances")
                .help("Instances to start for each preloaded model")
                .value_parser(clap::value_parser!(usize).range(1..))
                .num_args(1),
        )
        .get_matches();

    let pgml_repository = match PgmlRepository::try_from_env().await {
        Ok(repo) => repo,
        Err(err) => {
            warn!("Failed to initialise PGML repository: {}", err);
            None
        }
    };

    if pgml_repository.is_none() {
        warn!("PGML Admin dashboard disabled: DATABASE_URL is not configured");
    }
    let pgml_repository = web::Data::new(pgml_repository);

    let admin_auth_config = admin::AdminAuthConfig::from_env();
    if admin_auth_config.is_enabled() {
        info!(
            "Admin dashboard authentication enabled via {}",
            admin::ADMIN_API_TOKEN_ENV
        );
    } else {
        warn!(
            "Admin dashboard authentication disabled: set {} to protect admin APIs",
            admin::ADMIN_API_TOKEN_ENV
        );
    }
    let admin_auth_config = web::Data::new(admin_auth_config);

    //初始化模型
    let mut num_instances = 1; // 根據資源設定

    if let Some(value) = matches.get_one::<usize>("instances") {
        num_instances = *value;
    let preload_repos: Vec<String> = matches
        .get_many::<String>("model")
        .map(|vals| vals.cloned().collect())
        .unwrap_or_default();
    let preload_instances = matches.get_one::<usize>("instances").copied().unwrap_or(1);

    let cipher = SecretCipher::from_env().map_err(|err| match err {
        CryptoError::MissingMasterKey => {
            eprintln!(
                "LLMSERVER_MASTER_KEY environment variable is required. You can generate one with: {}",
                SecretCipher::random_master_key()
            );
            err
        }
        _ => err,
    })?;

    let db_path =
        std::env::var("LLMSERVER_DATABASE").unwrap_or_else(|_| "data/llm-admin.db".to_string());
    let database = Database::initialise(Path::new(&db_path), cipher)?;
    if let Some(password) = database.bootstrap_admin().await? {
        println!(
            "[bootstrap] Created default admin account. username=admin password={}. Please change it immediately via the web UI.",
            password
        );
    }
    let model_name = matches
        .get_one::<String>("model_name")
        .expect("model_name should always have a default value");

    // Text type LLM
    let mut llm_recipients = HashMap::<String, Vec<Recipient<ProcessMessages>>>::new();
    let mut audio_recipients = HashMap::<String, Vec<Recipient<ProcessAudio>>>::new();
    let mut shutdown_recipients = Vec::new();

    let model_config_table = load_model_configs()?;
    let manager = ModelManager::new(model_config_table.clone());

    for repo in preload_repos {
        if let Err(err) = manager.start_instances(&repo, preload_instances).await {
            eprintln!("Failed to preload {repo}: {err}");
        }
    }

    if audio_recipients.len() == 0 && llm_recipients.len() == 0 {
        panic!("You do not load any model");
    }
    let cache_dir =
        std::env::var("LLMSERVER_MODEL_CACHE").unwrap_or_else(|_| "data/model-cache".to_string());
    let downloader = HuggingFaceDownloader::new(PathBuf::from(cache_dir))?;

    let database_data = Data::new(database.clone());
    let manager_data = Data::new(manager.clone());
    let downloader_data = Data::new(downloader.clone());

    let bind_address =
        std::env::var("LLMSERVER_BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0:8443".to_string());

    HttpServer::new(move || {
        let pgml_repository = pgml_repository.clone();
        let admin_auth_config = admin_auth_config.clone();
        let (app, api) = App::new()
            .app_data(web::Data::new(llm_recipients.clone()))
            .app_data(web::Data::new(audio_recipients.clone()))
            .app_data(pgml_repository.clone())
            .app_data(admin_auth_config.clone())
            .app_data(database_data.clone())
            .app_data(manager_data.clone())
            .app_data(downloader_data.clone())
            .into_utoipa_app()
            .map(|app| app.wrap(Logger::default()))
            .service(
                scope::scope("/v1")
                    .service(llmserver_rs::chat::chat_completions)
                    .service(llmserver_rs::audio::audio_transcriptions)
                    .service(llmserver_rs::knowledge::karma::karma_enrich),
                    .service(chat::chat_completions)
                    .service(audio::audio_transcriptions)
                    .service(audio::audio_speech),
            )
            .service(
                scope::scope("/admin/api")
                    .service(admin::create_session)
                    .service(admin::list_models)
                    .service(admin::register_model)
                    .service(admin::delete_model),
            )
            .service(web::resource("/admin").route(web::get().to(admin::dashboard)))
            .service(health)
            .configure(admin::configure)
            .split_for_parts();

        app.service(SwaggerUi::new("/swagger-ui/{_:.*}").url("/api-docs/openapi.json", api))
    })
    .bind((Ipv4Addr::UNSPECIFIED, 8443))?
    .run()
    .await?;

    manager.shutdown_all().await;
    Ok(())
}
