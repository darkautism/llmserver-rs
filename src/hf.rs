use std::{fs, path::PathBuf};

use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloaderError {
    #[error("hugging face token is required for repository downloads")]
    MissingToken,
    #[error("huggingface hub error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone)]
pub struct HuggingFaceDownloader {
    cache_dir: PathBuf,
}

impl HuggingFaceDownloader {
    pub fn new(cache_dir: PathBuf) -> Result<Self, DownloaderError> {
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }
        Ok(Self { cache_dir })
    }

    pub fn download_repo(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        token: Option<&str>,
    ) -> Result<PathBuf, DownloaderError> {
        let token = token.ok_or(DownloaderError::MissingToken)?;
        let api = ApiBuilder::new()
            .with_token(Some(token.to_owned()))
            .with_cache_dir(self.cache_dir.clone())
            .build()?;
        let repo = Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        );
        let snapshot = api.repo(repo).snapshot(Default::default())?;
        Ok(snapshot)
    }
}
