use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use actix::Recipient;
use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use actix_web::{post, HttpResponse, Responder};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{OpenAiError, ProcessAudio};

#[derive(Deserialize, Serialize, utoipa::ToSchema)]
pub struct TranscriptionsResponse {
    pub text: String,
}

#[derive(Debug, MultipartForm)]
struct UploadForm {
    model: Text<String>,
    file: TempFile,
}

#[utoipa::path(
    responses(
        (status = OK, description = "Success", body = TranscriptionsResponse, content_type = "application/json")
    ),
    security(
        ("api_key" = [])
    ),
)]
#[post("/audio/transcriptions")]
pub async fn audio_transcriptions(
    form: MultipartForm<UploadForm>,
    asr_pool: actix_web::web::Data<Arc<Mutex<HashMap<String, Recipient<ProcessAudio>>>>>,
) -> impl Responder {
    log::info!("{:?}", form.file);
    log::info!("{:?}", form.model);

    let Ok(asr_pool_locked) = asr_pool.try_lock() else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!(
                "There is another instance running, please wait other instance finished."
            ),
            code: "busy".to_owned(),
            r#type: "busy".to_owned(),
            param: None,
        });
    };

    let Some(asr) = asr_pool_locked.get(&form.model.0) else {
        return HttpResponse::BadRequest().json(OpenAiError {
            message: format!(
                "The model {} does not exist or you do not have access to it.",
                form.model.0
            ),
            code: "model_not_found".to_owned(),
            r#type: "invalid_request_error".to_owned(),
            param: None,
        });
    };

    let path = form.file.file.as_ref().to_string_lossy().to_string();
    let send_future = asr.send(ProcessAudio::FilePath(path));

    match actix_web::rt::time::timeout(std::time::Duration::from_secs(5), send_future).await {
        Ok(Ok(Ok(receiver))) => {
            let sse_stream = receiver.map(move |content| match content {
                crate::AsrText::SenseVoice(voice_text) => voice_text.content,
            });

            let transcription_parts: Vec<String> = sse_stream.collect().await;
            let full_transcription = transcription_parts.join("");
            HttpResponse::Ok().json(json!({ "text": full_transcription }))
        }
        Ok(Ok(Err(e))) => HttpResponse::InternalServerError().json(OpenAiError {
            message: format!("Internal processing error: {:?}", e),
            code: "processing_error".to_owned(),
            r#type: "internal_error".to_owned(),
            param: None,
        }),
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
    }
}
