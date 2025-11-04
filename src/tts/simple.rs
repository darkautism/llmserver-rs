use std::io::Cursor;

use actix::Actor;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;

use crate::{AIModel, ProcessTts, ShutdownMessages, TTS};

#[derive(Debug, Clone, Deserialize)]
pub struct SimpleTtsConfig {
    pub model_name: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_character_duration_ms")]
    pub character_duration_ms: u32,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

fn default_sample_rate() -> u32 {
    16_000
}

fn default_character_duration_ms() -> u32 {
    180
}

fn default_amplitude() -> f32 {
    0.35
}

pub struct SimpleToneTts {
    config: SimpleTtsConfig,
}

impl Actor for SimpleToneTts {
    type Context = actix::Context<Self>;
}

impl actix::Handler<ProcessTts> for SimpleToneTts {
    type Result = Result<Vec<u8>, ()>;

    fn handle(&mut self, msg: ProcessTts, _ctx: &mut Self::Context) -> Self::Result {
        let mut cursor = Cursor::new(Vec::new());
        let spec = WavSpec {
            channels: 1,
            sample_rate: self.config.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::new(&mut cursor, spec).map_err(|_| ())?;
        let samples_per_char = (self.config.sample_rate as f32
            * (self.config.character_duration_ms as f32 / 1000.0))
            as usize;
        let amplitude = (i16::MAX as f32 * self.config.amplitude).clamp(0.0, i16::MAX as f32);

        for (idx, ch) in msg.text.chars().enumerate() {
            if ch.is_whitespace() {
                for _ in 0..samples_per_char {
                    writer.write_sample(0i16).map_err(|_| ())?;
                }
                continue;
            }
            let frequency = tone_for_character(ch, idx);
            for n in 0..samples_per_char {
                let t = n as f32 / self.config.sample_rate as f32;
                let value = (2.0 * std::f32::consts::PI * frequency * t).sin();
                let sample = (value * amplitude) as i16;
                writer.write_sample(sample).map_err(|_| ())?;
            }
        }
        writer.finalize().map_err(|_| ())?;
        Ok(cursor.into_inner())
    }
}

impl actix::Handler<ShutdownMessages> for SimpleToneTts {
    type Result = Result<(), ()>;

    fn handle(&mut self, _msg: ShutdownMessages, _ctx: &mut Self::Context) -> Self::Result {
        Ok(())
    }
}

impl AIModel for SimpleToneTts {
    type Config = SimpleTtsConfig;

    fn init(config: &Self::Config) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl TTS for SimpleToneTts {}

fn tone_for_character(ch: char, index: usize) -> f32 {
    let base = match ch.to_ascii_lowercase() {
        'a' | 'à' | 'á' | 'â' => 440.0,
        'b' => 466.16,
        'c' => 493.88,
        'd' => 523.25,
        'e' | 'è' | 'é' => 554.37,
        'f' => 587.33,
        'g' => 622.25,
        'h' => 659.25,
        'i' => 698.46,
        'j' => 739.99,
        'k' => 783.99,
        'l' => 830.61,
        'm' => 880.0,
        'n' => 932.33,
        'o' => 987.77,
        'p' => 1046.5,
        'q' => 1108.73,
        'r' => 1174.66,
        's' => 1244.51,
        't' => 1318.51,
        'u' | 'ü' => 1396.91,
        'v' => 1479.98,
        'w' => 1567.98,
        'x' => 1661.22,
        'y' => 1760.0,
        'z' => 1864.66,
        _ => 392.0,
    };
    let modulation = (index % 5) as f32 * 12.0;
    base + modulation
}
