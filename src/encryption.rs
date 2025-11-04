use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use derive_more::Display;
use rand::RngCore;

#[derive(Debug, Display)]
pub enum EncryptionError {
    #[display("invalid encryption key length")]
    InvalidKeyLength,
    #[display("failed to decode key: {0}")]
    Decode(String),
    #[display("cipher error")]
    CipherError,
    #[display("ciphertext too short")]
    CipherTextTooShort,
}

impl std::error::Error for EncryptionError {}

pub struct EncryptionService {
    cipher: Aes256Gcm,
}

impl EncryptionService {
    pub fn new_from_hex(hex_key: &str) -> Result<Self, EncryptionError> {
        let bytes = hex::decode(hex_key).map_err(|e| EncryptionError::Decode(e.to_string()))?;
        if bytes.len() != 32 {
            return Err(EncryptionError::InvalidKeyLength);
        }
        let cipher = Aes256Gcm::new_from_slice(&bytes).map_err(|_| EncryptionError::CipherError)?;
        Ok(Self { cipher })
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let mut ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| EncryptionError::CipherError)?;
        let mut payload = Vec::with_capacity(nonce_bytes.len() + ciphertext.len());
        payload.extend_from_slice(&nonce_bytes);
        payload.append(&mut ciphertext);
        Ok(payload)
    }

    pub fn decrypt(&self, payload: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        if payload.len() <= 12 {
            return Err(EncryptionError::CipherTextTooShort);
        }
        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| EncryptionError::CipherError)
    }
}
