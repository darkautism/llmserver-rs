use chacha20poly1305::{aead::Aead, aead::KeyInit, ChaCha20Poly1305, Key, Nonce};
use rand_core::{OsRng, RngCore};
use std::{env, fmt, sync::Arc};
use thiserror::Error;

#[derive(Clone)]
pub struct SecretCipher {
    key: Arc<Key>,
}

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("environment variable LLMSERVER_MASTER_KEY is missing")]
    MissingMasterKey,
    #[error("invalid master key length. expected 32 bytes after base64 decoding")]
    InvalidKeyLength,
    #[error("base64 decoding error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("encryption failure")]
    Encrypt,
    #[error("decryption failure")]
    Decrypt,
}

impl SecretCipher {
    pub fn from_env() -> Result<Self, CryptoError> {
        let key_b64 =
            env::var("LLMSERVER_MASTER_KEY").map_err(|_| CryptoError::MissingMasterKey)?;
        let key_bytes = base64::decode(key_b64.trim())?;
        if key_bytes.len() != 32 {
            return Err(CryptoError::InvalidKeyLength);
        }
        let mut key = Key::default();
        key.copy_from_slice(&key_bytes);
        Ok(Self { key: Arc::new(key) })
    }

    pub fn random_master_key() -> String {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        base64::encode(key)
    }

    pub fn encrypt(&self, plain: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let cipher = ChaCha20Poly1305::new(self.key.as_ref());
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let mut ciphertext = cipher
            .encrypt(nonce, plain)
            .map_err(|_| CryptoError::Encrypt)?;
        let mut payload = nonce_bytes.to_vec();
        payload.append(&mut ciphertext);
        Ok(payload)
    }

    pub fn decrypt(&self, payload: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if payload.len() < 12 {
            return Err(CryptoError::Decrypt);
        }
        let (nonce_bytes, ciphertext) = payload.split_at(12);
        let cipher = ChaCha20Poly1305::new(self.key.as_ref());
        let nonce = Nonce::from_slice(nonce_bytes);
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::Decrypt)
    }
}

impl fmt::Debug for SecretCipher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecretCipher").finish_non_exhaustive()
    }
}
