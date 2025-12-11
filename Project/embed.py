# embed.py
"""
This module provides:
 - get_text_embedding(prompt) -> torch tensor or numpy array
 - get_audio_embedding(wav_numpy, sr) -> same dim embedding

It tries to load a CLAP-style model (joint audio-text). If unavailable,
it falls back to SentenceTransformer for text and a simple MFCC-based
audio embedding for distance.
"""
import os
import numpy as np
import torch
import librosa
from tqdm import tqdm

# Try CLAP (LAION-CLAP). If unavailable, fallback.
CLAP_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModel
    # model name might differ; change to one you have access to.
    CLAP_MODEL = "laion/clap-htsat-unfused"  # try this first (paper used LAION-CLAP)
    proc = AutoProcessor.from_pretrained(CLAP_MODEL)
    clap_model = AutoModel.from_pretrained(CLAP_MODEL)
    clap_model.eval()
    CLAP_AVAILABLE = True
    print("[embed] CLAP model loaded.")
except Exception as e:
    CLAP_AVAILABLE = False
    print("[embed] CLAP model unavailable, fallback will be used:", str(e))

if not CLAP_AVAILABLE:
    from sentence_transformers import SentenceTransformer
    text_model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight text encoder
    print("[embed] SentenceTransformer loaded as fallback for text.")

def get_text_embedding(prompt):
    if CLAP_AVAILABLE:
        # Use processor for text
        inputs = proc(text=prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = clap_model.get_text_features(**inputs)  # may vary by model API
        emb = out[0].cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    else:
        emb = text_model.encode([prompt], normalize_embeddings=True)
        return emb[0]

def get_audio_embedding_from_array(wav_np, sr):
    """
    If CLAP available, pass audio through model. Otherwise compute enhanced audio features.
    wav_np: 1D numpy float32
    """
    if CLAP_AVAILABLE:
        # Resample if needed (CLAP typically expects 48kHz)
        inputs = proc(audios=wav_np, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = clap_model.get_audio_features(**inputs)
        emb = out[0].cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    else:
        # Enhanced audio features: MFCC + spectral features
        mf = librosa.feature.mfcc(y=wav_np.astype(float), sr=sr, n_mfcc=40)
        spectral_centroid = librosa.feature.spectral_centroid(y=wav_np.astype(float), sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=wav_np.astype(float), sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=wav_np.astype(float), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(wav_np.astype(float))
        
        feat = np.concatenate([
            mf.mean(axis=1), 
            mf.std(axis=1),
            spectral_centroid.mean(axis=1),
            spectral_rolloff.mean(axis=1),
            spectral_contrast.mean(axis=1),
            zcr.mean(axis=1)
        ])
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        return feat
