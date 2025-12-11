# utils.py
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa.display

def save_wav(path, audio, sr=48000):
    sf.write(path, audio.astype(np.float32), sr)
    print(f"Saved WAV: {path}")

def plot_spectrogram(audio, sr=48000, outpath=None, title="Spectrogram"):
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"Saved spectrogram: {outpath}")
    else:
        plt.show()
