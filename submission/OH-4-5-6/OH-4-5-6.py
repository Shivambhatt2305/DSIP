import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import librosa
import os
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
# Put your 9 audio paths here (1 clean + 8 noisy)
audio_files = [
    r"Original.wav",        # clean reference file
    r"sp30_airport_sn5.wav",
    r"sp30_babble_sn5.wav",
    r"sp30_car_sn5.wav",
    r"sp30_exhibition_sn5.wav",
    r"sp30_restaurant_sn5.wav",
    r"sp30_station_sn5.wav",
    r"sp30_street_sn5.wav",
    r"sp30_train_sn5.wav"
]

# Directories to save results
os.makedirs("plots_fft", exist_ok=True)
os.makedirs("filtered_IIR", exist_ok=True)
os.makedirs("filtered_FIR", exist_ok=True)

# Filter parameters
LOWCUT = 300.0
HIGHCUT = 3400.0
IIR_ORDER = 4
FIR_TAPS = 801

# -----------------------------
# Helper functions
# -----------------------------
def read_audio(path, target_sr=None):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if target_sr and sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def plot_fft(y, sr, filename):
    Y = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1/sr)
    plt.figure(figsize=(8,3))
    plt.semilogy(f, Y + 1e-12)
    plt.title(f"FFT - {filename}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(f"plots_fft/{filename}.png", dpi=150)
    plt.close()

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_iir(y, sr):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, sr, IIR_ORDER)
    y_f = signal.filtfilt(b, a, y)
    return y_f

def apply_fir(y, sr):
    nyq = 0.5 * sr
    taps = signal.firwin(FIR_TAPS, [LOWCUT/nyq, HIGHCUT/nyq], pass_zero=False)
    y_f = signal.lfilter(taps, 1.0, y)
    delay = (FIR_TAPS - 1) // 2
    y_f = np.roll(y_f, -delay)
    y_f[-delay:] = 0
    return y_f

def compute_corr(a, b):
    n = min(len(a), len(b))
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    a /= (np.std(a) + 1e-12)
    b /= (np.std(b) + 1e-12)
    r, _ = pearsonr(a, b)
    return r

# -----------------------------
# Main Execution
# -----------------------------
clean_path = audio_files[0]
clean_y, clean_sr = read_audio(clean_path)
print(f"Clean file: {clean_path} (sr={clean_sr})")

results = []

for i, path in enumerate(audio_files):
    name = os.path.splitext(os.path.basename(path))[0]
    y, sr = read_audio(path, target_sr=clean_sr)
    print(f"Processing {name}...")

    # FFT plot
    plot_fft(y, sr, name)

    # IIR filtering
    y_iir = apply_iir(y, sr)
    sf.write(f"filtered_IIR/IIR_{name}.wav", y_iir, sr)

    # FIR filtering
    y_fir = apply_fir(y, sr)
    sf.write(f"filtered_FIR/FIR_{name}.wav", y_fir, sr)

    # Correlation with clean
    corr_iir = compute_corr(clean_y, y_iir)
    corr_fir = compute_corr(clean_y, y_fir)

    results.append({
        "File": name,
        "IIR_Correlation": corr_iir,
        "FIR_Correlation": corr_fir
    })

# Save results table
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\n Processing complete!")
print("FFT plots → plots_fft/")
print("Filtered IIR files → filtered_IIR/")
print("Filtered FIR files → filtered_FIR/")
print("Correlation results → results.csv")