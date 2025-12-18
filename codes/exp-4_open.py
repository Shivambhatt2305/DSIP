import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import butter, bilinear, lfilter, cheby1

# === Filter Design Functions ===
def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
    analog_b, analog_a = butter(filter_order, cutoff_frequency, analog=True, btype='low')
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)
    return digital_b, digital_a

def design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple):
    analog_b, analog_a = cheby1(filter_order, ripple, cutoff_frequency, analog=True, btype='low')
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)
    return digital_b, digital_a

# === Load MP3 using librosa ===
audio, fs = librosa.load("01. Dil Haara Re.mp3", sr=None)  # sr=None keeps original sampling rate
print(f"Audio loaded: {len(audio)} samples, Sampling Rate = {fs} Hz")

# === Filter parameters ===
filter_order = 4
cutoff_frequency = 1000  # Hz
ripple = 0.5

# === Design filters ===
b_butter, a_butter = design_butterworth_filter(filter_order, cutoff_frequency, fs)
b_cheby, a_cheby   = design_chebyshev_filter(filter_order, cutoff_frequency, fs, ripple)

# === Apply filters ===
audio_butter = lfilter(b_butter, a_butter, audio)
audio_cheby  = lfilter(b_cheby, a_cheby, audio)

# === Plot Waveform Comparison ===
plt.figure(figsize=(14, 6))
plt.subplot(3, 1, 1)
plt.plot(audio[:5000], color='gray')
plt.title("Original Audio (Waveform)")
plt.subplot(3, 1, 2)
plt.plot(audio_butter[:5000], color='blue')
plt.title("Butterworth Filtered Audio (Waveform)")
plt.subplot(3, 1, 3)
plt.plot(audio_cheby[:5000], color='red')
plt.title("Chebyshev Filtered Audio (Waveform)")
plt.tight_layout()
plt.show()

# === Plot Spectrogram Comparison ===
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max),
                         sr=fs, x_axis='time', y_axis='log', cmap="magma")
plt.colorbar(format='%+2.0f dB')
plt.title("Original Audio (Spectrogram)")

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_butter)), ref=np.max),
                         sr=fs, x_axis='time', y_axis='log', cmap="magma")
plt.colorbar(format='%+2.0f dB')
plt.title("Butterworth Filtered (Spectrogram)")

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_cheby)), ref=np.max),
                         sr=fs, x_axis='time', y_axis='log', cmap="magma")
plt.colorbar(format='%+2.0f dB')
plt.title("Chebyshev Filtered (Spectrogram)")

plt.tight_layout()
plt.show()
