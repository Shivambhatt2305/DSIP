import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import correlate

def cross_correlation(signal1, signal2):
    corr = correlate(signal1, signal2, mode='full')
    corr = corr / (np.linalg.norm(signal1) * np.linalg.norm(signal2))  # normalize
    lags = np.arange(-len(signal1)+1, len(signal1))
    return corr, lags

def autocorrelation(signal):
    corr = correlate(signal, signal, mode='full')
    corr = corr / np.max(corr)  # normalize
    lags = np.arange(-len(signal)+1, len(signal))
    return corr, lags

SAMPLE_RATE = 22050   
DURATION = 60         

# ---- Read audio files ----
signal1, sr1 = librosa.load("D:/DSIP/codes/Song_1.mp3", sr=SAMPLE_RATE, mono=True, duration=DURATION)
signal2, sr2 = librosa.load("D:/DSIP/codes/Song_2.mp3", sr=SAMPLE_RATE, mono=True, duration=DURATION)
signal3, sr3 = librosa.load("D:/DSIP/codes/Song_3.mp3", sr=SAMPLE_RATE, mono=True, duration=DURATION)

# ---- Make sure lengths match ----
min_len = min(len(signal1), len(signal2), len(signal3))
signal1, signal2, signal3 = signal1[:min_len], signal2[:min_len], signal3[:min_len]

# ---- Compute correlations ----
auto_1, lags = autocorrelation(signal1)
auto_2, _    = autocorrelation(signal2)
auto_3, _    = autocorrelation(signal3)

cross_12, lags = cross_correlation(signal1, signal2)
cross_13, _    = cross_correlation(signal1, signal3)
cross_23, _    = cross_correlation(signal2, signal3)

# ---- Plot ----
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(lags, auto_1)
plt.title('Autocorrelation (Song 1)')
plt.xlabel('Lag (samples)')

plt.subplot(3, 2, 2)
plt.plot(lags, auto_2)
plt.title('Autocorrelation (Song 2)')
plt.xlabel('Lag (samples)')

plt.subplot(3, 2, 3)
plt.plot(lags, auto_3)
plt.title('Autocorrelation (Song 3)')
plt.xlabel('Lag (samples)')

plt.subplot(3, 2, 4)
plt.plot(lags, cross_12)
plt.title('Cross-correlation (Song 1 & Song 2)')
plt.xlabel('Lag (samples)')

plt.subplot(3, 2, 5)
plt.plot(lags, cross_13)
plt.title('Cross-correlation (Song 1 & Song 3)')
plt.xlabel('Lag (samples)')

plt.subplot(3, 2, 6)
plt.plot(lags, cross_23)
plt.title('Cross-correlation (Song 2 & Song 3)')
plt.xlabel('Lag (samples)')

plt.tight_layout()
plt.show()
