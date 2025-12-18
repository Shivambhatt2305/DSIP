import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load WAV file
sample_rate, audio = wavfile.read("bees.wav")

# If stereo, convert to mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Create time axis
time = np.linspace(0, len(audio) / sample_rate, num=len(audio))

# Plot waveform
plt.figure(figsize=(10, 4))
plt.plot(time, audio, linewidth=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform of Audio Signal")
plt.tight_layout()
plt.show()
