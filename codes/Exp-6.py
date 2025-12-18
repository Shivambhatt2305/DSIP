import numpy as np
import matplotlib.pyplot as plt

signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Compute the FFT of the signal
fft_result = np.fft.fft(signal)

# Compute the magnitude and phase spectra
magnitude_spectrum = np.abs(fft_result)
phase_spectrum = np.angle(fft_result)

# Compute the IFFT
reconstructed_signal = np.fft.ifft(fft_result)

# ---------- PRINT OUTPUTS ----------
print("Original Signal:")
print(signal)

print("\nFFT Result:")
print(fft_result)

print("\nMagnitude Spectrum:")
print(magnitude_spectrum)

print("\nPhase Spectrum:")
print(phase_spectrum)

print("\nReconstructed Signal (IFFT):")
print(reconstructed_signal.real)   # Take real part only

# ---------- PLOT THE RESULTS ----------
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.stem(signal)
plt.title("Original Signal")

plt.subplot(2, 2, 2)
plt.stem(magnitude_spectrum)
plt.title("Magnitude Spectrum")

plt.subplot(2, 2, 3)
plt.stem(phase_spectrum)
plt.title("Phase Spectrum")

plt.subplot(2, 2, 4)
plt.stem(reconstructed_signal.real)
plt.title("Reconstructed Signal (IFFT)")

plt.tight_layout()
plt.show()
