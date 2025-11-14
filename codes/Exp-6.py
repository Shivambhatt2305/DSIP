import numpy as np
import matplotlib.pyplot as plt
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# Compute the FFT of the signal
fft_result = np.fft.fft(signal)
# Compute the magnitude and Phase spectrum of the FFT result
magnitude_spectrum = np.abs(fft_result)
Phase_spectrum = np.angle(fft_result)
# Compute the IFFT of the FFT result
reconstructed_signal = np.fft.ifft(fft_result)
