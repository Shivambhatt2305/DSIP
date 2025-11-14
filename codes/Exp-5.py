import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Filter parameters
cutoff_frequency = 0.2  # Normalized cutoff frequency (0.0 to 0.5)
filter_length = 31      # Number of filter taps (odd for symmetry)

# Design the FIR filter using Hanning window method
filter_coefficients = firwin(filter_length, cutoff=cutoff_frequency, window='hann')

# Plot the impulse response of the filter
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.stem(filter_coefficients)
plt.title("Impulse Response")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Plot the frequency response of the filter
plt.subplot(2, 1, 2)
frequencies, response = freqz(filter_coefficients)
plt.plot(frequencies, 20 * np.log10(np.abs(response)))
plt.title("Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()
