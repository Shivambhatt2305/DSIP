import numpy as np
import matplotlib.pyplot as plt
def cross_correlation(signal1, signal2):
    # Compute the cross-correlation
    cross_corr = np.correlate(signal1, signal2, mode='full')
    return cross_corr
def autocorrelation(signal):
    # Compute the autocorrelation
    auto_corr = np.correlate(signal, signal, mode='full')
    return auto_corr
# Define the discrete-time signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])
# Compute the cross-correlation
cross_corr = cross_correlation(signal1, signal2) 
# Compute the autocorrelation
auto_corr = autocorrelation(signal1) 
# Plot the cross-correlation and autocorrelation signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(cross_corr)
plt.title('Cross-correlation')
plt.xlabel('Time Lag')
plt.ylabel('Magnitude') 
plt.subplot(2, 1, 2)
plt.stem(auto_corr)
plt.title('Autocorrelation')
plt.xlabel('Time Lag')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()
