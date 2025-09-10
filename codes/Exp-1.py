# import numpy as np
# import matplotlib.pyplot as plt

# def unit_impulse(length, position):
#     signal = np.zeros(length)
#     signal[position] = 1
#     return signal

# # Parameters
# start = -10  # Start value of the x-axis range
# stop = 10  # Stop value of the x-axis range
# step = 1  # Step size

# # Generate x-axis values
# x = np.arange(start, stop+step, step)

# # Generate unit impulse signal
# impulse_signal = unit_impulse(len(x), abs(start)//step)

# # Plot the signal
# plt.stem(x, impulse_signal)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Unit Impulse Signal')
# plt.grid(True)
# plt.show()

#2
import numpy as np
import matplotlib.pyplot as plt

def simulate_impulse_train(signal_length, period):
    impulse_train = np.zeros(signal_length)
    for n in range(signal_length):
        if n % period == 0:
            impulse_train[n] = 1
    return impulse_train

# Define the parameters for the impulse train
signal_length = 100  # Length of the impulse train
period = 10  # Period of the impulse train

# Simulate the impulse train
impulse_train = simulate_impulse_train(signal_length, period)

# Plot and display the impulse train
plt.stem(impulse_train)
plt.title('Impulse Train')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

