# Natural Sampling

## Aim  
To perform natural sampling on a continuous signal and analyze its reconstruction.  

## Tools Required  
- Python (colab)
- NumPy  
- Matplotlib  
- SciPy  

## Program  

### Natural Sampling  
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Define parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector

# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# Pulse Train Parameters
pulse_rate = 50  # Pulses per second
pulse_train = np.zeros_like(t)

# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

# Natural Sampling
nat_signal = message_signal * pulse_train

# Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]

# Create a time vector for the sampled points
sample_times = t[pulse_train == 1]

# Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]

# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

# Plot Results
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Pulse Train')
plt.legend()
plt.grid(True)

# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Natural Sampling')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Output Waveforms  
- **Original Message Signal:** The original sine wave before sampling.
  
  ![image](https://github.com/user-attachments/assets/e5ae651b-217a-45e5-9c52-59506977e519)

- **Pulse Train:** The rectangular pulses used for natural sampling.
  
  ![image](https://github.com/user-attachments/assets/5b2f7466-23cc-49ac-aa5a-a74223ba1b85)

- **Natural Sampling:** The sampled signal obtained by multiplying with the pulse train.
  
  ![image](https://github.com/user-attachments/assets/bd505eb9-1cfe-4a5c-9a4e-23488f640155)
  
- **Reconstructed Signal:** The signal after passing through a low-pass filter for smoothing.
   
  ![image](https://github.com/user-attachments/assets/7d4be66d-5d17-42d5-9714-2f5f26410fd2)

## Results  
- The continuous message signal was successfully sampled using natural sampling.   
