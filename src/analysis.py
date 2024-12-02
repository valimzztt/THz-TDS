import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from utils import phase_and_amplitude, phase_and_amplitude_simple, absorption_coefficient, approx_absorption_coefficient
from utils import refractive_index_from_reflection
# Constants
c = 3e8  # Speed of light in vacuum (m/s)
epsilon_0 = 8.854e-12  # Permittivity of free space (F/m)
# Relative data folder
data_folder = os.path.join(os.getcwd(),"processed_dataDec1")  # Update this if needed based on your project structure

# File paths
reference_file = os.path.join(data_folder, "Processed_Reference.d24")
sample_file = os.path.join(data_folder, "Processed_Sample.d24")

# Check if files exist
if not os.path.exists(reference_file):
    raise FileNotFoundError(f"Reference.d24 not found in: {reference_file}")
if not os.path.exists(sample_file):
    raise FileNotFoundError(f"Sample.d24 not found in: {sample_file}")


# Function to read and process .d24 files
def read_d24_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    start_line = next(i for i, line in enumerate(lines) if line.lstrip()[0].isdigit())
    data_lines = lines[start_line:]
    return pd.DataFrame(
        [list(map(float, line.split())) for line in data_lines],
        columns=["Time (ps)", "Signal"]
    )[["Time (ps)", "Signal"]]

# Load data
reference_data = read_d24_file(reference_file)
sample_data = read_d24_file(sample_file)

# Convert time to picoseconds
reference_signal = reference_data["Signal"] 
sample_signal = sample_data["Signal"] 
time = sample_data["Time (ps)"]
dt = sample_data["Time (ps)"].iloc[1]-sample_data["Time (ps)"].iloc[0]
# Fourier Transform to convert time domain to frequency domain
ref_fft = np.fft.fft(reference_signal)
sample_fft = np.fft.fft(sample_signal)
freqs = np.fft.fftfreq(len(time), d=dt)  # Frequency in Hz

# Use only positive frequencies
positive_idx = np.where(freqs > 0)
freqs = freqs[positive_idx]
ref_fft = ref_fft[positive_idx]
sample_fft = sample_fft[positive_idx]
freqs_for_fit =  (0.1, 10.0) 
sample_thickness = 1e-3
phase, R_amp = phase_and_amplitude(freqs, sample_fft, ref_fft,sample_signal , reference_signal, freqs_for_fit)
absorption = absorption_coefficient(phase, R_amp, freqs)


plt.figure(figsize=(12, 8))

# Phase Plot
plt.subplot(2, 1, 1)
plt.plot(freqs, phase, label="Phase (radians)", color='blue')
plt.title("Corrected Phase")
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (radians)")
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(freqs, absorption, label="Absorption", color='blue')
plt.xlim(0, 20)
plt.ylim(-0.25e-5, 0.25e-5)
plt.title("Absorption")
plt.xlabel("Frequency (THz)")
plt.ylabel("Dielectric Function (ε)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate refractive index
n, kappa = refractive_index_from_reflection(phase, R_amp, freqs, absorption)

plt.figure(figsize=(10, 6))
plt.plot(freqs, n, label="Refractive Index (n)", color="blue")
if kappa is not None:
    plt.plot(freqs * 1e-12, kappa, label="Extinction Coefficient (kappa)", color="orange")
plt.title("Refractive Index and Extinction Coefficient")
plt.xlabel("Frequency (THz)")
plt.ylabel("Value")
plt.legend()
plt.xlim(0, 10)
plt.grid(True)
plt.show()