from scipy.signal import hilbert
import numpy as np


# Define physical constants
c = 2.99792e8 # m/s
eps0 = 8.85419e-12 # F/m
Z0 = 376.730 # Ohm

import numpy as np
from scipy.optimize import curve_fit

# FFT Function
def calculate_fft(signal, time):
    """
    Compute the FFT of a time-domain signal and its corresponding frequency array.

    Parameters:
    - signal (array): Time-domain signal.
    - time (array): Time array (in seconds).

    Returns:
    - fft (array): FFT of the signal.
    - freqs (array): Corresponding frequency array (in Hz).
    """
    dt = time[1] - time[0]  # Time step in seconds
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=dt)  # Frequency in Hz
    # Use only positive frequencies
    positive_idx = np.where(freqs > 0)
    freqs = freqs[positive_idx]
    fft = fft[positive_idx]
    return fft, freqs

# Assume samp_fft and ref_fft are 1D arrays representing Fourier-transformed signals
# freq is the corresponding frequency array

def phase_and_amplitude(freq, samp_fft, ref_fft, samp_td, ref_td, freqs_for_fit):
    """
    Calculate phase and amplitude using Jepsen's method for 1D FFT arrays.

    Parameters:
    - freq (array): Frequency array (in Hz).
    - samp_fft (array): FFT of the sample signal (complex).
    - ref_fft (array): FFT of the reference signal (complex).
    - samp_td (tuple): Time-domain sample signal as (time array, amplitude array).
    - ref_td (tuple): Time-domain reference signal as (time array, amplitude array).
    - freqs_for_fit (tuple): Frequency range for phase fitting (start, end in Hz).

    Returns:
    - phase (array): Corrected phase.
    - T_amp (array): Amplitude transmission.
    """

    # Calculate amplitude transmission
    T_amp = np.abs(samp_fft) / np.abs(ref_fft)  # Amplitude transmission ratio

    # Find the time delays (time of the maximum signal in the time domain)
    t0_ref = ref_td[0]  # Reference time delay
    t0_samp = samp_td[0]  # Sample time delay
    delta_t = t0_samp - t0_ref  # Time delay difference

    # Calculate phase offset
    phase_offset = 2 * np.pi * delta_t * freq

    # Initial phases for reference and sample
    phi0_ref = 2 * np.pi * freq * t0_ref
    phi0_samp = 2 * np.pi * freq * t0_samp

    # Calculate reduced unwrapped phase difference
    dphi0_reduced = np.unwrap(
        np.angle(np.exp(1j * (np.angle(samp_fft) - phi0_samp)))
        - np.angle(np.exp(1j * (np.angle(ref_fft) - phi0_ref)))
    )

    # Apply a mask to select frequencies within the fitting range
    mask = (freq > freqs_for_fit[0]) & (freq < freqs_for_fit[1])
    print(freq)
    # Fit the phase to a linear function
    pars, _ = curve_fit(lambda x, a, b: a * x + b, freq[mask], dphi0_reduced[mask])

    # Correct the phase
    phase0 = dphi0_reduced - 2 * np.pi * round(pars[1] / (2 * np.pi))
    phase = phase0 - phi0_ref + phi0_samp + phase_offset

    return phase, T_amp
def phase_and_amplitude_simple(samp_fft, ref_fft):
    """
    Calculate phase and amplitude using the standard way

    Parameters:
    - samp_fft (array): FFT of the sample signal (complex).
    - ref_fft (array): FFT of the reference signal (complex).

    Returns:
    - phase (array): Corrected phase.
    - T_amp (array): Amplitude transmission.
    """

    # Transmission function
    T = samp_fft / ref_fft  # Complex transmission function

    # Amplitude A(omega)
    T_amp = np.abs(T)  # This is the amplitude

    # (Optional) Phase φ(omega)
    phase  = np.angle(T)  # This is the phase in radians
    return phase, T_amp


def absorption_coefficient(phase, T_amp, freqs):
    """
    Calculate the absorption coefficient alpha(omega) from phase and amplitude transmission.

    Parameters:
    - phase (array): Phase difference (radians) between the reference and sample signals.
    - T_amp (array): Amplitude transmission ratio |T(ω)|.
    - freqs (array): Frequency array (in Hz).
    - sample_thickness (float): Thickness of the sample (in meters).

    Returns:
    - alpha (array): Absorption coefficient (in m^-1).
    """
    # Speed of light
    c = 3e8  # m/s

    # Angular frequency
    omega = 2 * np.pi * freqs

    # Calculate absorption coefficient (standard form)
    numerator = 2 * omega * T_amp * np.sin(phase)
    denominator = c * (1 + T_amp**2 - 2 * T_amp * np.cos(phase))
    alpha = numerator / denominator


    return alpha

def approx_absorption_coefficient(phase, T_amp, freqs, sample_thickness):
    """
    Calculate the absorption coefficient using the approximated Jepsen method: Assuming that the 
    largest absorption that can be measured is the one of the reference metallic mirror and that 
    the sample and reference reflection have the same phase shift, 
    the approximated form of the absorption coefficient proposed in Jepsen and Fischer23 can be used: 

    Parameters:
    - phase (array): Phase difference (radians) between the reference and sample signals.
    - T_amp (array): Amplitude transmission ratio |T(ω)|.
    - freqs (array): Frequency array (in Hz).
    - sample_thickness (float): Thickness of the sample (in meters).

    Returns:
    - alpha (array): Approximated Absorption coefficient (in m^-1).
    """
    # Speed of light
    c = 3e8  # m/s

    # Calculate the approximated absorption coefficient alpha(omega)
    omega = freqs*2*np.pi
    alpha_approx = (2 * omega / c) * ((2 * phase) / ((1 - T_amp)**2 + phase**2))
    return alpha_approx

import numpy as np


def refractive_index_from_reflection(phase, T_amp, freqs, absorption=None):
    """
    Calculate the refractive index n(omega) from THz-TDS reflection data.

    Parameters:
    - sample_fft (array): FFT of the reflected signal from the sample.
    - reference_fft (array): FFT of the reflected signal from the reference (e.g., mirror).
    - freqs (array): Frequency array (in Hz).
    - absorption (array, optional): Absorption coefficient (alpha in m^-1), if available.

    Returns:
    - n (array): Real part of the refractive index.
    - kappa (array): Extinction coefficient (if absorption is provided).
    """
    # Speed of light
    c = 3e8  # m/s

    # Angular frequency
    omega = 2 * np.pi * freqs

    # Calculate absorption coefficient (standard form)
    numerator = (1- T_amp**2)
    denominator = c * (1 + T_amp**2 - 2 * T_amp * np.cos(phase))
    n = numerator / denominator

    return n, None