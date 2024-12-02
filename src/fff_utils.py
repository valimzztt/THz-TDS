import numpy as np          # type: ignore
from scipy.fft import rfft, rfftfreq        # type: ignore
from scipy.signal.windows import get_window         # type: ignore
import pywt         # type: ignore
import pandas as pd         # type: ignore

from .misc import *

def pad_to_power2(data, power_of_2=14):
    """
    Pads the data array with zeros so the number of data points is a power of 2.

    Parameters:
    data (ndarray): Array of data to be padded.
    power_of_2 (int): Power of 2 to pad the data to.

    Returns:
    ndarray: Padded data array.
    """
    N0 = len(data)
    N_pad = int((2**power_of_2 - N0) / 2)

    pad_data = np.append(np.zeros(N_pad), np.append(data, np.zeros(N_pad)))

    return pad_data

def do_fft(data, window='hann', min_time=-np.inf, max_time=np.inf, pad_power2=1):
    """
    Performs FFT on the data array.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and corresponding values in data[1].
    window (str): Window function to apply to the data.
    min_time (float): Minimum time for FFT.
    max_time (float): Maximum time for FFT.
    pad_power2 (int): Power of 2 to pad the data to. If number is smaller than the current length, pads until the next power of 2, 
                        which is the default setting.

    Returns:
    ndarray: 2D array with frequency, FFT amplitude, and FFT phase.
    """
    mask = (data[0] > min_time) & (data[0] < max_time)
    t = data[0][mask]
    E = data[1][mask]
    dt = abs(t[1] - t[0])
    
    peak_ind = np.argmax(E)
    N_right = len(E) - peak_ind
    N_pad = N_right - peak_ind

    # Pads to the left or right to make the window centered on the peak
    if N_pad > 0:
        new_E = np.append(np.zeros(N_pad), E)   
    else:
        new_E = np.append(E, np.zeros(np.abs(N_pad)))

    w = get_window(window, len(new_E), fftbins=False)
    
    if 2**pad_power2 < len(new_E):
        pad_power2 = int(np.log2(len(new_E))) + 1

    # Pads
    E = pad_to_power2(new_E, power_of_2=pad_power2)
    w = pad_to_power2(w, power_of_2=pad_power2)

    N = len(E)

    fft_result = rfft(E * w)
    fft_freq = rfftfreq(N, dt)

    fft_amp = np.abs(fft_result)

    fft_phase = -np.angle(fft_result)  # We unwrap and add the time delayed phase later. See Jepsen https://doi.org/10.1063/1.5047659 for details
    

    return np.array([fft_freq, fft_amp, fft_phase])


def do_fft_2d(data_df, window='Hann'):   
    """
    Applies FFT (Fast Fourier Transform) to all columns of a pandas DataFrame.

    Parameters:
    data_df (pd.DataFrame): DataFrame where the first column is the time or x-axis values, and the other columns are the signals to be transformed.
    window (str): Type of window function to apply before FFT. Default is 'Hann'.

    Returns:
    amp_df (pd.DataFrame): DataFrame containing the amplitude spectra of the signals.
    phase_df (pd.DataFrame): DataFrame containing the phase spectra of the signals.
    """
    amp_df = pd.DataFrame()
    phase_df = pd.DataFrame()
    for i, time in enumerate(data_df.columns[1:]):
        fft = do_fft(np.array([data_df.iloc[:,0], data_df[time]]), window=window)

        amp_df.insert(i, time, fft[1], True)
        phase_df.insert(i, time, fft[2], True)
    
    amp_df.insert(0, 'freq', fft[0], True)
    phase_df.insert(0, 'freq', fft[0], True)

    return amp_df, phase_df



def rms_fft(fft_array, min_f):
    """
    Calculates the noise floor, i.e., the RMS value of the FFT amplitude above a certain frequency.

    Parameters:
    fft_array (ndarray): 2D array with frequency in fft_array[0] and FFT amplitude in fft_array[1].
    min_f (float): Minimum frequency to consider for RMS calculation. All frequencies above are considered.

    Returns:
    float: RMS value of the FFT amplitude above min_f.
    """
    mask = (fft_array[0] > min_f)
    rms = np.sqrt(np.mean(fft_array[1][mask]**2))
    return rms


def wavelet(data, wavelet, min_time=-np.inf, max_time=np.inf):
    """
    Performs a Continuous Wavelet Transform (CWT) on the given data.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].
    wavelet (str): Wavelet to use for the CWT (e.g., 'mexh', 'morl', etc.). Testing several is recommended.
                        See https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families 
    min_time (float): Minimum time for the CWT. Default is -infinity.
    max_time (float): Maximum time for the CWT. Default is infinity.

    Returns:
    tuple: Contains the time array, frequency array, and the 2D matrix with the CWT.
    """

    mask = (data[0] > min_time) & (data[0] < max_time)
    time = data[0][mask]
    field = data[1][mask]

    # Define the widths for the wavelet transform, using log spacing
    widths = np.geomspace(1, 1024, num=100)
    # Alternatively, use a linear spacing
    # widths = np.linspace(1, 1024, num=100)
    
    sampling_period = np.diff(time).mean()

    cwt, freqs = pywt.cwt(field, widths, wavelet, sampling_period=sampling_period)
    
    cwt = np.abs(cwt[:-1, :-1])

    return time, freqs, cwt
