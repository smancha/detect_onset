from librosa.onset import onset_detect
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
from scipy.io import wavfile


def find_speech_onset(wav_path, nothing_before, plotted):
    """Detects and optionally plots the first three audio onsets in a WAV file.

    This function reads a WAV file, applies noise reduction, and uses Librosa's 
    onset detection to identify the timing of audio onset events. It filters out 
    any onsets that occur before a specified minimum time (in milliseconds) to reduce 
    false positives from noise or artifacts. Optionally, it can plot the 
    denoised waveform with onset markers.

    Args:
        wav_path (str): Path to the WAV file to process.
        nothing_before (float): Minimum onset time in milliseconds. Any onsets detected before this 
            threshold will be discarded.
        plotted (bool): If True, plots the denoised audio waveform with vertical dashed 
            lines marking the detected onsets. If False, no plot is shown.

    Returns:
        numpy.ndarray: An array containing up to the first three detected onset times 
            (in seconds) after filtering.

    Raises:
        FileNotFoundError: If the specified WAV file does not exist.
        ValueError: If the WAV file is empty or cannot be read.

    Notes:
        - Uses noisereduce package for noise removal.
        - Uses librosa package for onset detection.

    Example:
        >>> find_speech_onset("audio.wav", nothing_before=100, plotted=False)
        array([0.22, 1.05, 2.47])
    """

    # read in wav
    rate, signal = wavfile.read(wav_path)

    # convert to np array
    signal = signal.astype(np.float32)

    # remove noise
    denoised_signal = nr.reduce_noise(y=signal, sr=rate)

    # detect onsets with librosa
    onsets = onset_detect(y=denoised_signal, sr=rate, units='time')

    # filter onsets that are too fast, converted to ms from the input parameter
    onsets = onsets[onsets >= (nothing_before / 1000.0)]



    if plotted:
        print(f"First three onsets are: {np.round(onsets[:3]*1000,1)} ms")
        # Time axis for waveform
        time = np.arange(len(denoised_signal)) / rate

        # Plot
        plt.figure(figsize=(14, 4))
        plt.plot(time, denoised_signal, label='Waveform', alpha=0.7)
        lowest = np.min(denoised_signal)
        highest = np.max(denoised_signal)
        plt.vlines(onsets, ymin=lowest, ymax=highest, color='r', linestyle='--', label='Onsets')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform with Onsets')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return np.round(onsets[:3]*1000,1)
        
    # return first 3 onsets for sanity
    else:
        print(f"First three onsets are: {np.round(onsets[:3]*1000,1)} ms")
        return np.round(onsets[:3]*1000,1)
