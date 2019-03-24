import numpy as np
import aifc
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib
import pandas as pd
%matplotlib inline
path="/User/Kunal/"
# ReadAIFF function
def ReadAIFF(file):
''' Reads the frames from the audio clip and returns the uncompressed data '''
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.fromstring(strSig, np.short).byteswap()

# Plot spectrogram function
def plot_spectrogram(filename, whale_flag):
''' Plots a single spectrogram '''
    sound = ReadAIFF(filename)
    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(111)
    # Setting spectrogram parameters
    my_cmap = matplotlib.cm.get_cmap('hsv_r');
    params = {'NFFT':256, 'Fs':2000, 'noverlap':192, 'cmap' : my_cmap}
    plt.specgram(sound, **params);
    title0 = 'Spectrogram - Non-whale sound' if whale_flag == 0 else 'Spectrogram - Whale sound'
    ax1.set_title(title0, fontsize = 16)
    ax1.set_xlabel('Time (seconds)', fontsize = 12)
    ax1.set_ylabel('Frequency (Hz)', fontsize = 12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize = 12)

def plot_sbs_spectrogram(filename_whale, filename_non_whale):
    whale = ReadAIFF(filename_whale)
    nonwhale = ReadAIFF(filename_non_whale)
    plt.figure(figsize = (14,4))
    ax1 = plt.subplot(121)
    my_cmap = matplotlib.cm.get_cmap('hsv_r');
    params = {'NFFT':256, 'Fs':2000, 'noverlap':192, 'cmap' : my_cmap}
    plt.specgram(whale, **params);
    ax1.set_title('Whale sound spectrogram', fontsize = 12)
    ax1.set_xlabel('Time (seconds)', fontsize = 12)
    ax1.set_ylabel('Frequency (Hz)', fontsize = 12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize = 12)
    ax2 = plt.subplot(122)
    plt.specgram(nonwhale, **params);
    ax2.set_title('Non-whale sound spectrogram', fontsize = 12)
    ax2.set_xlabel('Time (seconds)', fontsize = 12)
    ax2.set_ylabel('Frequency (Hz)', fontsize = 12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize = 12)
    plt.show()

# Compare several examples:
plot_sbs_spectrogram(path_data + 'train/train6.aiff', path_data + 'train/train1.aiff')
plot_sbs_spectrogram(path_data + 'train/train7.aiff', path_data + 'train/train2.aiff')
plot_sbs_spectrogram(path_data + 'train/train9.aiff', path_data + 'train/train3.aiff')
plot_sbs_spectrogram(path_data + 'train/train12.aiff', path_data + 'train/train4.aiff')