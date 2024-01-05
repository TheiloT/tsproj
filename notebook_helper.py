import spikeforest as sf
import spikeinterface.full as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
import yaml
from cdlgr.dataset.dataset import get_dataset

def load_dataset(name):    
    config = yaml.load(open(f'cdlgr/config/dataset/{name}.yaml'), Loader=yaml.FullLoader)
    config["preprocess"] = False
    config["dataset"] = config
    config["output"] = {"verbose": False}
    return get_dataset(config)

def get_spectrogram(recording, end_frame=100000):
    return signal.spectrogram(recording.get_traces(end_frame=end_frame).flatten(), recording.get_sampling_frequency(),
                               nperseg=512, noverlap=256, window=('tukey', 0.25))

def fft_rec(rec, end_frame=100000):    
    tr = rec.get_traces(end_frame=end_frame)
    fs = rec.get_sampling_frequency()
    print(fs)
    t = np.arange(0,end_frame)/fs

    print(tr.shape, t.shape)
    
    # Perform FFT
    fft_result = np.fft.fft(tr)#[:len(tr)//2]
    frequencies = np.fft.fftfreq(len(tr), 1/fs)#[:len(tr)//2]
    
    # Plot the original signal and its frequency domain representation
    plt.figure(figsize=(13, 7))
    plt.subplot(2, 1, 1)
    plt.plot(t, tr)
    plt.title('Original Signal')
    
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('Frequency Domain Representation')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(0, 6000)
    
    plt.tight_layout()
    plt.show()