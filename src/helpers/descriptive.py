import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


PATH = os.path.dirname(os.path.abspath(__file__))


def describe_signal_sim(file_name):
    filename = os.path.join(PATH, os.pardir, os.pardir, 'experiments', 'sim', 'data', file_name)
    with h5py.File(filename,'r') as f:
        data = f['data']
        print(f"Time length: {data.attrs['T']}s")
        print(f"Sampling frequency: {data.attrs['fs']}Hz")
        print("Numberof events:", data.attrs['numOfevents'])
        print("Number of sources:", data.attrs['indices'].shape[0])
        print("Noise variance:", data.attrs['noisevar'])


def draw_signal_sim(file_name, zoom=None):
    filename = os.path.join(PATH, os.pardir, os.pardir, 'experiments', 'sim', 'data', file_name)
    with h5py.File(filename,'r') as f:
        data = f['data']
        print("Number of samples:", data.shape)
        fs = data.attrs['fs']
        print(f"Sampling frequency: {fs} Hz")
        indices = data.attrs['indices']
        # Adjust zoom
        zoom = zoom or (0, data.shape[0])
        data = data[zoom[0]:zoom[1]]
        zoomed_indices = []
        for fidx in range(indices.shape[0]):
            zoomed_indices.append(indices[fidx, np.logical_and(fs*indices[fidx, :] >= zoom[0], fs*indices[fidx, :] < zoom[1])])
        _ = plt.subplots(figsize=(15, 5))
        plt.plot(np.arange(data.shape[0])/fs, np.array(data))
        plt.xlabel("Time (s)")
        colors = ['r', 'g', 'b', 'y']
        for fidx in range(len(zoomed_indices)):
            for j in range(len(zoomed_indices[fidx])):
                plt.axvline(x=zoomed_indices[fidx][j], color=colors[fidx], linestyle='--')

    plt.show()
    
    
if __name__ == "__main__":
    draw_signal_sim("T_3_noise_0.1_num_50_0.hdf5")