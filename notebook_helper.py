import spikeforest as sf
import spikeinterface.full as si
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
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

def fft_rec(rec, end_frame=100000, freq_lim=6000):    
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
    plt.xlim(0, freq_lim)
    
    plt.tight_layout()
    plt.show()
    
    
# Experiments helpers
def fetch_metric(path, metric="accuracy", mode="test"):
    """
    Fetch accuracy stored in experiment folder indicated by path, for train or test mode.
    """
    df = pd.read_csv(os.path.join(path, f"perf-{mode}.csv"))
    return (df[metric].values, df.index.values)


def show_plot(path):
    """
    Fetch plot stored in experiment folder indicated by path.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(plt.imread(path))
    plt.axis("off")
    plt.show()


def find_file_with_suffix(path, suffix):
    """
    Find file with specified suffix in path.
    """
    files = glob.glob(os.path.join(path, f"*{suffix}"))
    for file in files:
        if file.endswith(suffix):
            return file
    return None
    

def show_plots(path, num_elements=2, num_iterations=10):
    """
    Show plots stored in experiment folder indicated by path.
    """
    print("Initial dictionary:")
    for el in range(num_elements):
        show_plot(os.path.join(path, f"dictionary--01-{el:03}.png"))
    # print("Final dictionary:")
    # for el in range(num_elements):
    #     show_plot(os.path.join(path, f"dictionary-{num_iterations-1:03}-{el:03}.png"))
    print("Final dictionary (shifted and interpolated truth):")
    for el in range(num_elements):
        show_plot(os.path.join(path, f"dictionary-i-{num_iterations-1:03}-{el:03}.png"))
    print("True positive on test set:")
    TP_name = find_file_with_suffix(path, "test_TP.png")
    if TP_name:
        show_plot(os.path.join(path, os.path.basename(TP_name)))
    else:
        print("No true positive for test run.")
    print("False positive on test set:")
    FP_name = find_file_with_suffix(path, "test_FP.png")
    if FP_name:
        show_plot(os.path.join(path, os.path.basename(FP_name)))
    else:
        print("No false positive for test run.")
    print("Confusion matrix:")
    show_plot(os.path.join(path, "confusion_matrix-test.png"))
    print("Time:")
    show_plot(os.path.join(path, "time.png"))
   

def report_average_multi_run_perfs(dataset, experiment, metric="accuracy", num_elements=2, interpolations=np.array([0, 10]), seeds=np.arange(1, 6), mode="test"):
    """
    Average metrics over multiple runs of the same experiment.
    """
    exp_path = os.path.join("outputs", f"{dataset}_experiments", f"experiment_{experiment}", "multi")
    units = np.arange(num_elements)

    metrics_by_interp_and_unit = {interpolation: {unit: [] for unit in units} for interpolation in interpolations}
    print(f"Averaged {metric} by interpolation and unit:")
    # Iterate over each interpolation value
    for interpolation in interpolations:
        # Average over seed values
        for seed in seeds:
            metric_by_unit = fetch_metric(os.path.join(exp_path, f"interp_{num_elements}_{interpolation}", f"seed_{seed}"), metric=metric, mode=mode)[0]
            for unit in units:
                metrics_by_interp_and_unit[interpolation][unit].append(metric_by_unit[unit])
        print("interpolation:", interpolation)
        for unit in units:
            print(f"unit {unit}: {np.mean(metrics_by_interp_and_unit[interpolation][unit]):.3f} +/- {np.std(metrics_by_interp_and_unit[interpolation][unit]):.3f}")
        print()


def report_multirun_dict_error(dataset, experiment, num_elements=2, interpolations=np.array([0, 10]), seeds=np.arange(1, 6)):
    """
    Report distance between dictionaries over multiple runs of the same experiment.
    """
    exp_path = os.path.join("outputs", f"{dataset}_experiments", f"experiment_{experiment}", "multi")
    units = np.arange(num_elements)

    error_by_interp_and_unit = {interpolation: {unit: {"error": [], "offset": []} for unit in units} for interpolation in interpolations}
    print("Averaged error and time offset by interpolation and unit:")
    # Iterate over each interpolation value
    for interpolation in interpolations:
        # Average over seed values
        for seed in seeds:
            error_and_offset_by_unit = np.load(os.path.join(exp_path, f"interp_{num_elements}_{interpolation}", f"seed_{seed}", "interpolated_dictionary_error.npz"))
            error_by_unit = error_and_offset_by_unit["error"]
            offset_by_unit = error_and_offset_by_unit["time_offset"]
            for unit in units:
                error_by_interp_and_unit[interpolation][unit]["error"].append(error_by_unit[unit])
                error_by_interp_and_unit[interpolation][unit]["offset"].append(offset_by_unit[unit])
        print("interpolation:", interpolation)
        for unit in units:
            print(f"unit {unit}:")
            print(f"\terror: {np.mean(error_by_interp_and_unit[interpolation][unit]['error']):.3f} +/- {np.std(error_by_interp_and_unit[interpolation][unit]['error']):.3f}")
            print(f"\toffset: {np.mean(error_by_interp_and_unit[interpolation][unit]['offset']):.3f} ms +/- {np.std(error_by_interp_and_unit[interpolation][unit]['offset']):.3f} ms")
        print()


def report_multi_run_results(experiment, num_elements=2):
    dataset="synth"
    report_average_multi_run_perfs(dataset, experiment, metric="accuracy", num_elements=num_elements)
    print()
    report_multirun_dict_error(dataset, experiment, num_elements=num_elements)