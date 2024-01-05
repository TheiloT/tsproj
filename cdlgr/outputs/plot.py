import spikeinterface.full as si
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import numpy as np

def plot_preprocessed(dataset, config: DictConfig):
    if config["output"]["preprocess"]["plot_traces"] or config["output"]["preprocess"]["plot_waveforms"] or config["output"]["preprocess"]["plot_snrs"]:
        # to do avoid duplicated waveforms compute
        wv = si.extract_waveforms(dataset.recording, dataset.sorting_true, max_spikes_per_unit=2500,
                                mode="memory")
    if config["output"]["preprocess"]["plot_traces"]: 
        # w = plot_traces(recording=dataset.recording, backend="matplotlib", time_range=[0, 10])
        si.plot_spikes_on_traces(
            wv
        )
        plt.show()

    
    if config["output"]["preprocess"]['plot_waveforms'] or config["output"]["preprocess"]["plot_snrs"]:
        # bug if signal shorter than chunk size
        snrs = si.compute_snrs(wv)
        snrs_arr = np.array(list(snrs.values()))

    if config["output"]["preprocess"]['plot_waveforms']:
        # ags = np.argsort(snrs_arr)[::-1]
        si.plot_unit_waveforms(
            wv,
            # unit_ids=ags[:5]+1,
            alpha_waveforms=0.1
        )
        plt.show()
    if config["output"]["preprocess"]["plot_snrs"]:
        plt.plot(snrs.keys(), snrs.values(), 'o')
        plt.show()


def plot_traces(traces, fs, label):
    """ Plot traces """
    plt.figure()
    plt.plot(np.arange(traces.shape[0])/fs, traces)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a. u.)")
    plt.title(f"{label.capitalize()} signal traces")
    plt.savefig(f"{label}_traces.png")
    
    
def plot_templates(templates, fs, channel, normalized=True):
    """ Plot templates, normalized or not """
    times = 1000*np.arange(templates[0].shape[0])/fs
    plt.figure(figsize=(10,5))
    for i in range(templates.shape[0]):
        template = templates[i, :, channel]
        if normalized:
            template /= np.linalg.norm(template)
        plt.plot(times, template, label=f"Template {i}")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel(f"Amplitude {'' if normalized else '(a. u.)'}")
    plt.title(f"Templates {'(normalized)' if normalized else ''}")
    plt.savefig(f"templates{'-n' if normalized else ''}.png")
    

def plot_reconstructed(traces_seg, seg_idx, reconstructed_final, seg_nb, active_templates, active_i, min_diff, min_diff_unit, mode, label):
    """ 
    Plot the original trace and the reconstructed trace
    
    :param traces_seg: List of traces segments
    :param seg_idx: Index of the start sample of the segment to plot
    :param reconstructed_final: The reconstructed trace
    :param seg_nb: Segment number
    :param active_templates: Active templates
    :param active_i: Active templates (among interpolated templates)
    :param min_diff: Minimum difference
    :param min_diff_unit: Unit of minimum difference
    :param mode: Mode of reconstruction ("split" or "whole")
    :param label: Label for the plot    
    """
    plt.close('all')
    plt.figure()
    plt.plot(traces_seg[seg_idx][:], label="original", marker="x")
    plt.plot(reconstructed_final[seg_nb, :], label="reconstructed", marker="+")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (a. u.)")
    plt.legend()
    if mode == "split":
        plt.title(f"Active templates: {active_templates} - interpolated {active_i}, unit {min_diff_unit}, min. dist {min_diff}")
    elif mode == "whole":
        plt.title(f"Reconstruction of the whole signal")
    else:
        raise ValueError(f"Unknown mode {mode}")
    plt.savefig(f"reconstructed_{seg_nb}_{seg_idx}_{label}.png")
    plt.close()


def plot_firing(traces_seg, seg_idx, reconstructed_final, seg_nb, template, firing_pos, closest_unit, sample_win, ftype, label):
    """ 
    Plot the original trace and the reconstructed trace
    
    :param traces_seg: List of traces segments
    :param seg_idx: Index of the start sample of the segment to plot
    :param reconstructed_final: The reconstructed trace
    :param seg_nb: Segment number
    :param template: Fired template
    :param firing_pos: Sample position of the fired template within the segment
    :param closest_unit: Unit closest to firing
    :param sample_win: Sample window around the firing position within which to plot
    :param ftype: Type of firing (true or false positive)
    :param label: Label for the plot    
    """
    plt.close('all')
    plt.figure()
    plt.plot(traces_seg[seg_idx][max(0, firing_pos-sample_win):firing_pos+sample_win], label="original", marker="x")
    plt.plot(reconstructed_final[seg_nb, :][max(0, firing_pos-sample_win):firing_pos+sample_win], label="reconstructed", marker="+")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (a. u.)")
    plt.legend()
    plt.title(fr"{ftype} $-$ Fired template: {template}, closest unit: {closest_unit}")
    plt.savefig(f"reconstructed_{seg_nb}_{seg_idx}_{label}_{ftype}.png")
    plt.close()
    

def plot_template_and_truth(template_pred, template_truth, unit, error, fs, iteration):
    """
    Plot the estimated template and the true template. Error is $\sqrt(1 - \langle \hat{a}, a \rangle^2)$ with $\hat{a}$ the estimated template and $a$ the true template.
    """
    plt.figure(figsize=(10,5))
    times = 1000*np.arange(template_pred.shape[0])/fs
    plt.plot(times, template_truth, label="True", marker='x')
    plt.plot(times, template_pred, label="Estimated", marker='+')
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (normalized)")
    plt.title(f"Element {unit} - Error {error:.3f}")
    plt.legend()
    plt.savefig(f"dictionary-{iteration:03}-{unit:03}.png")
    
    
def plot_template_and_truth_interp(template_pred, template_truth_interp, unit, error, idx, fs, interpolate, offset, iteration):
    """
    Plot the estimated template and the true (interpolated) template. The true template is shifted by the offset (in ms) that brings it closest to the estimation (with a positive offset shifting the template to the right), and the associated shift is indicated to measure the horizontal offset learnt by the algorithm. Error is $\sqrt(1 - \langle \hat{a}, a \rangle^2)$ with $\hat{a}$ the estimated template and $a$ the true template.
    """
    plt.figure(figsize=(10,5))
    times = 1000*np.arange(template_pred.shape[0])/fs
    time_offset = 1000*offset/fs if offset is not None else np.nan
    true_label = "True" if interpolate == 0 else fr"True (interpolated by ${idx}\times\Delta_K$)"
    if offset is not None:
        plt.plot(np.roll(template_truth_interp, offset), label=true_label, marker='x')
    plt.plot(template_pred, label="Estimated", marker='+') 
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (normalized)")
    plt.title(f"Element {unit} - Error {error:.3f} - Best shift {time_offset:.2f} ms")
    plt.legend()
    plt.savefig(f"dictionary-i-{iteration:03}-{unit:03}.png")
