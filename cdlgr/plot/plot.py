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
        
def plot_reconstructed(traces_seg, seg_idx, reconstructed_final, seg_nb, active_atoms, active_i, min_diff, min_diff_unit, label):
    """ 
    Plot the original trace and the reconstructed trace
    
    :param traces_seg: List of traces segments
    :param seg_idx: Index of the start sample of the segment to plot
    :param reconstructed_final: The reconstructed trace
    :param seg_nb: Segment number
    :param active_atoms: Active atoms
    :param active_i: Active atoms (among interpolated atoms)
    :param min_diff: Minimum difference
    :param min_diff_unit: Unit of minimum difference
    :param label: Label for the plot    
    """
    plt.close('all')
    plt.figure()
    plt.plot(traces_seg[seg_idx][:], label="original", marker="x")
    plt.plot(reconstructed_final[seg_nb, :], label="reconstructed", marker="+")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (a. u.)")
    plt.legend()
    plt.title(f"Active filters: {active_atoms} - interpolated {active_i}, unit {min_diff_unit}: dist {min_diff}")
    plt.savefig(f"reconstructed_{seg_nb}_{seg_idx}_{label}.png")
    plt.close()


def plot_firing(traces_seg, seg_idx, reconstructed_final, seg_nb, atom, firing_pos, closest_atom, sample_win, ftype, label):
    """ 
    Plot the original trace and the reconstructed trace
    
    :param traces_seg: List of traces segments
    :param seg_idx: Index of the start sample of the segment to plot
    :param reconstructed_final: The reconstructed trace
    :param seg_nb: Segment number
    :param atom: Fired atom
    :param firing_pos: Sample position of the fired atom within the segment
    :param closest_atom: Atom closest to firing
    :param sample_win: Sample window around the firing position within which to plot
    :param ftype: Type of firing (true or false positive)
    :param label: Label for the plot    
    """
    plt.close('all')
    plt.figure()
    plt.plot(traces_seg[seg_idx][max(0, firing_pos-sample_win):firing_pos+sample_win], label="original", marker="x")
    plt.plot(reconstructed_final[seg_nb, :][max(0, firing_pos-sample_win):firing_pos+sample_win], label="reconstructed", marker="+")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (a. u.)")
    plt.legend()
    plt.title(f"{ftype}: Fired atom: {atom}, closest atom: {closest_atom}")
    plt.savefig(f"reconstructed_{seg_nb}_{seg_idx}_{label}_{ftype}.png")
    plt.close()
    