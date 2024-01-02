import spikeinterface.full as si
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import numpy as np

def plot_preprocessed(dataset, config: DictConfig):
    if config["output"]["plot_traces"] or config["output"]["plot_waveforms"] or config["output"]["plot_snrs"]:
        # to do avoid duplicated waveforms compute
        wv = si.extract_waveforms(dataset.recording, dataset.sorting_true, max_spikes_per_unit=2500,
                                mode="memory")
    if config["output"]["plot_traces"]: 
        # w = plot_traces(recording=dataset.recording, backend="matplotlib", time_range=[0, 10])
        si.plot_spikes_on_traces(
            wv
        )
        plt.show()

    
    if config["output"]['plot_waveforms'] or config["output"]["plot_snrs"]:
        # bug if signal shorter than chunk size
        snrs = si.compute_snrs(wv)
        snrs_arr = np.array(list(snrs.values()))

    if config["output"]['plot_waveforms']:
        # ags = np.argsort(snrs_arr)[::-1]
        si.plot_unit_waveforms(
            wv,
            # unit_ids=ags[:5]+1,
            alpha_waveforms=0.1
        )
        plt.show()
    if config["output"]["plot_snrs"]:
        plt.plot(snrs.keys(), snrs.values(), 'o')
        plt.show()