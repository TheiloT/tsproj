import hydra
from omegaconf import DictConfig, OmegaConf
import os
from cdlgr.dataset.dataset import get_dataset
from spikeinterface.widgets import plot_traces, plot_spikes_on_traces, plot_study_run_times, plot_unit_waveforms
import spikeinterface.full as si
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from cdlgr.outputs.plot import plot_preprocessed
from cdlgr.model.dictionary import Dictionary
from cdlgr.model.cdl import CDL

@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg: DictConfig):
    if cfg["output"]["verbose"] > 0:
        print()
        print(OmegaConf.to_yaml(cfg))
        print(f"Working directory : {os.getcwd()}")
        print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    dataset = get_dataset(cfg)
    plot_preprocessed(dataset, cfg)
   
    dictionary = Dictionary(dataset, cfg)
    dictionary.initialize()

    cdl = CDL(dictionary, cfg)
    traces_seg = cdl.split_traces()
    cdl.train(traces_seg)
    if cfg["output"]["verbose"] > 0:
        print()
    cdl.test()

if __name__ == "__main__":
    main()