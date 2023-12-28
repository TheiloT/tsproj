from omegaconf import DictConfig, OmegaConf
import spikeforest as sf
import spikeinterface.full as si
import spikeinterface.widgets as sw
from dataclasses import dataclass
from pprint import pprint

@dataclass
class Dataset:
    recording: si.BaseRecording
    sorting_true: si.BaseSorting

def get_dataset(config: DictConfig):
    type_dataset = config["dataset"]["type"]

    t_start = config["dataset"]["tstart_s"]
    t_stop = config["dataset"]["tstop_s"]

    if type_dataset == "spikeforest":
        print(f"Loading dataset {config['dataset']['name']}/{config['dataset']['recording']}...")
        if "uri" in config["dataset"]:
            uri = config["dataset"]["uri"]
            all_recordings = sf.load_spikeforest_recordings(uri)
            pprint([(all_recording.study_name, all_recording.recording_name) for all_recording in all_recordings])
            dataset_raw = [R for R in all_recordings if R.study_name == config["dataset"]["name"] and R.recording_name == config["dataset"]["recording"]][0]
        else:
            dataset_raw = sf.load_spikeforest_recording(study_name=config["dataset"]["name"],
                                                    recording_name=config["dataset"]["recording"])
        print(f'{dataset_raw.study_set_name}/{dataset_raw.study_name}/{dataset_raw.recording_name}')
        print(f'Num. channels: {dataset_raw.num_channels}')
        print(f'Duration (sec): {dataset_raw.duration_sec}')
        print(f'Sampling frequency (Hz): {dataset_raw.sampling_frequency}')
        print(f'Num. true units: {dataset_raw.num_true_units}')
        print('')

        print("Getting recording extractor...")
        recording = dataset_raw.get_recording_extractor()
        print("Getting sorting true extractor...")
        sorting_true = dataset_raw.get_sorting_true_extractor()
        print(f'Recording extractor info: {recording.get_num_channels()} channels, {recording.get_sampling_frequency()} Hz, {recording.get_total_duration()} sec')
        print(f'Sorting extractor info: unit ids = {sorting_true.get_unit_ids()}, {sorting_true.get_sampling_frequency()} Hz')
        print('')
        for unit_id in sorting_true.get_unit_ids():
            st = sorting_true.get_unit_spike_train(unit_id=unit_id)
            print(f'Unit {unit_id}: {len(st)} events')
        print('')
        print('Channel locations:')
        print('X:', recording.get_channel_locations()[:, 0].T)
        print('Y:', recording.get_channel_locations()[:, 1].T)

        if t_start is not None or t_stop is not None:
            print("Subsetting recording...")
            recording = recording.frame_slice(start_frame=int(t_start * recording.get_sampling_frequency()), end_frame=int(t_stop * recording.get_sampling_frequency()))
            # recording = si.SubRecordingExtractor(recording, start_frame=int(t_start * recording.get_sampling_frequency()), end_frame=int(t_stop * recording.get_sampling_frequency()))
            print("Subsetting sorting...")
            sorting_true = sorting_true.frame_slice(start_frame=int(t_start * sorting_true.get_sampling_frequency()), end_frame=int(t_stop * sorting_true.get_sampling_frequency()))
            # sorting_true = si.SubSortingExtractor(sorting_true, start_frame=int(t_start * sorting_true.get_sampling_frequency()), end_frame=int(t_stop * sorting_true.get_sampling_frequency()))


        if config["dataset"]["preprocess"]:
            print("Preprocessing recording...")
            recording = si.bandpass_filter(recording, freq_min=config["dataset"]["preprocess_params"]["freq_min"], freq_max=config["dataset"]["preprocess_params"]["freq_max"])
            recording = si.common_reference(recording, reference='global')
            recording = si.whiten(recording, int_scale=200,
                                  chunk_size=1000)

        wv = si.extract_waveforms(recording, sorting_true, max_spikes_per_unit=2500,
                                mode="memory")
        # for i in range(8):
        #     sw.plot_spikes_on_traces(wv, channel_ids=[i for i in range(i*8, (i+1)*8)])
        #     import matplotlib.pyplot as plt
        #     plt.show()
        sw.plot_spikes_on_traces(wv, channel_ids=[18])
        import matplotlib.pyplot as plt
        plt.show()
        
        return Dataset(recording=recording, sorting_true=sorting_true)
    
    elif type_dataset == "synth":
        recording, sorting = si.generate_ground_truth_recording(
            durations=[config["dataset"]["duration_s"]],
            num_units=3,
            seed=0,
            noise_kwargs=dict(
                noise_level=1,
                strategy='on_the_fly'
            ),
        )        
        if t_start is not None or t_stop is not None:
            print("Subsetting recording...")
            recording = recording.frame_slice(start_frame=int(t_start * recording.get_sampling_frequency()), end_frame=int(t_stop * recording.get_sampling_frequency()))
            sorting = sorting.frame_slice(start_frame=int(t_start * sorting.get_sampling_frequency()), end_frame=int(t_stop * sorting.get_sampling_frequency()))
            #
        return Dataset(recording=recording, sorting_true=sorting)