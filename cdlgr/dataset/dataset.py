import numpy as np
from omegaconf import DictConfig, OmegaConf
import spikeforest as sf
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
from dataclasses import dataclass
from pprint import pprint
from time import time

@dataclass
class Dataset:
    recording: si.BaseRecording
    recording_test: si.BaseRecording
    sorting_true: si.BaseSorting
    sorting_true_test: si.BaseSorting

def get_dataset(config: DictConfig):
    type_dataset = config["dataset"]["type"]

    t_start = config["dataset"]["tstart_s"]
    t_stop = config["dataset"]["tstop_s"]

    t_start_test = config["dataset"]["tstart_test_s"]
    t_stop_test = config["dataset"]["tstop_test_s"]

    if type_dataset == "spikeforest":
        if config["output"]["verbose"] > 0:
            print(f"Loading dataset {config['dataset']['name']}/{config['dataset']['recording']}...")
        if "uri" in config["dataset"]:
            uri = config["dataset"]["uri"]
            all_recordings = sf.load_spikeforest_recordings(uri)
            # pprint([(all_recording.study_name, all_recording.recording_name) for all_recording in all_recordings])
            dataset_raw = [R for R in all_recordings if R.study_name == config["dataset"]["name"] and R.recording_name == config["dataset"]["recording"]][0]
        else:
            dataset_raw = sf.load_spikeforest_recording(study_name=config["dataset"]["name"],
                                                    recording_name=config["dataset"]["recording"])
        if config["output"]["verbose"] > 0:
            print(f'{dataset_raw.study_set_name}/{dataset_raw.study_name}/{dataset_raw.recording_name}')
            print(f'Num. channels: {dataset_raw.num_channels}')
            print(f'Duration (sec): {dataset_raw.duration_sec}')
            print(f'Sampling frequency (Hz): {dataset_raw.sampling_frequency}')
            print(f'Num. true units: {dataset_raw.num_true_units}')
            print('')

        if config["output"]["verbose"] > 0:
            print("Getting recording extractor...")
        recording = dataset_raw.get_recording_extractor()
        if config["output"]["verbose"] > 0:
            print("Getting sorting true extractor...")
        sorting_true = dataset_raw.get_sorting_true_extractor()
        if config["output"]["verbose"] > 0:
            print(f'Recording extractor info: {recording.get_num_channels()} channels, {recording.get_sampling_frequency()} Hz, {recording.get_total_duration()} sec')
            print(f'Sorting extractor info: unit ids = {sorting_true.get_unit_ids()}, {sorting_true.get_sampling_frequency()} Hz')
            print('')
        for unit_id in sorting_true.get_unit_ids():
            st = sorting_true.get_unit_spike_train(unit_id=unit_id)
            if config["output"]["verbose"] > 1:
                print(f'Unit {unit_id}: {len(st)} events')
        if config["output"]["verbose"] > 1:
            print('')
            print('Channel locations:')
            print('X:', recording.get_channel_locations()[:, 0].T)
            print('Y:', recording.get_channel_locations()[:, 1].T)

    elif type_dataset == "synth":
        ####################################
        # Generate data as in the paper
        ####################################
        # Noise
        # noise_start = config["dataset"]["gen"]["noise"]["start"]
        # noise_end = config["dataset"]["gen"]["noise"]["stop"]
        # noise_step = config["dataset"]["gen"]["noise"]["step"]
        # noise_vars = np.arange(noise_start, noise_end+noise_step-0.001, noise_step)
        # noise_vars = [round(noise,3) for noise in noise_vars]
        
        # Define dictionaries
        num_sources = config["dataset"]["sources"]["num"]
        filter_length = config["dataset"]["sources"]["length_ms"]/1000
        factor = 10/filter_length
        dictionary = {}
        dictionary[0] = lambda x: (factor*x)*np.exp(-(factor*x)**2)*np.cos(2*np.pi*(factor*x)/4)
        dictionary[1] = lambda x: (factor*x)*np.exp(-(factor*x)**2)
        
        # Generate data
        # Number of channels is 1
        fs = config["dataset"]["fs"]
        amps = config["dataset"]["gen"]["amps"]
        numOfevents = config["dataset"]["gen"]["numOfEvents"]
        T = config["dataset"]["gen"]["T"]
        generation_seed = config["dataset"]["gen"]["seed"]
        
        if config["output"]["verbose"] > 0:
            print("Noise ", config["dataset"]["gen"]["noise"])
            print("Generating data")
        truth, event_indices = generate_Simulated_continuous(config, numOfevents, T, fs, dictionary, filter_length, amps, seed=generation_seed)
        if generation_seed is not None:
            np.random.seed(generation_seed)
        signal = truth + config["dataset"]["gen"]["noise"]*np.random.randn(T*fs)
        np.random.seed(int(time()))

        # print("Saving data to", folder_name)
        # filename = os.path.join(PATH, 'experiments', folder_name, 'data','T_{}_noise_{}_num_{}_{}.hdf5'.format(T,noisevar, config_d['numOfevents'], i))
        # with h5py.File(filename,'w') as f:
        #     # dset = f.create_dataset("data", data = signal[:-1])
        #     dset = f.create_dataset("data", data = signal)
        #     dset.attrs['fs'] = fs
        #     dset.attrs['T'] = config_d['T']
        #     dset.attrs['numSources'] = config_d['numSources']
        #     dset.attrs['numOfevents'] = config_d['numOfevents']
        #     dset.attrs['indices'] = event_indices
        #     dset.attrs['noisevar'] = noisevar
        #     dset.attrs['amps'] = amps
        #     dset.attrs['filter_length'] = config_d['filter_length']
        if config["output"]["verbose"] > 0:
            print("\nData generated")
        
        
        ####################################
        # Convert to a spikeinterface BaseRecording
        ####################################
        # Recording
        traces = np.expand_dims(signal, axis=1)
        recording = se.NumpyRecording(traces_list=[traces], sampling_frequency=fs)
        recording.set_dummy_probe_from_locations(np.zeros((1, 2)))  # Dummy probe with 1 channel and dimension 2
        recording.annotate(is_filtered=True)  # Perform no pre-processing on this dataset
        
        # Sorting
        times = (fs*(event_indices.flatten() + filter_length/2)).astype(int)
        labels = np.zeros(num_sources * numOfevents, dtype=int)
        for i in range(num_sources):
            labels[i*numOfevents:(i+1)*numOfevents] = i
        sorting_true = se.NumpySorting.from_times_labels([times], [labels], fs)
        
    recording_test, sorting_test = subset_data(config, recording, sorting_true, t_start_test, t_stop_test, "test")
    recording, sorting_true = subset_data(config,recording, sorting_true, t_start, t_stop, "training")

    if config["dataset"]["preprocess"] and config["dataset"]["type"] != "synth":
        if config["output"]["verbose"] > 0:
            print("Preprocessing recording...")
        recording = si.bandpass_filter(recording, freq_min=config["dataset"]["preprocess_params"]["freq_min"], freq_max=config["dataset"]["preprocess_params"]["freq_max"])
        recording = si.common_reference(recording, reference='global')
        recording = si.whiten(recording, int_scale=200,
                                chunk_size=1000)
    
    return Dataset(recording=recording, sorting_true=sorting_true, recording_test=recording_test, sorting_true_test=sorting_test)
        

def generate_Simulated_continuous(config: DictConfig, numOfevents, T, fs, dictionary, filter_length, amps=[0,1], seed=None):
    """
    Generate continuous data and its sampled version.
    For now, assume that we know the templates. These templates start from -5 to 5

    Inputs
    ======
    dictionary: a dictionary of continuous functions
    filter_length: filter length (in seconds)

    amps: array (two elements)
        lower boudn and upper bound for the amplitudes
        
    seed: int
        seed for the random number generator. If None, no seed is used.


    Outputs
    =======
    signal: array_like
        Generated signal

    """
    assert(len(amps)==2 and amps[0]<amps[1]), "Wrong amplitude arguments"

    if seed is not None:
        np.random.seed(seed)
    
    numOfelements = len(dictionary.keys())

    signal = np.zeros(T*fs)
    interval = 1/fs

    # Generate event indices
    events_indices = np.zeros((numOfelements, numOfevents))

    for fidx in np.arange(numOfelements):
        if config["output"]["verbose"] > 1:
            print(f"\rGenerating {fidx+1}/{numOfelements} elements", end="")
        events_idx = np.sort(T*np.random.rand(numOfevents))
        if config["output"]["verbose"] > 1:
            print("events", events_idx)

        # Event index generation
        idx_diff = np.where(events_idx[1:] - events_idx[:-1]<filter_length)[0]
        condition = len(idx_diff) == 0 and (events_idx[0] > filter_length) and (events_idx[-1] < T - filter_length)

        counter = 0
        while not condition:
            if config["output"]["verbose"] > 1:
                print(f"\rGenerating {fidx+1}/{numOfelements} elements (retry {counter+1})", end="")
            counter += 1
            if events_idx[0] <= filter_length:
                new_idx = T*np.random.rand()
                events_idx[0] = new_idx
            elif events_idx[-1]>= T-filter_length:
                new_idx = T*np.random.rand()
                events_idx[-1] = new_idx
            else:
                for i in idx_diff:
                    new_idx = T*np.random.rand()
                    events_idx[i+1] = new_idx

            events_idx = np.sort(events_idx)

            idx_diff = np.where(events_idx[1:] - events_idx[:-1]<filter_length)[0]
            condition = len(idx_diff) == 0 and (events_idx[0] > filter_length) and (events_idx[-1] < T - 1*filter_length)

        events_indices[fidx,:] = events_idx

        # Signal generation
        for idx, event_timestamp in enumerate(events_idx):
            start_sample = 0
            amp = np.random.uniform(amps[0], amps[1])

            # Find the closest sample to the event starting point
            start_sample = int(np.ceil(event_timestamp * fs))

            # Distance between the template starting point (continuous) and the first sample grid
            delta = start_sample * interval - event_timestamp

            maxamp = -100
            filter_length_in_samples = int(filter_length * fs)
            filter_realization = np.zeros(filter_length_in_samples)

            for sidx in np.arange(filter_length_in_samples):
                ts = -filter_length_in_samples/2 * interval + delta + sidx*interval
                point = dictionary[fidx](ts)
                filter_realization[sidx] = point
                if point>maxamp:
                    maxamp = point
            signal[start_sample : start_sample + filter_length_in_samples] += filter_realization/maxamp*amp
            
    np.random.seed(int(time()))
            
    return signal, events_indices


def subset_data_slice(config: DictConfig, data, t_start, t_stop, message):
    if t_start is not None or t_stop is not None:
        if config["output"]["verbose"] > 0:
            print(f"Subsetting {message}...")
        return data.frame_slice(start_frame=int(t_start * data.get_sampling_frequency()), end_frame=int(t_stop * data.get_sampling_frequency()))
    return data


def subset_data(config: DictConfig, recording, sorting, t_start, t_stop, message):
    recording = subset_data_slice(config, recording, t_start, t_stop, message)
    sorting_true = subset_data_slice(config, sorting, t_start, t_stop, message)
    return recording, sorting_true