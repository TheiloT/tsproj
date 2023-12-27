import spikeforest as sf

import kachery_cloud as kcl
# import mountaintools as mt

def main():
    # mt.configDownloadFrom('spikeforest.public')
    # X = mt.readDir("sha1dir://c0879a26f92e4c876cd608ca79192a84d4382868.manual_franklab/tetrode_600s/sorter1_1")
    # for study_set_name, d in X['dirs'].items():
    #     for study_name, d2 in d['dirs'].items():
    #         for recording_name, d3 in d2['dirs'].items():
    #             print(f'{study_set_name}/{study_name}/{recording_name}')

    # f1 = kcl.load_file_info("sha1dir://c0879a26f92e4c876cd608ca79192a84d4382868.manual_franklab/tetrode_600s/sorter1_1/firings_true.mda")
    # print(f1)
    # f2 = kcl.load_file("sha1dir://c0879a26f92e4c876cd608ca79192a84d4382868.manual_franklab/tetrode_600s/sorter1_1")
    # print(f2)
    # uri_default = r"sha1://1d343ed7e876ffd73bd8e0daf3b8a2c4265b783c?spikeforest-recordings.json"
    # # uri frank lab
    uri = r"QmYo54whckFsVxtc1Hv48aKzXyggmK25MBhXb4VpJDVrWz?spikeforest-recordings"
    R = sf.load_spikeforest_recordings(uri=uri)
    for rec in R:
        print(f'{rec.study_set_name}/{rec.study_name}/{rec.recording_name}')
    print(f'{R.study_set_name}/{R.study_name}/{R.recording_name}')
    print(f'Num. channels: {R.num_channels}')
    print(f'Duration (sec): {R.duration_sec}')
    print(f'Sampling frequency (Hz): {R.sampling_frequency}')
    print(f'Num. true units: {R.num_true_units}')
    print('')

    recording = R.get_recording_extractor()
    sorting_true = R.get_sorting_true_extractor()

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

if __name__ == '__main__':
    main()