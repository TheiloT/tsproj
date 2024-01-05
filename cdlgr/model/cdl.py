from time import perf_counter
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from cdlgr.model.dictionary import Dictionary
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import pandas as pd
from tqdm import tqdm
import warnings
import json
from itertools import permutations
from copy import deepcopy

from cdlgr.outputs.plot import plot_reconstructed, plot_firing, plot_traces


class CDL:
    def __init__(self, dictionary, config):
        self.config = config
        self.dictionary: Dictionary = dictionary

        self.num_iterations = self.config["model"]["cdl"]["num_iterations"]
        self.interpolate = self.config["model"]["cdl"]["interpolate"]
        
        self.sparsity_tol = self.config["model"]["cdl"]["sparsity_tol"]
        self.error_tol = self.config["model"]["cdl"]["error_tol"]

    def split_traces(self):
        time_preprocessing_begin = perf_counter()
        if self.config["output"]["verbose"] > 0:
            print("\nSplitting traces...")                
        self.channel = self.config["dataset"]["channel"]
        
        fs = self.config["dataset"].get("fs", None)
        fs = fs if fs is not None else self.dictionary.dataset.recording.get_sampling_frequency()

        if self.config["output"]["verbose"] > 0:
            print("\tRetrieving traces...")
        traces = self.dictionary.dataset.recording.get_traces(
            start_frame=0,
            end_frame=min(10000, self.dictionary.dataset.recording.get_num_frames()),
            channel_ids=[self.channel],
        )

        def get_frames(start_frame=None, end_frame=None):
            return self.dictionary.dataset.recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                channel_ids=[self.channel],
            ).flatten()
        
        if self.config["output"]["plot"] > 0:
            plot_traces(traces, fs, "train")

        if self.config["dataset"]["window"]["split"]:
            detect_threshold = 5
            exclude_sweep_ms = self.config["dataset"]["sources"]["length_ms"]/2 if (self.config["dataset"]["type"] == "synth") else 0.1
            peaks = detect_peaks(self.dictionary.dataset.recording, detect_threshold=detect_threshold, exclude_sweep_ms=exclude_sweep_ms,
                                 random_chunk_kwargs={'chunk_size':min(10000, self.dictionary.dataset.recording.get_num_frames() - 5)})
                                #, detect_threshold=5, n_shifts=5, peak_span_ms=0.5, peak_span_samples=None, filter=None, filter_kwargs=None, return_idxs=True, return_times=False, return_peak_span=False, return_channel_idxs=False, verbose=False
            peaks = pd.DataFrame(peaks)
            peaks = peaks[peaks["channel_index"] == self.channel]

            peak_size = int(self.config["dataset"]["window"]["window_size_s"] * fs)
            half_size = peak_size // 2
            traces_seg = {}
            recording_length = self.dictionary.dataset.recording.get_num_frames()
            if self.config["output"]["plot"] > 0:
                plt.figure()
            for i, (_, peak) in tqdm(enumerate(peaks.iterrows()), disable=self.config["output"]["verbose"] == 0):
                peak_idx = int(peak["sample_index"])
                idx = peak_idx
                traces_seg[idx] = np.zeros(peak_size)
                # print(peak_idx)
                if peak_idx - half_size < 0:
                    traces_seg[idx][half_size - peak_idx:] = get_frames(end_frame=peak_idx+half_size) #traces[:peak_idx + half_size]
                elif peak_idx + half_size > recording_length:
                    traces_seg[idx][:recording_length - (peak_idx - half_size)] = get_frames(start_frame=peak_idx - half_size)
                else:
                    traces_seg[idx][:] = get_frames(start_frame=peak_idx - half_size, end_frame=peak_idx + half_size)
                if self.config["output"]["plot"] > 0:
                    plt.plot(traces_seg[idx][:])
            if self.config["output"]["plot"] > 0:
                plt.xlabel("Sample index")
                plt.ylabel("Amplitude (a. u.)")
                plt.title("Traces segments")
                plt.savefig("traces_seg.png")

            # initial dictionary with atoms around peaks
            if self.config["model"]["dictionary"]["init_templates"] == "signal":
                for k in range(self.dictionary.num_elements):
                    self.dictionary.dictionary[:, k] = get_frames(peaks.sample_index.values[k]-self.dictionary.element_length//2,
                                                peaks.sample_index.values[k]+self.dictionary.element_length//2+1)
                # Ensure consistency in the indexation of the filters between true and estimated dictionaries
                best_perm, best_total_error = None, np.infty
                copy_dictionary = deepcopy(self.dictionary)
                for perm in permutations(range(self.dictionary.num_elements)):
                    copy_dictionary.dictionary = self.dictionary.dictionary[:, perm].copy()
                    error = np.sum(copy_dictionary.recovery_error_interp(-1, numOfsubgrids=self.config["model"]["cdl"]["interpolate"], save_plots=False))
                    if error < best_total_error:
                        best_total_error = error
                        best_perm = perm
                self.dictionary.dictionary = self.dictionary.dictionary[:, best_perm]

                self.dictionary.normalize()
        else:
            traces_seg = {}
            traces_seg[0] = get_frames()
            warnings.warn("Performance evaluation only works with window split")
            input()

        time_preprocessing_end = perf_counter()
        if self.config["output"]["verbose"] > 0:
            print("Preprocessing time: ", time_preprocessing_end - time_preprocessing_begin)
        np.savetxt("time_preprocessing.txt", [time_preprocessing_end - time_preprocessing_begin], fmt="%f")
        
        if self.config["output"]["verbose"] > 0:
            print("Splitting done.\n")

        return traces_seg
    
    @property
    def interpolator_type(self):
        return self.config["model"]["cdl"]["interpolator_type"]

    def train(self, traces_seg):
        if self.config["output"]["verbose"] > 0:
            print("Running CDL...")       

        self.dictionary.recovery_error(-1)
        self.dictionary.recovery_error_interp(-1, self.interpolate)
        # exit()

        time_total_begin = perf_counter()
        time_csc = []
        time_update = []
        for i in range(self.num_iterations):
            if self.config["output"]["verbose"] > 0:
                print(f"Iteration {i+1}/{self.num_iterations}")
            sparse_coeffs, interpolated_dict, interpolator, time_csc_diff = self.run_csc(traces_seg)
            time_csc.append(time_csc_diff)
            
            if i != self.num_iterations - 1:
                time_update_begin = perf_counter()
                self.dictionary.update(traces_seg, sparse_coeffs, interpolator)   
                time_update_end = perf_counter()
                time_update.append(time_update_end - time_update_begin)

            if self.config["output"]["plot"] > 1 or (self.config["output"]["plot"] > 0 and i == self.num_iterations-1):
                error = self.dictionary.recovery_error(i+1)
                error2 = self.dictionary.recovery_error_interp(i+1, self.interpolate)[0]
                if self.config["output"]["verbose"] > 0:
                    print("Dictionary error (original and interpolated):", error, error2)
                    print()

        time_total_end = perf_counter()

        if self.config["output"]["verbose"] > 0:
            print("Total time: ", time_total_end - time_total_begin)
            print("CSC time: ", np.sum(time_csc))
            print("Update time: ", np.sum(time_update))
            print()

        np.savez("time.npz", time_csc=time_csc, time_update=time_update, time_total=time_total_end - time_total_begin)

        if self.config["output"]["plot"] > 1:
            plt.close('all')
            plt.plot(time_csc, label="CSC")
            plt.plot(time_update, label="CDU")
            plt.xlabel("Iteration")
            plt.ylabel("Time (s)")
            plt.title("Time per iteration, total {:.2f}s".format(time_total_end - time_total_begin))
            plt.legend()
            plt.tight_layout()
            plt.savefig("time.png")

        self.reconstruct(traces_seg, sparse_coeffs, interpolated_dict, self.dictionary.dataset.sorting_true)

        self.dictionary.save()

        self.save_coeffs(sparse_coeffs)

    def test(self):
        if self.dictionary.dataset.recording_test is not None:
            if self.config["output"]["verbose"] > 0:
                print("==============")
                print("Testing...")
            traces_seg = {}
            traces_seg[0] = self.dictionary.dataset.recording_test.get_traces(
                channel_ids=[self.channel],
            ).flatten()

            self.sparsity_tol = self.config["model"]["cdl"]["sparsity_tol_test"]
            sparse_coeffs, interpolated_dict, _, time_diff = self.run_csc(traces_seg)

            self.reconstruct(traces_seg,
                             sparse_coeffs,
                             interpolated_dict,
                             self.dictionary.dataset.sorting_true_test,
                             "whole",
                             "test")
            
            self.save_coeffs(sparse_coeffs, "test")
            np.savetxt("time_test.txt", [time_diff], fmt="%f")

    def run_csc(self, traces_seg):
        time_csc_begin = perf_counter()
        interpolated_dict, interpolator = self.dictionary.interpolate(self.interpolate, kind=self.interpolator_type)
        sparse_coeffs = {}
        for j in tqdm(traces_seg.keys(), disable=self.config["output"]["verbose"] == 0):
            sparse_coeffs[j] = self.code_sparse(
                    self.csc(
                        traces_seg[j], interpolated_dict, sparsity=None, boundary=True
                    ), interpolated_dict
                )     
        time_csc_end = perf_counter()
        return sparse_coeffs, interpolated_dict, interpolator, time_csc_end - time_csc_begin

    def save_coeffs(self, sparse_coeffs, mode='train'):
        sp = {}
        for k in sparse_coeffs.keys():
            k_key = str(k)
            sp[k_key] = {}
            for i in sparse_coeffs[k].keys():
                i_key = str(i)
                sp[k_key][i_key] = {"idx": sparse_coeffs[k][i]["idx"].tolist(), "amp": sparse_coeffs[k][i]["amp"].tolist()}

        with open(f"sparse_coeffs-{mode}.json", "w") as f:
            json.dump(sp, f, indent=4)

    def get_distance_to_min_diff_unit(self, spike_idx, sorting_true):
        min_diff = np.inf
        min_diff_unit = None
        for unit in sorting_true.unit_ids:
            spikes_idxes = sorting_true.get_unit_spike_train(unit_id=unit)
            min_diff_true = np.abs(spikes_idxes - spike_idx).min()
            if min_diff_true < min_diff:
                min_diff = min_diff_true
                min_diff_unit = unit
        return min_diff, min_diff_unit

    def reconstruct(self, traces_seg, sparse_coeffs, interpolated_dict, sorting_true, mode="split", label="train"):
        """
        Reconstruct the signal from the sparse coefficients.
        
        Inputs
        ======
        traces_seg: dictionary
            Each key represents a segment
            The corresponding value is a 1-D array
        sparse_coeffs: dictionary
            Each key represents the start sample of a segment
            The corresponding value is a sparse code; see code_sparse
        interpolated_dict: 2-D array
            Each column represents a filter
        sorting_true: spikeinterface.SortingExtractor
            Ground truth sorting
        mode: string
            split: split the signal into segments
            whole: reconstruct the whole signal
        """
        spikes = pd.DataFrame(sorting_true.to_spike_vector(concatenated=True))
        if self.config["output"]["verbose"] > 1:
            print(spikes)
        total_number_of_spikes = spikes.shape[0]
      
        spikes_sorting = pd.DataFrame(columns=["sample_index", "unit_index", "amplitude", "error"])

        N_seg = len(traces_seg.keys())
        seg_size = traces_seg[list(traces_seg.keys())[0]].shape[0]

        reconstructed = np.zeros((N_seg, seg_size + interpolated_dict.shape[0] - 1))
        reconstructed_final = np.zeros((N_seg, seg_size))
        
        for seg_nb, seg_idx in enumerate(sparse_coeffs.keys()):
            active_i = []
            idxes = []
            amps = []
            for atom_i in sparse_coeffs[seg_idx].keys():            
                for firing_nb, firing_idx in enumerate(sparse_coeffs[seg_idx][atom_i]["idx"]):
                    if firing_idx > seg_size:
                        warnings.warn(f"idx {firing_idx} larger than seg_size {seg_size}")
                        continue
                    if atom_i not in active_i or mode != "split":
                        active_i.append(atom_i)    
                        idxes.append(firing_idx)
                        amps.append(sparse_coeffs[seg_idx][atom_i]["amp"][firing_nb])
                    else:
                        if self.config["output"]["verbose"] > 0:
                            print("Atom already active")                    
                    reconstructed[seg_nb, firing_idx : firing_idx + len(interpolated_dict[:, atom_i])] += (
                        sparse_coeffs[seg_idx][atom_i]["amp"][firing_nb] * interpolated_dict[:, atom_i]
                    )
            reconstructed_final[seg_nb, :] = reconstructed[seg_nb, :][interpolated_dict.shape[0] - 1:]

            active_atoms = []
            for idx, atom_i in enumerate(active_i):
                if self.interpolate != 0:
                    active_atoms.append(atom_i//self.interpolate)
                else:
                    active_atoms.append(atom_i) 

                if mode == "split":
                    spikes_sorting.loc[seg_nb] = [seg_idx, active_atoms[-1], amps[idx], 0]
                else:
                    reconstructed_part = reconstructed[seg_nb, idxes[idx]:idxes[idx]+self.dictionary.element_length*2]
                    original_part = traces_seg[seg_idx][idxes[idx]-self.dictionary.element_length:idxes[idx]+self.dictionary.element_length]
                    normalized_reconstructed = reconstructed_part/np.linalg.norm(reconstructed_part)
                    normalized_original = original_part/np.linalg.norm(original_part)

                    max_length = min(len(normalized_reconstructed), len(normalized_original))
                    normalized_reconstructed = normalized_reconstructed[:max_length]
                    normalized_original = normalized_original[:max_length]

                    error = np.sqrt(1-np.dot(normalized_reconstructed, normalized_original)**2)
                    spikes_sorting.loc[len(spikes_sorting)] = [idxes[idx]-self.dictionary.element_length//2, active_atoms[-1], amps[idx], error]
            
            if spikes_sorting["sample_index"].values.shape[0] > 0:
                spike_idx = spikes_sorting["sample_index"].values[-1]
                min_diff, min_diff_unit = self.get_distance_to_min_diff_unit(spike_idx, sorting_true)
            else:
                min_diff = np.inf
                min_diff_unit = None                    
                    
            if (seg_nb < 10) or min_diff > 15:
                if self.config["output"]["plot"] > 1:
                    plot_reconstructed(traces_seg, seg_idx, reconstructed_final, seg_nb, active_atoms, active_i, min_diff, min_diff_unit, mode, label)
            else:
                if self.config["output"]["plot"] > 1:
                    print("\rNot saving spike {}/{}".format(seg_nb, total_number_of_spikes), end="")
        if self.config["output"]["verbose"] > 0:
            print()

        spikes_sorting.sort_values(by=["amplitude"], inplace=True)
        if self.config["output"]["verbose"] > 1:
            print(spikes_sorting.tail(20))

        sub_spike_sortings = []

        if mode == "whole":
            for unit in spikes_sorting.unit_index.unique():
                sub_spike_sorting = spikes_sorting[spikes_sorting.unit_index == unit]
                diffs = np.diff(sub_spike_sorting.amplitude.values)
                diffs = np.insert(diffs, 0, 0)
                diffs_norm = diffs/sub_spike_sorting.amplitude.values
                sub_spike_sorting["diff"] = diffs_norm

                rate = self.config["model"]["cdl"]["rel_amp_split_test"]
                idx_amp_min = sub_spike_sorting.loc[sub_spike_sorting["diff"] > rate, "amplitude"].max()
                if np.isnan(idx_amp_min):
                    idx_amp_min = 0
                sub_spike_sorting = sub_spike_sorting[sub_spike_sorting["amplitude"] > idx_amp_min]
                sub_spike_sortings.append(sub_spike_sorting)

            if self.config["output"]["verbose"] > 1:
                print(spikes_sorting.tail(50))

            spikes_sorting = pd.concat(sub_spike_sortings)            

        spikes_sorting["sample_index"] = spikes_sorting["sample_index"].astype(int)
        spikes_sorting["unit_index"] = spikes_sorting["unit_index"].astype(int)

        sorting_cdlgr = si.NumpySorting.from_times_labels(spikes_sorting.sample_index.values, spikes_sorting.unit_index.values, sampling_frequency=self.dictionary.dataset.recording.get_sampling_frequency())
        if self.config["output"]["verbose"] > 1:
            print(sorting_cdlgr.to_spike_vector())
    
        length_ms = self.config["dataset"].get("sources", {}).get("length_ms", None)
        delta_time = length_ms if length_ms is not None else 4  # in ms
        fs = self.config["dataset"].get("fs", None)
        fs = fs if fs is not None else self.dictionary.dataset.recording.get_sampling_frequency()
        cmp = sc.compare_sorter_to_ground_truth(sorting_true, sorting_cdlgr, exhaustive_gt=True, delta_time=delta_time, sampling_frequency=fs)
        if self.config["output"]["verbose"] > 0:
            print("Confusion matrix:")
            print(cmp.get_confusion_matrix())
            print()
            cmp.print_summary()
            cmp.print_performance()
            print()

        perf = cmp.get_performance()
        perf.to_csv(f"perf-{label}.csv")

        if self.config["output"]["plot"] > 1:
            plt.close('all')
            fig1, ax1 = plt.subplots()
            perf2 = pd.melt(perf, var_name='measurement')
            ax1 = sns.swarmplot(data=perf2, x='measurement', y='value', ax=ax1)
            ax1.set_xticklabels(labels=ax1.get_xticklabels(), rotation=45)
            fig1.tight_layout()
            fig1.savefig(f"perf-{label}.png")

            plt.figure()
            sw.plot_agreement_matrix(cmp, ordered=True)
            plt.tight_layout()
            plt.savefig(f"agreement_matrix-{label}.png")

            plt.figure()
            sw.plot_confusion_matrix(cmp)
            plt.tight_layout()
            plt.savefig(f"confusion_matrix-{label}.png")
        
        # Find true and false positives
        def plot_one_firing(firings, ftype):
            found = False
            for seg_nb, seg_idx in enumerate(firings.keys()):
                for unit in firings[seg_idx].keys():
                    if len(firings[seg_idx][unit]) > 0:
                        firing = firings[seg_idx][unit][0]
                        sample_win = int(self.config["output"]["fp_threshold_ms"]/1000 * fs)
                        plot_firing(traces_seg, seg_idx, reconstructed_final, seg_nb, unit, firing["firing_idx"], firing["closest_atom"], sample_win, ftype, label)
                        found = True
                        if self.config["output"]["verbose"] > 0:
                            print(f"{ftype} spike found.")
                        break
                if found:
                    break
            if not found and self.config["output"]["verbose"] > 0:
                print(f"No {ftype} spike found.")
        
        if self.config["output"]["plot"] > 1:
            if self.config["output"]["verbose"] > 0:
                print("Finding good detections and false positives...")
            true_positives, false_positives = self.find_good_and_bad_firings(traces_seg, sparse_coeffs, sorting_true, mode)
            plot_one_firing(true_positives, "TP")
            plot_one_firing(false_positives, "FP")
            if self.config["output"]["verbose"] > 0:
                print()
            
        return reconstructed
    
    def find_good_and_bad_firings(self, traces_seg, sparse_coeffs, sorting_true, mode="split"):
        """
        Find well detected spikes and false positives
        
        Inputs
        ======
        traces_seg: dictionary
            Each key represents a segment
            The corresponding value is a 1-D array
        sparse_coeffs: dictionary
            Each key represents the start sample of a segment
            The corresponding value is a sparse code; see code_sparse
        interpolated_dict: 2-D array
            Each column represents a filter
        sorting_true: spikeinterface.SortingExtractor
            Ground truth sorting
        mode: string
            split: split the signal into segments
            whole: reconstruct the whole signal
        
        Returns
        =======
        true_positives: dictionary
            Each key represents a segment.
            The corresponding value is a dictionary.
                Each key represents a unit.
                The corresponding value is the list of dictionaries (firing_sample_idx, closest_unit) for each good detection of that unit.
        false_positives: dictionary
            Each key represents a segment.
            The corresponding value is a dictionary.
                Each key represents a unit.
                The corresponding value is the list of (firing_sample_idx, closest_unit) for each false positive of that unit.
        """
        seg_size = traces_seg[list(traces_seg.keys())[0]].shape[0]
        fs = self.config["dataset"].get("fs", None)
        fs = fs if fs is not None else self.dictionary.dataset.recording.get_sampling_frequency()

        false_positives = {}
        true_positives = {}
        
        for seg_idx in sparse_coeffs.keys():
            false_positives[seg_idx] = {i:[] for i in range(self.dictionary.num_elements)}
            true_positives[seg_idx] = {i:[] for i in range(self.dictionary.num_elements)}
            for atom_i in sparse_coeffs[seg_idx].keys():
                atom = atom_i // self.interpolate if self.interpolate != 0 else atom_i
                for firing_idx in sparse_coeffs[seg_idx][atom_i]["idx"]:
                    if firing_idx > seg_size:
                        warnings.warn(f"idx {firing_idx} larger than seg_size {seg_size}")
                        continue
                    if mode == "whole":
                        idx = firing_idx
                    elif mode == "split":
                        idx = seg_idx + firing_idx
                    else:
                        raise ValueError("Mode not supported")
                    spike_idx = idx
                    min_diff, min_diff_unit = self.get_distance_to_min_diff_unit(spike_idx, sorting_true)
                    if min_diff > (self.config["output"]["fp_threshold_ms"]/1000 * fs):
                        min_diff, min_diff_unit = None, None
                    if (min_diff_unit != atom):
                        false_positives[seg_idx][atom].append({
                            "firing_idx": firing_idx,  # Firing sample idx within the segment
                            "closest_atom": min_diff_unit
                            })
                    else:
                        true_positives[seg_idx][atom].append({
                            "firing_idx": firing_idx,  # Firing sample idx within the segment
                            "closest_atom": min_diff_unit
                            })
                     
        return true_positives, false_positives
                
    def code_sparse(self, dense_coeffs, interpolated_dict):
        """
        Sparse representation of the dense coeffs

        Inputs
        ======
        dense_coeffs: array_like. (numOfelements * clen)
            This array contains many zero elements

        Outputs
        =======
        sparse_coeffs: dictionary
            Each key represents a filter
            The corresponding value is a 2-D array
                First row: Nonzero indices
                Second row: Ampltiudes corresponding to nonzero indices
        """
        numOfelements = interpolated_dict.shape[1]

        sparse_coeffs = {}
        clen = len(dense_coeffs) // numOfelements

        for fidx in np.arange(numOfelements):
            indices = np.nonzero(dense_coeffs[fidx * clen : (fidx + 1) * clen])[0]

            temp = {}
            # If no nonzero components
            if len(indices) == 0:
                temp["idx"] = np.array([], dtype=int)
                temp["amp"] = np.array([])
                sparse_coeffs[fidx] = temp
            else:
                temp["idx"] = indices
                temp["amp"] = dense_coeffs[indices + fidx * clen]
                sparse_coeffs[fidx] = temp

        return sparse_coeffs

    def terminate_csc(self, numOfiter, numOfmaxcoeffs, err_residual, err_bound):
        return (err_residual < err_bound) or (numOfiter >= numOfmaxcoeffs)

    def csc(self, y_seg, dictionary, sparsity=None, err=None, boundary=True):
        """
        Given data segment, extract convolutional codes

        Inputs
        ======
        y_seg: A segment of data
        boundary: boolean (default=1)
                If 1, accounts for truncated templates as well (clen = slen + dlen - 1)
                If 0, accounts for whole templates only (cldn = slen - dlen + 1)

        """
        assert (
            len(np.where(abs(np.linalg.norm(dictionary, axis=0) - 1) > 1e-6)[0]) == 0
        ), "Not normalized"

        numOfsamples, numOfelements = dictionary.shape

        slen = len(y_seg)
        clen = slen + numOfsamples - 1
        coeffs = np.zeros(clen * numOfelements)

        numOfmaxcoeffs = self.sparsity_tol if sparsity is None else sparsity
        err_bound = self.error_tol if err is None else err

        chosen_vals = np.zeros(numOfelements)
        chosen_idx = np.zeros(numOfelements, dtype=np.int32)

        residual = y_seg
        err_residual = np.linalg.norm(residual) / np.sqrt(np.size(residual))

        # Dictionary to collect expanding set of dictionary
        temp_idx = np.zeros(numOfmaxcoeffs, dtype=np.int32)
        dictionary_active = np.zeros((slen, numOfmaxcoeffs))

        iternum = 0
        lower_mat = [1]

        while not self.terminate_csc(iternum, numOfmaxcoeffs, err_residual, err_bound):
            #######################
            # Selection step
            #
            # This step can be fruther sped up with careful book-keeping of residuals
            #######################
            for idx in np.arange(numOfelements):
                d = dictionary[:, idx]
                cross = abs(scipy.signal.correlate(residual, d, mode="full"))

                if boundary:
                    t_start = int(np.floor(numOfsamples / 2))
                    t_end = slen + t_start
                else:
                    t_start = numOfsamples - 1
                    t_end = slen - t_start

                cross = cross / self.computeNorm(d, slen)
                m = np.argmax(cross[t_start:t_end]) + t_start

                chosen_idx[idx] = m
                chosen_vals[idx] = cross[m]

            filter_idx = np.argmax(
                chosen_vals
            )  # Returns the filter with the highest inner product
            coeff_idx = chosen_idx[filter_idx]  # index within the chosen filter
            if self.config["output"]["verbose"] > 1:
                print("Choice", chosen_vals[filter_idx], filter_idx, coeff_idx)

            #######################
            # Projection step
            #######################

            # placeholder for coefficients
            temp_idx[iternum] = filter_idx * clen + coeff_idx

            if coeff_idx < numOfsamples - 1:  # Boundary condition
                offset = coeff_idx + 1
                elem = dictionary[-offset:, filter_idx] / np.linalg.norm(
                    dictionary[-offset:, filter_idx]
                )
                elem = np.pad(elem, (0, slen - len(elem)), "constant")
            elif coeff_idx > slen - 1:  # Boundary condition
                offset = numOfsamples - (coeff_idx - (slen - 1))
                elem = dictionary[:offset, filter_idx] / np.linalg.norm(
                    dictionary[:offset, filter_idx]
                )
                elem = np.pad(elem, (slen - len(elem), 0), "constant")
            else:  # Valid correlation. Entire support of the dictionary lies within the signal
                start_idx = coeff_idx - (numOfsamples - 1)
                elem = dictionary[:, filter_idx] / np.linalg.norm(
                    dictionary[:, filter_idx]
                )
                elem = np.pad(
                    elem, (start_idx, slen - numOfsamples - start_idx), "constant"
                )

            dictionary_active[:, iternum] = elem
            [lower_mat, sparse_code] = self.sc_cholesky(
                lower_mat, dictionary_active[:, :iternum], elem, y_seg
            )

            residual = y_seg - np.matmul(
                dictionary_active[:, : iternum + 1], sparse_code
            )
            coeffs[temp_idx[: iternum + 1]] = sparse_code

            iternum += 1
            # print("CSC", iternum, err_residual)

            err_residual = np.linalg.norm(residual) / np.sqrt(np.size(residual))
        # print(coeffs)
        # print(coeffs.shape)
        return coeffs

    def computeNorm(self, delem, slen):
        """
        Compute norm of the all possible timeshifts of the dictionary

        Inputs
        ======
        delem: array-like. dictionary element

        """
        numOfsamples = delem.shape[0]
        clen = slen + numOfsamples - 1
        norms = np.zeros(clen)
        for idx in np.arange(clen):
            if idx < numOfsamples - 1:
                norms[idx] = np.linalg.norm(delem[-(idx + 1) :], 2)
            elif idx > slen - 1:
                dlen = numOfsamples - (idx - (slen - 1))
                norms[idx] = np.linalg.norm(delem[:dlen], 2)
            else:
                norms[idx] = 1

        return norms

    def sc_cholesky(self, lower, d, newelem, sig):
        """
        Efficient implementation of the least squares step to compute sparse code using Cholesky decomposition

        Inputs
        ======

        lower: lower triangular matrix
        d: current dictionary (set of regressors)
        newelem: new dictionary element to be added
        sig: signal to regress against

        Outputs
        =======

        lower_new: newly computed lower triangluar matrix
        sparse_code: sparse code for the newly updated dictionary

        """

        dim = np.shape(lower)[0]

        if d.size == 0:
            lower_new = lower  # assuming lower is just 1
            sparse_code = [np.matmul(np.transpose(newelem), sig)]
        else:
            if dim == 1:
                lower = [lower]

            temp = np.matmul(np.transpose(d), newelem)
            tempvec = scipy.linalg.solve_triangular(lower, temp, lower=True)

            ###################################
            # Construct lower triangular matrix
            ###################################
            lower_new = np.zeros((dim + 1, dim + 1))
            lower_new[:dim, :dim] = lower
            lower_new[dim, :dim] = np.transpose(tempvec)
            lower_new[dim, dim] = np.sqrt(1 - np.matmul(np.transpose(tempvec), tempvec))

            d_new = np.zeros((d.shape[0], d.shape[1] + 1))
            d_new[:, : d.shape[1]] = d
            d_new[:, -1] = newelem

            temp = np.matmul(np.transpose(d_new), sig)
            sparse_code = scipy.linalg.cho_solve([lower_new, 1], temp)

        return lower_new, sparse_code
