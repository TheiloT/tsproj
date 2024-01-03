from time import perf_counter
import numpy as np
import scipy
from matplotlib import pyplot as plt
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
        print("Splitting traces...")                
        self.channel = self.config["dataset"]["channel"]

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
        
        plt.figure()
        plt.plot(traces)
        plt.savefig("traces.png")

        if self.config["dataset"]["window"]["split"]:
            detect_threshold = 5
            exclude_sweep_ms = self.config["dataset"]["sources"]["length_ms"]/2 if (self.config["dataset"]["type"] == "synth") else 0.1
            peaks = detect_peaks(self.dictionary.dataset.recording, detect_threshold=detect_threshold, exclude_sweep_ms=exclude_sweep_ms,
                                 random_chunk_kwargs={'chunk_size':min(10000, self.dictionary.dataset.recording.get_num_frames() - 5)})
                                #, detect_threshold=5, n_shifts=5, peak_span_ms=0.5, peak_span_samples=None, filter=None, filter_kwargs=None, return_idxs=True, return_times=False, return_peak_span=False, return_channel_idxs=False, verbose=False
            peaks = pd.DataFrame(peaks)
            peaks = peaks[peaks["channel_index"] == self.channel]
            # peaks["sample_index"] = peaks["sample_index"].astype(int)
            # print(peaks.shape)
            # print(peaks)

            peak_size = 110 # even
            peak_size = int(self.config["dataset"]["window"]["window_size_s"] * self.dictionary.dataset.recording.get_sampling_frequency())
            half_size = peak_size // 2
            # traces_seg = np.zeros((peaks.shape[0], peak_size))
            # print(traces_seg.shape)
            traces_seg = {}
            recording_length = self.dictionary.dataset.recording.get_num_frames()
            plt.figure()
            for i, (_, peak) in tqdm(enumerate(peaks.iterrows())):
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
                plt.plot(traces_seg[idx][:])
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
                    print(perm)
                    print(error)
                    if error < best_total_error:
                        best_total_error = error
                        best_perm = perm
                self.dictionary.dictionary = self.dictionary.dictionary[:, best_perm]
                print(best_perm)

                self.dictionary.normalize()
        else:
            traces_seg = {}
            traces_seg[0] = get_frames()
            warnings.warn("Performance evaluation only works with window split")
            input()

        time_preprocessing_end = perf_counter()
        print("Preprocessing time: ", time_preprocessing_end - time_preprocessing_begin)
        np.savetxt("time_preprocessing.txt", [time_preprocessing_end - time_preprocessing_begin], fmt="%f")

        return traces_seg
    
    @property
    def interpolator_type(self):
        return self.config["model"]["cdl"]["interpolator_type"]

    def train(self, traces_seg):
        print("Running CDL...")       

        self.dictionary.recovery_error(-1)
        self.dictionary.recovery_error_interp(-1, self.interpolate)
        # exit()

        time_total_begin = perf_counter()
        time_csc = []
        time_update = []
        for i in range(self.num_iterations):
            print(f"Iteration {i+1}/{self.num_iterations}")
            self.dictionary.plot(i)
            sparse_coeffs, interpolated_dict, interpolator, time_csc_diff = self.run_csc(traces_seg)
            time_csc.append(time_csc_diff)
            
            if i != self.num_iterations - 1:
                time_update_begin = perf_counter()
                self.dictionary.update(traces_seg, sparse_coeffs, interpolator)   
                time_update_end = perf_counter()
                time_update.append(time_update_end - time_update_begin)

                error = self.dictionary.recovery_error(i)
                error2 = self.dictionary.recovery_error_interp(i, self.interpolate)
                print("Dictionary error ", error, error2)

        time_total_end = perf_counter()

        print("Total time: ", time_total_end - time_total_begin)
        print("CSC time: ", np.sum(time_csc))
        print("Update time: ", np.sum(time_update))

        np.savez("time.npz", time_csc=time_csc, time_update=time_update, time_total=time_total_end - time_total_begin)

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
        for j in tqdm(traces_seg.keys()):
            sparse_coeffs[j] = self.code_sparse(
                    self.csc_old(
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

    def reconstruct(self, traces_seg, sparse_coeffs, interpolated_dict, sorting_true, mode="split", label="train"):       
        spikes = pd.DataFrame(sorting_true.to_spike_vector(concatenated=True))
        print(spikes)
        total_number_of_spikes = spikes.shape[0]
      
        spikes_sorting = pd.DataFrame(columns=["sample_index", "unit_index", "amplitude", "error"])

        N_seg = len(traces_seg.keys())
        seg_size = traces_seg[list(traces_seg.keys())[0]].shape[0]

        reconstructed = np.zeros((N_seg, seg_size + interpolated_dict.shape[0] - 1))
        reconstructed_final = np.zeros((N_seg, seg_size))
        for k_idx, k in enumerate(sparse_coeffs.keys()): 
            active_i = []
            idxes = []
            amps = []
            for i in sparse_coeffs[k].keys():            
                for j, idx in enumerate(sparse_coeffs[k][i]["idx"]):
                    if idx > seg_size:
                        warnings.warn(f"idx {idx} larger than seg_size {seg_size}")
                        print("WARNING segment_size")
                        continue
                    if i not in active_i or mode != "split":
                        active_i.append(i)    
                        idxes.append(idx)
                        amps.append(sparse_coeffs[k][i]["amp"][j])
                    else:
                        print("Atom already active")                    
                    # print((sparse_coeffs[k][i]["amp"][j] * interpolated_dict[:, i]).shape)
                    # print(k_idx, idx, idx + len(interpolated_dict[:, i]))
                    # print(reconstructed.shape)
                    # print(interpolated_dict.shape)
                    # print(reconstructed[k_idx, idx : idx + len(interpolated_dict[:, i])].shape)
                    reconstructed[k_idx, idx : idx + len(interpolated_dict[:, i])] += (
                        sparse_coeffs[k][i]["amp"][j] * interpolated_dict[:, i]
                    )
            reconstructed_final[k_idx, :] = reconstructed[k_idx, :][interpolated_dict.shape[0] - 1:]

            active_atoms = []
            for idx, i in enumerate(active_i):
                if self.interpolate != 0:
                    active_atoms.append(i//self.interpolate)
                else:
                    active_atoms.append(i) 

                if mode == "split":
                    spikes_sorting.loc[k_idx] = [k, active_atoms[-1], amps[idx], 0]
                else:
                    reconstructed_part = reconstructed[k_idx, idxes[idx]:idxes[idx]+self.dictionary.element_length*2]
                    original_part = traces_seg[k][idxes[idx]-self.dictionary.element_length:idxes[idx]+self.dictionary.element_length]
                    # error = np.linalg.norm(reconstructed_part - original_part)/np.linalg.norm(original_part)
                    # error = reconstructed_part.dot(original_part)/(np.linalg.norm(reconstructed_part)*np.linalg.norm(original_part))
                    normalized_reconstructed = reconstructed_part/np.linalg.norm(reconstructed_part)
                    normalized_original = original_part/np.linalg.norm(original_part)

                    max_length = min(len(normalized_reconstructed), len(normalized_original))
                    normalized_reconstructed = normalized_reconstructed[:max_length]
                    normalized_original = normalized_original[:max_length]

                    error = np.sqrt(1-np.dot(normalized_reconstructed, normalized_original)**2)
                    # error = np.linalg.norm(reconstructed_part - original_part)/np.linalg.norm(original_part)
                    # from dtaidistance import dtw
                    # error = dtw.distance(reconstructed_part, original_part)
                    # if mode != "split":
                    #     plt.figure()
                    #     plt.plot(normalized_reconstructed, label="reconstructed")
                    #     plt.plot(normalized_original, label="original")
                    #     plt.legend()
                    #     plt.title(f"Active filters: {active_atoms} - interpolated {active_i}")
                    #     plt.show()
                    spikes_sorting.loc[len(spikes_sorting)] = [idxes[idx]-self.dictionary.element_length//2, active_atoms[-1], amps[idx], error]
                
            spike_idx = spikes_sorting["sample_index"].values[-1]
            min_diff = np.inf
            min_diff_unit = None
            for unit in sorting_true.unit_ids:
                spikes_idxes = sorting_true.get_unit_spike_train(unit_id=unit)
                min_diff_true = np.abs(spikes_idxes - spike_idx).min()
                if min_diff_true < min_diff:
                    min_diff = min_diff_true
                    min_diff_unit = unit
                    
            if (k_idx < 10) or min_diff > 15:

                plt.close('all')
                plt.figure()
                plt.plot(traces_seg[k][:], label="original", marker="x")
                plt.plot(reconstructed_final[k_idx, :], label="reconstructed", marker="+")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude (a. u.)")
                plt.legend()
                plt.title(f"Active filters: {active_atoms} - interpolated {active_i}, unit {min_diff_unit}: dist {min_diff}")
                if len(traces_seg[k][:]) > 1000:
                    plt.show()
                plt.savefig(f"reconstructed_{k_idx}_{k}_{label}.png")
                plt.close()
            else:
                print("\rNot saving spike {}/{}".format(k_idx, total_number_of_spikes), end="")
        print()

        spikes_sorting.sort_values(by=["amplitude"], inplace=True)
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

            print(spikes_sorting.tail(50))

            spikes_sorting = pd.concat(sub_spike_sortings)            

        # spikes_sorting.drop(labels="amplitude", axis=1, inplace=True)
        spikes_sorting["sample_index"] = spikes_sorting["sample_index"].astype(int)
        spikes_sorting["unit_index"] = spikes_sorting["unit_index"].astype(int)

        # plt.close()
        # plt.figure()
        # plt.hist(spikes_sorting.amplitude, bins=100, density=True)
        # plt.show()

        sorting_cdlgr = si.NumpySorting.from_times_labels(spikes_sorting.sample_index.values, spikes_sorting.unit_index.values, sampling_frequency=self.dictionary.dataset.recording.get_sampling_frequency())
        print(sorting_cdlgr.to_spike_vector())
    
        length_ms = self.config["dataset"].get("sources", {}).get("length_ms", None)
        delta_time = length_ms if length_ms is not None else 4  # in ms
        fs = self.config["dataset"].get("fs", None)
        cmp = sc.compare_sorter_to_ground_truth(sorting_true, sorting_cdlgr, exhaustive_gt=True, delta_time=delta_time, sampling_frequency=fs)
        print(cmp.get_confusion_matrix())
        cmp.print_summary()
        cmp.print_performance()

        print(cmp.match_event_count)
        print(cmp.match_score)

        # print(sorting_cdlgr.get_unit_ids())
        # print(sorting_true.get_unit_ids())
        # for row in cmp.match_event_count.iterrows():
        #     sorting_true_id = row[1].name
        #     sorting_cdlgr_id = row[1].values.argmax()
        #     print(row[1].name, row[1].values.argmax())
        #     firings_cdlgr = sorting_cdlgr.get_unit_spike_train(sorting_cdlgr_id)
        #     firings_true = sorting_true.get_unit_spike_train(sorting_true_id)
        #     for firing_cdlgr in firings_cdlgr:
        #         # find closest firings_true
        #         idx = np.abs(firings_true - firing_cdlgr).argmin()
        #         print(firings_true[idx], firing_cdlgr)
        #         if np.abs(firings_true[idx] - firing_cdlgr) > 10:
        #             # firings_true = np.delete(firings_true, idx)
        #             print("Spike not matching")
        #             template_true = sorting_true.get_all_templates([sorting_true_id])[0]

        #             spikes_unit_cdlgr = pd.DataFrame(sorting_cdlgr.get_unit_spike_train(sorting_cdlgr_id))

           
            


        # reconstructed = np.zeros(traces.shape[0] + interpolated_dict.shape[0] - 1)
        # print(reconstructed.shape)
        # for j, idx in enumerate(sparse_coeffs[i]["idx"]):
        #     print(idx, idx + len(interpolated_dict[:, i]))
            
        #     reconstructed[idx : idx + len(interpolated_dict[:, i])] += (
        #         (sparse_coeffs[i]["amp"][j] * interpolated_dict[:, i])#[:traces.shape[0] - idx]
        #     )
        perf = cmp.get_performance()
        plt.close('all')
        fig1, ax1 = plt.subplots()
        perf2 = pd.melt(perf, var_name='measurement')
        import seaborn as sns
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

        # optimization: add term to have different templates

        # reconstructed = reconstructed[interpolated_dict.shape[0] - 1:]
        
        # plt.figure()
        # plt.plot(traces, label="original")
        # plt.plot(reconstructed, label="reconstructed")
        # plt.legend()
        # plt.savefig("reconstructed.png")

        return reconstructed


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
        # print(len(dense_coeffs))
        # print(numOfelements)
        # print(clen)
        for fidx in np.arange(numOfelements):
            indices = np.nonzero(dense_coeffs[fidx * clen : (fidx + 1) * clen])[0]
            # print(indices)

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

        # print(sparse_coeffs)

        return sparse_coeffs

    def terminate_csc(self, numOfiter, numOfmaxcoeffs, err_residual, err_bound):
        return (err_residual < err_bound) or (numOfiter >= numOfmaxcoeffs)

    def csc_old(self, y_seg, dictionary, sparsity=None, err=None, boundary=True):
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
