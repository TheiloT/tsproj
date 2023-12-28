import numpy as np
import scipy
from matplotlib import pyplot as plt
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from cdlgr.model.dictionary import Dictionary
import spikeinterface.comparison as sc
import pandas as pd
from tqdm import tqdm
import warnings

class CDL:
    def __init__(self, dictionary, config):
        self.config = config
        self.dictionary: Dictionary = dictionary

    def split_traces(self):
        print("Splitting traces...")                
        self.channel = self.config["dataset"]["channel"]

        traces = self.dictionary.dataset.recording.get_traces()[:, self.channel]
        
        plt.figure()
        plt.plot(traces)
        plt.savefig("traces.png")

        if self.config["dataset"]["window"]["split"]:
            peaks = detect_peaks(self.dictionary.dataset.recording,
                                random_chunk_kwargs={'chunk_size':min(10000, 
                                                self.dictionary.dataset.recording.get_num_frames() - 5)})#, detect_threshold=5, n_shifts=5, peak_span_ms=0.5, peak_span_samples=None, filter=None, filter_kwargs=None, return_idxs=True, return_times=False, return_peak_span=False, return_channel_idxs=False, verbose=False

            peaks = pd.DataFrame(peaks)
            peaks = peaks[peaks["channel_index"] == self.channel]
            # peaks["sample_index"] = peaks["sample_index"].astype(int)
            print(peaks.shape)
            print(peaks)

            peak_size = 110 # even
            peak_size = int(self.config["dataset"]["window"]["window_size_s"] * self.dictionary.dataset.recording.get_sampling_frequency())
            half_size = peak_size // 2
            # traces_seg = np.zeros((peaks.shape[0], peak_size))
            # print(traces_seg.shape)
            traces_seg = {}
            plt.figure()
            for i, (_, peak) in enumerate(peaks.iterrows()):
                peak_idx = int(peak["sample_index"])
                idx = peak_idx
                traces_seg[idx] = np.zeros(peak_size)
                print(peak_idx)
                if peak_idx - half_size < 0:
                    traces_seg[idx][half_size - peak_idx:] = traces[:peak_idx + half_size]
                elif peak_idx + half_size > traces.shape[0]:
                    traces_seg[idx][:traces.shape[0] - (peak_idx - half_size)] = traces[peak_idx - half_size:]
                else:
                    traces_seg[idx][:] = traces[peak_idx - half_size : peak_idx + half_size]
                plt.plot(traces_seg[idx][:])
            plt.savefig("traces_seg.png")
        else:
            traces_seg = {}
            traces_seg[0] = traces
            warnings.warn("Performance evaluation only works with window split")
            input()

        return traces_seg

    def run(self, traces_seg):
        print("Running CDL...")
        self.num_iterations = self.config["model"]["cdl"]["num_iterations"]
        self.interpolate = self.config["model"]["cdl"]["interpolate"]
        
        self.sparsity_tol = self.config["model"]["cdl"]["sparsity_tol"]
        self.error_tol = self.config["model"]["cdl"]["error_tol"]
        # exit()

        for i in range(self.num_iterations):
            print(f"Iteration {i+1}/{self.num_iterations}")
            self.dictionary.plot(i)
            interpolated_dict, interpolator = self.dictionary.interpolate(self.interpolate, kind='sinc')
            # print(interpolated_dict.shape)
            # sparse_coeffs = self.csc(traces, self.dictionary.dictionary, sparsity=None, boundary=False)
            sparse_coeffs = {}
            for j in tqdm(traces_seg.keys()):
                sparse_coeffs[j] = self.code_sparse(
                    self.csc_old(
                        traces_seg[j], interpolated_dict, sparsity=None, boundary=True
                    ), interpolated_dict
                )
            print(sparse_coeffs)
            if i != self.num_iterations - 1:
                self.dictionary.update(traces_seg, sparse_coeffs, interpolator)

        self.reconstruct(traces_seg, sparse_coeffs, interpolated_dict)

    def reconstruct(self, traces_seg, sparse_coeffs, interpolated_dict):
        # assert self.interpolate == 0, "Reconstruction only works for non-interpolated dictionaries"
        

        spikes = pd.DataFrame(self.dictionary.dataset.sorting_true.to_spike_vector(concatenated=True))
        total_number_of_spikes = spikes.shape[0]
        stat_per_unit = {}
        expected_stats = {}

        def get_unit(k_idx):
            print(k_idx)
            diff = np.abs(spikes.sample_index - k_idx)
            unit_row = np.argmin(diff)
            val_min = diff[unit_row]
            if val_min > 6:
                print(val_min)
                raise Exception(f"Could not find unit for {k_idx}")
            return spikes.unit_index.values[unit_row]
            # unit = spikes[(spikes.sample_index == k_idx) | 
            #               (spikes.sample_index == k_idx + 1) | 
            #               (spikes.sample_index == k_idx - 1)].unit_index.values[0]
        
        print(spikes)
        print(traces_seg.keys())
        print(get_unit(170))

        for unit in np.unique(spikes.unit_index.values):
            stat_per_unit[unit] = {"tp": 0, "fp": 0, "fn": 0}
            expected_stats[unit] = len(spikes[spikes.unit_index == unit])

        spikes_sorting = pd.DataFrame(columns=["sample_index", "unit_index"])


        # print(spikes)
        # print(traces_seg.keys())

        N_seg = len(traces_seg.keys())
        seg_size = traces_seg[list(traces_seg.keys())[0]].shape[0]
        print(N_seg, seg_size)
        reconstructed = np.zeros((N_seg, seg_size + interpolated_dict.shape[0] - 1))
        reconstructed_final = np.zeros((N_seg, seg_size))
        for k_idx, k in enumerate(sparse_coeffs.keys()): 
            active_i = []
            for i in sparse_coeffs[k].keys():            
                for j, idx in enumerate(sparse_coeffs[k][i]["idx"]):
                    if i not in active_i:
                        active_i.append(i)
                    reconstructed[k_idx, idx : idx + len(interpolated_dict[:, i])] += (
                        sparse_coeffs[k][i]["amp"][j] * interpolated_dict[:, i]
                    )
            reconstructed_final[k_idx, :] = reconstructed[k_idx, :][interpolated_dict.shape[0] - 1:]

            active_atoms = []
            for i in active_i:
                if self.interpolate != 0:
                    active_atoms.append(i//self.interpolate)
                else:
                    active_atoms.append(i) 

                spikes_sorting.loc[k_idx] = [k, active_atoms[-1]]

            unit = get_unit(k)
            if unit in active_atoms:
                stat_per_unit[unit]["tp"] += 1
                
            plt.figure()
            plt.plot(traces_seg[k][:], label="original")
            plt.plot(reconstructed_final[k_idx, :], label="reconstructed")
            plt.legend()
            plt.title(f"Active filters: {active_atoms} - interpolated {active_i}")
            plt.savefig(f"reconstructed_{k_idx}_{k}.png")
            plt.close()

        print(spikes_sorting)
        print(stat_per_unit)
        print(expected_stats)
        print(total_number_of_spikes)

        sorting_cdlgr = si.NumpySorting.from_times_labels(spikes_sorting.sample_index.values, spikes_sorting.unit_index.values, sampling_frequency=self.dictionary.dataset.recording.get_sampling_frequency())
        print(sorting_cdlgr.to_spike_vector())

        cmp = sc.compare_sorter_to_ground_truth(self.dictionary.dataset.sorting_true, sorting_cdlgr, exhaustive_gt=True)
        print(cmp.get_confusion_matrix())
        cmp.print_summary()
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
        plt.show()


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


        return sparse_coeffs

    def terminate_csc(self, numOfiter, numOfmaxcoeffs, err_residual, err_bound):
        return (err_residual < err_bound) or (numOfiter >= numOfmaxcoeffs)

    def csc(self, y_seg, dictionary, sparsity=None, err=None, boundary=True):
        num_elements = dictionary.shape[1]
        num_samples = dictionary.shape[0]
        N = y_seg.shape[0]

        zs = []
        ks = np.zeros((num_samples, N))  # Track used atoms

        def build_atom(k, delay):  # Build the shifted template of length N
            dk = np.zeros((N))
            dk[delay : delay + len(dictionary[:, k])] = dictionary[:, k]
            return dk

        residual = y_seg.copy()
        for _ in range(4):  # self.sparsity_tol):
            chosen_vals = np.zeros(num_elements)
            chosen_idx = np.zeros(num_elements, dtype=np.int32)
            for i in range(num_elements):
                atom = dictionary[:, i]
                corr = np.correlate(residual, atom, "valid")
                corr[ks[i, : len(corr)] == 1] = 0  # ignore already used atoms
                argmax = np.argmax(np.abs(corr))
                prod_scal = corr[argmax]
                # if np.abs(prod_scal) > np.abs(best_val):
                #     best_k, best_val = (k, argmax, prod_scal), prod_scal
                chosen_idx[i] = argmax
                chosen_vals[i] = prod_scal

            filter_idx = np.argmax(
                chosen_vals
            )  # Returns the filter with the highest inner product
            coeff_idx = chosen_idx[filter_idx]  # index within the chosen filter
            val_dix = chosen_vals[filter_idx]
            ks[
                filter_idx, coeff_idx
            ] = 1  # check coeff idx not larger than len(atoms) ?

            zs.append((filter_idx, coeff_idx, val_dix))

            sub_space_matrix = np.zeros((N, len(zs)))
            xp = np.zeros(len(zs))
            for i, (k2, delay2, prod_scal2) in enumerate(zs):
                sub_space_matrix[:, i] = build_atom(k2, delay2)
                xp[i] = prod_scal2
            inv_mat = np.linalg.pinv(sub_space_matrix.T @ sub_space_matrix)
            projection = sub_space_matrix @ inv_mat @ sub_space_matrix.T @ y_seg.copy()

            residual = y_seg.copy() - projection

        coeffs = {}
        for i in range(num_elements):
            coeffs[i] = {"idx": [], "amp": []}
        for i, (k, delay, prod_scal) in enumerate(zs):
            coeffs[k]["idx"].append(delay)
            coeffs[k]["amp"].append(prod_scal)
        for i in range(num_elements):
            coeffs[i]["idx"] = np.array(coeffs[i]["idx"], dtype=int)
            coeffs[i]["amp"] = np.array(coeffs[i]["amp"])

        return coeffs

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

        err_residual = np.linalg.norm(residual)
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
