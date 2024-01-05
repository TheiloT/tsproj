import numpy as np
import matplotlib.pyplot as plt
import scipy
import spikeinterface.full as si
from cdlgr.dataset.dataset import Dataset
from cdlgr.outputs.plot import plot_template_and_truth, plot_template_and_truth_interp, plot_templates

class Dictionary:
    def __init__(self, dataset, config):
        self.dataset: Dataset = dataset
        self.config = config

        self.fs = dataset.recording.get_sampling_frequency()
        self.element_length = int(config["model"]["dictionary"]["element_length_ms"]*self.fs/1000)
        self.num_elements = config["model"]["dictionary"]["num_elements"]

        if config["output"]["verbose"] > 0:
            print(f"Creating dictionary with {self.num_elements} elements of length {self.element_length} samples ({self.element_length/self.fs*1000} ms).")
        np.random.seed(0)
        self.dictionary = np.random.rand(self.element_length, self.num_elements)
        self.dictionary /= np.linalg.norm(self.dictionary, axis=0)

    def save(self):
        np.savez("dictionary.npz", dictionary=self.dictionary,
                 true_dictionary=self.true_dictionary,
                 fs=self.fs)

    def initialize(self):
        if self.config["output"]["verbose"] > 0:
            print("Initializing dictionary...")

        self.channel = self.config["dataset"]["channel"]
        length_ms = self.config["dataset"].get("sources", {}).get("length_ms", None)
        ms_before = length_ms/2 if length_ms is not None else 1.0
        ms_after = length_ms/2 + 1000/self.fs if length_ms is not None else 2.0
        wv = si.extract_waveforms(self.dataset.recording, self.dataset.sorting_true, max_spikes_per_unit=2500, ms_before=ms_before, ms_after=ms_after, mode="memory")
        templates = wv.get_all_templates()  # Shape (num_units, num_samples, num_channels)

        # remove code duplication
        self.true_dictionary = np.zeros_like(self.dictionary)
        for i in range(templates.shape[0]):
            if self.config["dataset"]["type"] == "synth":
                self.true_dictionary[:, i] = templates[i, :, self.channel]
            else:
            # center with respect to the middle of the template
                templates_center_idx = np.argmin(templates[i, :, self.channel])
                half_length_template = min(templates_center_idx, templates.shape[1]-templates_center_idx-1)
                template_loc = templates[i, templates_center_idx-half_length_template:templates_center_idx+half_length_template+1, self.channel]
                if template_loc.shape[0] < self.element_length:
                    self.true_dictionary[:, i] = np.pad(template_loc, (0, self.element_length - template_loc.shape[0]), 'constant')
                else:
                    self.true_dictionary[:, i] = template_loc[template_loc.shape[0]//2-self.element_length//2:template_loc.shape[0]//2+self.element_length//2+1]
        self.true_dictionary /= np.linalg.norm(self.true_dictionary, axis=0)

        if self.config["output"]["plot"] > 1:
            plot_templates(templates, self.fs, self.channel, normalized=False)
        if self.config["output"]["plot"] > 0:
            plot_templates(templates, self.fs, self.channel, normalized=True)

        if self.config["model"]["dictionary"]["init_templates"] == "real":
            if self.config["output"]["verbose"] > 0:
                print("Initializing dictionary with real templates...")
            for i in range(min(self.num_elements, templates.shape[0])):
                if templates.shape[1] < self.element_length:
                    self.dictionary[:, i] = np.pad(templates[i, :, self.channel], (0, self.element_length - templates.shape[1]), 'constant')
                else:
                    self.dictionary[:, i] = templates[i, templates.shape[1]//2-self.element_length//2:templates.shape[1]//2+self.element_length//2+1, self.channel]

            self.dictionary /= np.linalg.norm(self.dictionary, axis=0)
    
    def recovery_error_interp(self, iteration, numOfsubgrids, save_plots=True):
        """

        Compute the error between the corresponding columns of the two dictionaries.

        Inputs
        ======

        dict1: dictionary 1
        dict2: dictionary 2

        Outputs
        =======

        err_distance: error distance between the filters

        """
        dict1 = self.true_dictionary
        dict2 = self.dictionary

        if dict2.shape[1] > dict1.shape[1]:
            if self.config["output"]["verbose"] > 0:
                print("Truncating dictionary")
            dict2 = dict2[:, :dict1.shape[1]]

        assert(np.shape(dict1)==np.shape(dict2)), "Dictionaries are of different dim!"
        filternum = np.shape(dict1)[1]

        offset = (self.element_length - 1) // 2
        offsets = np.arange(-offset,offset+1,dtype=int)

        err_distance = np.zeros(filternum)

        dict1 = dict1/np.linalg.norm(dict1, axis=0)
        dict2 = dict2/np.linalg.norm(dict2, axis=0)

        dico = Dictionary(self.dataset, self.config)
        dico.dictionary = dict1
        dict1_interpolated, _ = dico.interpolate(numOfsubgrids, kind=self.config["model"]["cdl"]["interpolator_type"])
        numOfinterp = int(dict1_interpolated.shape[1]/filternum)

        indices = np.zeros(filternum, dtype=int)
        for unit in np.arange(filternum):
            estimated_dict = dict2[:,unit]/np.linalg.norm(dict2[:,unit])
            best_offset, min_dist = None, np.infty
            for j in range(numOfinterp):
                interpolated_gt = dict1_interpolated[:,unit*numOfinterp+j]/np.linalg.norm(dict1_interpolated[:,unit*numOfinterp+j])
                for o in offsets:
                    dist = np.sqrt(1-np.power(np.dot(np.roll(interpolated_gt,o),estimated_dict),2))
                    if dist<min_dist:
                        min_dist = dist
                        indices[unit] = j
                        best_offset = o
            err_distance[unit] = min_dist
            
            if save_plots:
                idx = indices[unit]
                interpolated_gt = dict1_interpolated[:,unit*numOfinterp+idx]
                interpolated_gt /= np.linalg.norm(interpolated_gt)
                plot_template_and_truth_interp(estimated_dict, interpolated_gt, unit, err_distance[unit], idx, self.fs, self.config["model"]["cdl"]["interpolate"], best_offset, iteration)
                
        
        return err_distance, indices
    
    def recovery_error(self, iteration, save_plots=True):

        """
        Compute the error between the corresponding columns of the two dictionaries.

        Inputs
        ======

        dict1: dictionary 1
        dict2: dictionary 2

        Outputs
        =======

        err_distance: error distance between the filters

        """
        dict2 = self.dictionary
        dict1 = self.true_dictionary
        
        print(dict1.shape, dict2.shape)

        if dict2.shape[1] > dict1.shape[1]:
            if self.config["output"]["verbose"] > 0:
                print("Truncating dictionary")
            dict2 = dict2[:, :dict1.shape[1]]


        assert(np.shape(dict1)==np.shape(dict2)), "Dictionaries are of different dim!"
        filternum = np.shape(dict1)[1]

        err_distance = np.zeros(filternum)

        dict1 = dict1/np.linalg.norm(dict1, axis=0)
        dict2 = dict2/np.linalg.norm(dict2, axis=0)

        for i in np.arange(filternum):
            diff = 1-np.power(np.matmul(np.transpose(dict1[:,i]), dict2[:,i]),2)
            # Numerical issue
            if abs(diff)< 1e-6:
                diff = 0

            err_distance[i] = np.sqrt(diff)
            
            plot_template_and_truth(dict2[:,i], dict1[:,i], i, err_distance[i], self.fs, iteration)
                   


        return err_distance
            
    def normalize(self):
        self.dictionary /= np.linalg.norm(self.dictionary, axis=0)
   
    def getSignalIndices(self, dlen, indices):
        """
        Extract the signal for which the corresponding coefficients are non-zero

        """

        arrindices = np.zeros(dlen*np.size(indices), dtype=int)
        for i, value in enumerate(indices):
            arrindices[i*dlen:(i+1)*dlen] = np.arange(value, value+dlen)

        return arrindices

    def update(self, y_seg_set, coeffs, interpolator=None):
        assert(len(y_seg_set.keys())==len(coeffs.keys())), "The dimension of data and coeff need to match"

        d = self.dictionary
        # print("Dictionary", d.shape)

        if len(interpolator)==0:
            interpolator={}
            delta_fn = np.zeros(d.shape[0])
            delta_fn[int(d.shape[0]/2)] = 1
            interpolator[0] = delta_fn

        d_updated = np.copy(d)

        for base_fidx in np.arange(self.num_elements):
            # Variable to track signal_extracted (required also for coefficient update)
            y_extracted_set = {}
            indices_set = {}
            coeffs_set = {}
            delay_set = {}

            # Collecting the extracted data segments
            for key, y_seg in y_seg_set.items():
                slen = len(y_seg)
                clen = slen + self.element_length - 1

                numOfinterp = len(interpolator.keys())
                # print("numOfinterp", numOfinterp)

                coeffs_seg = {}
                filter_delay_indices = {}
                # Collapse the interpolated codes together
                for fidx in np.arange(self.num_elements):
                    coeffs_seg[fidx] = {'idx':np.array([], dtype=int), 'amp':np.array([])}
                    dense_code = np.zeros(clen)
                    delay_indices = -np.ones(clen, dtype=int)


                    for interp_idx in range(numOfinterp):
                        j = fidx * numOfinterp + interp_idx

                        dense_code[coeffs[key][j]['idx']] += coeffs[key][j]['amp']
                        delay_indices[coeffs[key][j]['idx']] = interp_idx

                    nonzero_indices = np.where(abs(dense_code)>1e-6)[0]

                    coeffs_seg[fidx]['idx'] = nonzero_indices
                    coeffs_seg[fidx]['amp'] = dense_code[nonzero_indices]

                    filter_delay_indices[fidx] = np.array([i for i in delay_indices if i>-1])

                ########################
                # Construct error signal and extract the segments
                ########################
                if len(coeffs_seg[base_fidx]['idx'])>0:
                    temp_indices = coeffs_seg[base_fidx]['idx']
                    # We don't want to use the templates at the boundary
                    indices = np.array([i for i in range(len(temp_indices)) if temp_indices[i]>= self.element_length-1 and temp_indices[i]<=slen-1])

                    if len(indices)>0:
                        indices_set[key] = coeffs_seg[base_fidx]['idx'][indices]
                        coeffs_set[key] = coeffs_seg[base_fidx]['amp'][indices]
                        delay_set[key] = filter_delay_indices[base_fidx][indices]

                        patch_indices = self.getSignalIndices(self.element_length, indices_set[key]) - (self.element_length - 1)
                        residual = np.copy(y_seg)
                        for fidx in np.arange(self.num_elements):
                            # Subtract the contributions from others
                            if fidx != base_fidx:
                                convolved_sig = np.zeros(clen)
                                for i, (idx, amp) in enumerate(zip(coeffs_seg[fidx]['idx'], coeffs_seg[fidx]['amp'])):
                                    if (idx >= self.element_length-1 and idx <= slen-1):	# We don't want to use the templates at the boundary
                                        mtx = self.compute_interp_matrix(interpolator[filter_delay_indices[fidx][i]], self.element_length)
                                        convolved_sig[idx : idx + self.element_length] += amp * np.matmul(mtx, d_updated[:, fidx])

                                residual -= convolved_sig[self.element_length-1:]

                        y_extracted_set[key] = residual[patch_indices].reshape((self.element_length,-1), order='F')

            if self.config["output"]["verbose"] > 0:
                print("Updating Filter {}".format(base_fidx))

            ##############################
            # Update the filters
            ##############################
            if len(y_extracted_set.keys()) > 0:

                denominator = np.zeros((self.element_length, self.element_length))
                numerator = np.zeros(self.element_length)

                for key in y_extracted_set.keys():

                    y_extracted_seg = y_extracted_set[key]

                    for i, (idx_1, coeff_1, delay_1) in enumerate(zip(indices_set[key], coeffs_set[key], delay_set[key])):
                        numerator += coeff_1 * np.matmul(np.transpose(self.compute_interp_matrix(interpolator[delay_1], self.element_length)), y_extracted_seg[:, i])

                        for idx_2, coeff_2, delay_2 in zip(indices_set[key], coeffs_set[key], delay_set[key]):
                            if abs(idx_1 - idx_2) < self.element_length:
                                denominator += coeff_1 * coeff_2 * self.compute_diff_matrix(interpolator[delay_1], interpolator[delay_2], idx_1, idx_2, self.element_length)

                elem = np.matmul(np.linalg.pinv(denominator), numerator)
                d_updated[:,base_fidx] = elem/np.linalg.norm(elem)

            elif self.config["output"]["verbose"] > 0:
                print("Non matching!")
                pass

        # return d_updated, indices_set, coeffs_set, y_extracted_set
        self.dictionary = d_updated

        return y_extracted_set

    def compute_interp_matrix(self, interpolator, dlen):
        """
        For no interpolator case, the result should just be an identity matrix

        Inputs
        ======
        interpolator: array

        """
        interplen = len(interpolator)
        assert np.mod(interplen,2)==1, "Interpolator legnth must be odd"
        assert interplen<=dlen, "Interpolator length must be less than dictionary template length"

        interpolator_flipped = np.flip(interpolator, axis=0)

        start_clip = int((dlen-1)/2)
        end_clip = start_clip + dlen
        mtx = np.zeros((dlen, 2*dlen-1))

        for idx in np.arange(dlen):
            start_idx = start_clip+idx-int(interplen/2)
            end_idx = start_idx + interplen
            mtx[idx, start_idx : end_idx] = interpolator_flipped

        shift_mat = mtx[:, start_clip:end_clip]

        return shift_mat
    
    def compute_diff_matrix(self, interpolator_i, interpolator_j, i, j, dlen):
        """
        Multiply two matrices for the denominator of the updated dictionary
        """
        mtx_i = self.compute_interp_matrix(interpolator_i, dlen)
        mtx_j = self.compute_interp_matrix(interpolator_j, dlen)

        interplen = len(interpolator_i)

        if j - i >= interplen:
            diff_matrix = np.zeros((dlen, dlen))
        else:
            offset = abs(i-j)

            if offset < dlen:
                temp_mtx_1 = np.zeros((dlen + offset, dlen))
                temp_mtx_2 = np.zeros((dlen + offset, dlen))

                if j>i:
                    temp_mtx_1[:dlen, :] = mtx_i
                    temp_mtx_2[offset:, :] = mtx_j
                else:
                    temp_mtx_1[offset:, :] = mtx_i
                    temp_mtx_2[:dlen, :] = mtx_j

                diff_matrix = np.matmul(temp_mtx_1.T, temp_mtx_2)

        return diff_matrix
        
    def interpolate(self, numOfsubgrids, normalize=True, kind='cubic'):
        """
        Generates interpolated dictionary given the original dictionary and number of subgrids
        (TODO): Incorporate with generate_sinc_Dictionary

        Inputs
        ======
        numOfsubgrids: integer
            Specifies number of subgrids. For example if 10, divides the original sampling grid into 10 equal partitions
        kind: 'cubic' (default), 'linear', or 'sinc'
            Interpolator function

        Outputs
        =======
        interpolator: Dictionary
            Dictionary of interpolator functions

        """

        interpolator = {}
        if numOfsubgrids<=1:
            return self.dictionary, interpolator

        if self.config["output"]["verbose"] > 0:
            print("Interpolating with 1/{} sub-grid".format(numOfsubgrids))

        interval = 1/numOfsubgrids
        delay_arr = np.arange(interval, 1, interval)

        numOfsamples, numOfelements = self.dictionary.shape
        assert((0 not in delay_arr) and (1 not in delay_arr)), "Only non-integer delays are allowed"
        assert np.mod(numOfsamples,2)==1, "The filter length must be odd."

        numOfdelays = len(delay_arr)

        # Extra element is for the original template
        d_interpolated = np.zeros((numOfsamples, numOfelements*(numOfdelays+1)))
        d_interpolated[:, np.arange(numOfelements)*(numOfdelays + 1)] = self.dictionary.copy()

        # The first interpolator should be a shifted delta function (to produce the original element)
        delta_fn = np.zeros(numOfsamples)
        delta_fn[int(numOfsamples/2)] = 1
        interpolator[0] = delta_fn

        for didx, delay in enumerate(delay_arr,1):
            if kind == 'cubic':
                x_interp = np.linspace(-2,2,5,endpoint=True) - delay
                f_interp = []
                for idx in x_interp:
                    if abs(idx)>=2:
                        f_interp.append(0)
                    elif 1<=abs(idx) and abs(idx)<2:
                        f_new = -0.5*np.power(abs(idx),3) + 2.5*np.power(abs(idx),2) - 4*abs(idx)+2
                        f_interp.append(f_new)
                    else:
                        f_new = 1.5*np.power(abs(idx),3) - 2.5*np.power(abs(idx),2) + 1
                        f_interp.append(f_new)
            elif kind == 'linear':
                x_interp = np.linspace(-1,1,3,endpoint=True) - delay
                f_interp = []
                for idx in x_interp:
                    if abs(idx)>=1:
                        f_interp.append(0)
                    else:
                        f_new = 1 - abs(idx)
                        f_interp.append(f_new)
            elif kind == 'sinc':
                if np.mod(numOfsamples, 2)==0:
                    x = np.arange(numOfsamples) - int(numOfsamples/2)+1
                else:
                    x = np.arange(numOfsamples) - int(numOfsamples/2)

                f_interp = np.sinc(x-delay)
            else:
                raise NotImplementedError("This interpolator is not implemented!")

            interpolator[didx] = f_interp

            for fidx in np.arange(numOfelements):
                elem = self.dictionary[:,fidx]
                d_interpolated[:, fidx*(numOfdelays+1)+didx] = scipy.signal.convolve(elem, f_interp, mode='same')

        if normalize:
            d_interpolated = d_interpolated/np.linalg.norm(d_interpolated, axis=0)

        # self.dictionary = d_interpolated

        return d_interpolated, interpolator
