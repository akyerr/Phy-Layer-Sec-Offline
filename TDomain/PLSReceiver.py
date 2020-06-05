from numpy import conj, sqrt, convolve, zeros, var, diag, angle, dot, exp, array, real, argwhere, absolute, ceil, sum, argmax, roll, floor, sum, insert, copy, append
from numpy.random import normal, randint
from numpy.fft import fft
from numpy.linalg import svd
import scipy.stats.mstats as scipy
import matplotlib.pyplot as plt



class PLSReceiver:
    def __init__(self, plot_diagnostics, pls_params, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb,SNRdB, SNR_type):
        """
        Initialization of class
        :param plot_diagnostics: boolean allowing for plots to be generated
        :param pls_params: object from PLSParameters class containing basic parameters
        :param synch: object of SynchSignal class containing synch parameters and synch mask
        :param symb_pattern: List of 0s and 1s representing pattern of symbols - synch is represented by 0, data by 1
        :param total_num_symb: Total number of OFDM symbols (synch + data)
        :param num_data_symb: Number of data OFDM symbols
        :param num_synch_symb: Number of synch OFDM symbols
        :param SNRdB: Signal to Noise Ratio in dB
        :param SNR_type: Analog or Digital SNR
        """
        self.plots = plot_diagnostics

        self.bandwidth = pls_params.bandwidth
        self.bin_spacing = pls_params.bin_spacing
        self.num_ant = pls_params.num_ant
        self.bit_codebook = pls_params.bit_codebook

        self.NFFT = pls_params.NFFT
        self.CP = pls_params.CP
        self.OFDMsymb_len = self.NFFT + self.CP
        self.num_data_bins = pls_params.num_data_bins

        self.used_data_bins = pls_params.used_data_bins
        self.subband_size = self.num_ant

        self.num_subbands = pls_params.num_subbands
        self.num_PMI = self.num_subbands

        self.synch = synch
        self.symb_pattern = symb_pattern

        self.channel_time = pls_params.channel_time
        self.max_impulse = pls_params.max_impulse
        self.total_num_symb = total_num_symb

        print("Total Number of Symbols: ", self.total_num_symb)
        self.total_symb_len = self.total_num_symb*self.OFDMsymb_len
        print("Total Symbol Length: ", self.total_symb_len)
        self.num_data_symb = num_data_symb
        print("Number of Data Symbols: ", self.num_data_symb)
        self.num_synch_symb = num_synch_symb
        print("Number of Synch Symbols: ", self.num_synch_symb)
        self.synch_data_pattern = [int(self.num_synch_symb / self.num_data_symb), int(self.total_num_symb / self.num_data_symb - self.num_synch_symb / self.num_data_symb)]
        print("Synch Data Pattern: ", self.synch_data_pattern)
        self.synch_data_pattern_sum = self.synch_data_pattern[0] + self.synch_data_pattern[1]
        self.SNRdB = SNRdB
        self.SNR_type = SNR_type

        self.codebook = pls_params.codebook
        self.mask = self.synch.mask

        self.power_requirements = 0

    def receive_sig_process(self, buffer_tx_time, ref_sig):
        """
        This method deals with all the receiver functions - generate rx signal (over channel + noise), synhronization,
        channel estimation, SVD, precoder detection, bit recovery
        :param buffer_tx_time: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        :param ref_sig: Matrix of QPSK reference signals. QPSK values for each frequency bin in each data symbol
        :return:
        """
        buffer_rx_time = self.rx_signal_gen(buffer_tx_time)

        # Shuffle the signal for simulation.
        roll_amt = randint(-10000, 10000)
        roll_amt = 0
        print("Rolling Signal by ", roll_amt, " amount!")
        buffer_rx_time = roll(buffer_rx_time, roll_amt)

        if self.plots is True:
            plt.title("Phase Adjusted RX Signal")
            plt.plot(buffer_rx_time[1, :480].real)
            plt.plot(buffer_rx_time[1, :480].imag)
            plt.show()

        # synchronization - return just data symbols with CP removed
        buffer_rx_data = self.synchronize(buffer_rx_time)

        if self.plots is True:
            plt.title("RX Buffer: DATA ONLY")
            plt.plot(buffer_rx_data[1, 0:128].real)
            plt.plot(buffer_rx_data[1, 0:128].imag)
            plt.show()

        # Channel estimation in each of the used bins
        chan_est_bins = self.channel_estimate(buffer_rx_data, ref_sig)

        # Map bins to sub-bands to form matrices for SVD - gives estimated channel matrix in each sub-band
        chan_est_sb = self.bins2subbands(chan_est_bins)

        # SVD in each sub-band
        lsv, _, rsv = self.sv_decomp(chan_est_sb)

        # rsv is supposed to be the received dft precoder
        bits_sb_estimate = self.PMI_estimate(rsv)[1]

        return lsv, rsv, bits_sb_estimate

    def rx_signal_gen(self, buffer_tx_time):
        """
        Generates the time domain rx signal at each receive antenna (Convolution with channel and add noise)
        :param buffer_tx_time: Time domain tx signal streams on each antenna (matrix)
        :return buffer_rx_time: Time domain rx signal at each receive antenna
        """
        buffer_rx_time = zeros((self.num_ant, self.total_symb_len), dtype=complex)
        for rx in range(self.num_ant):
            rx_sig_ant = 0  # sum rx signal at each antenna
            for tx in range(self.num_ant):
                chan = self.channel_time[rx, tx, :]
                tx_sig = buffer_tx_time[tx, :]
                rx_sig_ant += convolve(tx_sig, chan)
                # print(self.buffer_data_tx_time[tx, :].shape)
                # print(self.channel_time[rx, tx, :].shape)

            buffer_rx_time[rx, :] = rx_sig_ant[:-int(self.max_impulse - 1)]
        buffer_rx_time = self.awgn(buffer_rx_time)

        return buffer_rx_time

    def awgn(self, in_signal):
        """
        Adds AWGN noise on per antenna basis
        :param in_signal: Input signal
        :return noisy_signal: Signal with AWGN added on per antenna basis
        """

        sig_pow = var(in_signal)  # Determine the expected value of tx signal power

        bits_per_symb = self.num_data_bins * self.bit_codebook
        samp_per_symb = self.OFDMsymb_len

        # Calculate noise variance
        if self.SNR_type == 'Digital':
            noise_var = (1 / bits_per_symb) * samp_per_symb * sig_pow * 10 ** (-self.SNRdB / 10)
        elif self.SNR_type == 'Analog':
            noise_var = sig_pow * 10 ** (-self.SNRdB / 10)
        else:
            noise_var = None
            exit(0)

        for rx in range(self.num_ant):

            awg_noise = sqrt(noise_var/2)*(normal(0, 1, in_signal[rx,:].shape) + 1j*normal(0, 1, in_signal[rx, :].shape))

            in_signal[rx, :] += awg_noise

        noisy_signal = in_signal

        return noisy_signal

    def synchronize(self, buffer_rx_time):
        """
        Time domain synchronization - correlation with Zadoff Chu Synch mask
        :param buffer_rx_time: Time domain rx signal at each receive antenna
        :return buffer_rx_data_wo_cp: Time domain rx signal at each receive antenna with CP removed
        """

        rx_data_time_power = self.power_estimate(buffer_rx_time, len(buffer_rx_time[1]))
        if rx_data_time_power > self.power_requirements:
            symb_pattern_buffer = copy(self.symb_pattern)
            periodicity = len(self.mask)
            total_complete_data_symbs = 0

            if self.plots is True:
                plt.title("Signal Mask for Correlation")
                plt.plot(self.mask.real)
                plt.plot(self.mask.imag)
                plt.show()

            window_size = (len(self.mask))
            window_slide_dist = 1
            num_windows = int(
                ceil(((len(buffer_rx_time[1, :periodicity]) - window_size) / window_slide_dist) + 1))

            correlation_value_buffer = zeros((self.num_ant, num_windows, 1))
            correlation_index_buffer = zeros((self.num_ant, num_windows, 1))

            index_offset, corr_values = self.correlate(buffer_rx_time[1, 0:periodicity])
            correlation_value_buffer[1, :, 0] = corr_values.max()
            correlation_index_buffer[1, :, 0] = index_offset.argmax()

            if self.plots is True:
                plt.title("Zadoff-Chu Correlation")
                plt.plot(corr_values)
                plt.show()

            correct_index = int(scipy.gmean(correlation_index_buffer[1, :, :]))
            print("The correct index offset is: ", correct_index)

            power_estimate = self.power_estimate(buffer_rx_time, correct_index)
            print("Estimated Power of Signal Before the start of the Data: ", power_estimate)

            print("Is there data before the start?")

            if correct_index > 0:
                if power_estimate > self.power_requirements:
                    num_symbs_before_synch = (correct_index - 1) / (self.NFFT + self.CP)
                    print("Yes! There are ", correct_index - 1, " time-series elements before the start of the Synch Symbol.")
                    print("Number of symbols before Starting synch symbol: ", num_symbs_before_synch)

                    num_data_before_synch = 0
                    num_synch_before_synch = 0

                    num_symbs_left = num_symbs_before_synch

                    num_complete_data = 0
                    num_complete_synch = 0
                    num_partial_data = 0
                    num_partial_synch = 0

                    # Calculating the true start of the Data, including what portions are wasted symbols.
                    for symb in range(0, int(ceil(num_symbs_before_synch)), int(self.synch_data_pattern_sum)):
                        for symb_type in [1, 0]:

                            if num_symbs_left - self.synch_data_pattern[symb_type] >= 0:
                                num_symbs_left = num_symbs_left - self.synch_data_pattern[symb_type]
                                if symb_type == 0:
                                    num_synch_before_synch += self.synch_data_pattern[symb_type]
                                    print("Synch Symbol(s) Detected!")
                                if symb_type == 1:
                                    print("Data Symbol(s) Detected!")
                                    num_data_before_synch += self.synch_data_pattern[symb_type]

                            elif num_symbs_left - self.synch_data_pattern[symb_type] < 0 and num_symbs_left > 0:

                                previos_num_symbs_left = num_symbs_left
                                num_symbs_left = num_symbs_left - self.synch_data_pattern[symb_type]
                                if symb_type == 0:

                                    if 1 < previos_num_symbs_left < 2:
                                        num_synch_before_synch += 1

                                    else:
                                        print("Error!")
                                    print("The rest of the data stream has ", num_symbs_left, " of a synch symbol!")
                                    num_complete_synch = num_synch_before_synch
                                    num_partial_synch = num_symbs_left - floor(num_symbs_left)
                                    num_complete_data = num_data_before_synch

                                elif symb_type == 1:
                                    print("The rest of the data stream has ", num_symbs_left, " of a data symbol!")
                                    num_complete_data = num_data_before_synch
                                    num_partial_data = num_symbs_left - floor(num_symbs_left)
                                    num_complete_synch = num_synch_before_synch

                    if num_symbs_before_synch < 1:
                        num_partial_data = num_symbs_before_synch
                        total_complete_data_symbs = self.num_data_symb - 1
                        print("The number of partial data symbols before the correlation is, ", num_partial_data)
                        print("The number of complete data symbols before the correlation is, ", num_complete_data)
                        print("========================================================================================")
                        print("The number of partial synch symbols before the correlation is, ", num_partial_synch)
                        print("The number of complete synch symbols before the correlation is, ", num_complete_synch)

                    else:
                        total_complete_data_symbs = self.num_data_symb
                        print("The number of partial data symbols before the correlation is, ", num_partial_data)
                        print("The number of complete data symbols before the correlation is, ", num_complete_data)
                        print("========================================================================================")
                        print("The number of partial synch symbols before the correlation is, ", num_partial_synch)
                        print("The number of complete synch symbols before the correlation is, ", num_complete_synch)
                else:
                    print("No!")
                    num_complete_data = 0
                    num_partial_data = 0
                    num_complete_synch = 0
                    num_partial_synch = 0
                    print("Signal Prior to Correct Index does not meet power requirements!")
            else:
                print("No!")
                num_complete_data = 0
                num_partial_data = 0
                num_complete_synch = 0
                num_partial_synch = 0

            if num_partial_synch > 0:
                print("There are stray SYNCH Symbols!")
                modulo_patch = (num_complete_synch + num_partial_synch) % self.synch_data_pattern[0]
                corrected_index = int(floor(modulo_patch * (self.NFFT + self.CP)))
                if num_complete_synch >= 1:
                    symb_pattern_buffer = symb_pattern_buffer[int(ceil(num_partial_synch) + num_complete_synch):]
                else:
                    symb_pattern_buffer = symb_pattern_buffer[int(self.synch_data_pattern[0]):]
                # print("Synch-Data Pattern without wasted symbols:", symb_pattern_buffer)

            elif num_partial_data > 0:
                print("There is a stray DATA symbol!")
                modulo_patch = (num_complete_data + num_partial_data) % self.synch_data_pattern[1]
                corrected_index = int(floor(modulo_patch * (self.NFFT + self.CP) + self.synch_data_pattern[0] * (self.NFFT + self.CP)))
                symb_pattern_buffer = roll(symb_pattern_buffer, -int(self.synch_data_pattern[0]))
                symb_pattern_buffer = symb_pattern_buffer[:-int(sum(self.synch_data_pattern))]
                # print("Synch-Data Pattern without wasted symbols: ", symb_pattern_buffer)

            else:
                corrected_index = correct_index
                total_complete_data_symbs = self.num_data_symb

            buffer_rx_data = self.strip_synchs(buffer_rx_time, corrected_index, total_complete_data_symbs, symb_pattern_buffer)

            # Recovering wasted symbols by stitching the end of the buffer with the start of the buffer.
            if power_estimate > self.power_requirements and num_partial_data > 0:
                recovered_data = self.recover_lost_data(buffer_rx_time, num_partial_data)
                buffer_rx_data = append(recovered_data, buffer_rx_data, axis=1)

            # Remove CP
            buffer_rx_data_wo_cp = self.remove_cp(buffer_rx_data)
        else:
            print("Power of signal is too low! Returning zeros!")
            buffer_rx_data_wo_cp = zeros((self.num_ant, self.num_data_symb))

        return buffer_rx_data_wo_cp

    def remove_cp(self, buffer_rx_data):
        buffer_rx_data_wo_cp = zeros((self.num_ant, self.num_data_symb * self.NFFT), dtype=complex)
        num_of_cuts = int(len(buffer_rx_data[0, :]) / (self.NFFT + self.CP))
        for index in range(num_of_cuts):

            start = index * (self.NFFT + self.CP) + self.CP
            end = (index + 1) * (self.NFFT + self.CP)

            buffer_start = index * self.NFFT
            buffer_end = (index + 1) * self.NFFT

            buffer_rx_data_wo_cp[:, buffer_start: buffer_end] = buffer_rx_data[:, start: end]

        return buffer_rx_data_wo_cp

    def strip_synchs(self, buffer_rx_time, corrected_index, num_complete_data, symb_pattern):
        """
        Removes synchronization elements from the data stream.
        :param buffer_rx_time: Time domain rx signal at each antenna
        :param corrected_index: The correct starting index, an index for removing any partial symbols
        :param num_complete_data: number of data symbols found before the synchronization point in the data stream
        :param symb_pattern: variable containing the full pattern of the input data, 1 for a data symbol, 0 for a synchronization symbol
        :return: a buffer containing only the data elements
        """

        data_only_buffer = zeros((self.num_ant, num_complete_data * (self.NFFT + self.CP)), dtype=complex)

        count = 0
        for index in range(symb_pattern.shape[0]):
            start = int(index * (self.NFFT + self.CP) + corrected_index)
            end = int((index + 1) * (self.NFFT + self.CP) + corrected_index)
            if symb_pattern[index] == 0:
                pass
            elif symb_pattern[index] == 1:
                buffer_start = count * (self.NFFT + self.CP)
                buffer_end = (count + 1) * (self.NFFT + self.CP)
                data_only_buffer[:, buffer_start:buffer_end] = buffer_rx_time[:, start:end]
                count += 1
            else:
                print("Error in Strip Synchs!")

        return data_only_buffer

    def power_estimate(self, input_data, correct_index):
        """
        Estimates the power of a given signal.
        :param input_data: numpy array containing data of interest across N antennas
        :param correct_index: synchronization point, which is used to determine the end of the buffer that matters.
        :return: signal power of the particular data stream
        """
        signal_power = sum((input_data[1, 0:correct_index] * conj(
            input_data[1, 0:correct_index]))) / len(input_data[1, :])

        return signal_power

    def recover_lost_data(self, buffer_rx_time, num_partial_data):
        """
        Partial data due to buffer constraints is dealt with through this function.
        :param buffer_rx_time: Time domain rx signal at each antenna
        :param num_partial_data: Num of symbols cut-off from the buffer due to buffer constraints.
        :return: recovered data: the reconstructed data consisitng of the beginning and end of the buffer (will change in GNU Radio)
        """
        partial_data_before = buffer_rx_time[:, 0:int(num_partial_data * (self.NFFT + self.CP))]
        partial_data_after = buffer_rx_time[:, -int((self.NFFT + self.CP) - num_partial_data * (self.NFFT + self.CP)):]
        recovered_data = append(partial_data_after, partial_data_before, axis=1)

        return recovered_data

    def correlate(self, rx_signal):
        """
        Begins the correlation process and handles values from resulting functions.
        :param rx_signal: Time domain rx signal at each antenna
        :return: index_offset: correlation values for index offset.
        :return: corr_values: correlation values that show the magnitude of the correlation for a given time-step
        """
        in0 = rx_signal[:]
        corr = self.autocorr(in0, self.mask)
        index_offset = corr
        corr_values = absolute(corr)

        return index_offset, corr_values

    def autocorr(self, in0, signal_mask):
        """
        Mathematical implementation of the Correlation Function
        :param in0: Time domain rx signal at each antenna
        :param signal_mask: Variable containing time-domain rx signal without data symbols.
        :return: result: correlation values from mathematical function
        """
        window_size = int(signal_mask.shape[0])
        signal_mask_end = window_size
        window_slide_dist = int(1)
        num_windows = len(in0)

        result = zeros([num_windows])

        for index in range(num_windows):
            start = index * window_slide_dist
            end = index * window_slide_dist + window_size
            if end > len(in0):
                end = int(len(in0))
                signal_mask_end = end - start
            result[index] = sum(dot(in0[start:end], conj(signal_mask[0: signal_mask_end])))
        return result

    def channel_estimate(self, buffer_rx_data, ref_sig):
        """
        In PLS, only refeence signals are sent. So we use the data symbols to estimate the chanel rather than the synch.
        :param buffer_rx_data: Buffer of rx data symbols with CP removed
        :param ref_sig: QPSK ref signals on each bin for each data symbol
        :return chan_est_bins: Estimated channel in each of the used data bins
        """
        SNRlin = 10**(self.SNRdB/10)
        chan_est_bins = zeros((self.num_ant, self.num_data_symb*self.num_data_bins), dtype=complex)
        for symb in range(self.num_data_symb):
            symb_start = symb*self.NFFT
            symb_end = symb_start + self.NFFT

            used_symb_start = symb*self.num_data_bins
            used_symb_end = used_symb_start + self.num_data_bins
            for ant in range(self.num_ant):
                time_data = buffer_rx_data[ant, symb_start: symb_end]
                data_fft = fft(time_data, self.NFFT)
                data_in_used_bins = data_fft[self.used_data_bins]

                chan_est_bins[ant, used_symb_start: used_symb_end] = data_in_used_bins*conj(ref_sig[symb, :])/(abs(ref_sig[symb, :])) # channel at the used bins
        return chan_est_bins

    def bins2subbands(self, chan_est_bins):
        """
        Example: if num antenna = 2, every 2 adjacent bins form a sub-band. Each bin has a column vector (num antennas)
        By combining two adjacent column vectors, we get a matrix in each sub-band.
        :param chan_est_bins: Estimated channel at each used data frequency bin on each antenna
        :return chan_est_sb: channel matrix for each sub-band for each symbol
        """

        chan_est_sb = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            symb_start = symb*self.num_data_bins
            symb_end = symb_start + self.num_data_bins
            chan_est = chan_est_bins[:, symb_start: symb_end] # extract est channel in one symbol
            for sb in range(self.num_subbands):
                sb_start = sb*self.subband_size
                sb_end = sb_start + self.subband_size

                chan_est_sb[symb, sb] = chan_est[:, sb_start: sb_end]
        # print(chan_est_sb[23])
        return chan_est_sb

    def sv_decomp(self, chan_est_sb):
        """
        Perform SVD for the matrix in each sub-band in each data symbol
        :param chan_est_sb: Estimated channel in each sub-band in each data symbol
        :return lsv, sval, rsv: Left, Right Singular Vectors and Singular Values for the matrix in each sub-band
        in each data symbol
        """
        lsv = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        sval = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        rsv = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        for symb in range(self.num_data_symb):
            for sb in range(0, self.num_subbands):
                U, S, VH = svd(chan_est_sb[symb, sb])
                # print(chan_est_sb[symb, sb].shape)
                V = conj(VH).T
                ph_shift_u = diag(exp(-1j * angle(U[0, :])))
                ph_shift_v = diag(exp(-1j * angle(V[0, :])))
                lsv[symb, sb] = dot(U, ph_shift_u)
                sval[symb, sb] = S
                rsv[symb, sb] = dot(V, ph_shift_v)

        return lsv, sval, rsv

    @staticmethod
    def dec2binary(x, num_bits):
        """
        Covert decimal number to binary array of ints (1s and 0s)
        :param x: input decimal number
        :param num_bits: Number bits required in the binary format
        :return bits: binary array of ints (1s and 0s)
        """
        bit_str = [char for char in format(x[0, 0], '0' + str(num_bits) + 'b')]
        bits = array([int(char) for char in bit_str])
        return bits

    def PMI_estimate(self, rx_precoder):
        """
        Apply minumum distance to estimate the transmitted precoder, its index in the codebook and the binary equivalent
        of the index
        :param rx_precoder: observed precoder (RSV of SVD)
        :return PMI_sb_estimate, bits_sb_estimate: Preocder matrix index and bits for each sub-band for each data symbol
        """
        PMI_sb_estimate = zeros((self.num_data_symb, self.num_subbands), dtype=int)
        bits_sb_estimate = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            for sb in range(self.num_subbands):
                dist = zeros(len(self.codebook), dtype=float)

                for prec in range(len(self.codebook)):
                    diff = rx_precoder[symb, sb] - self.codebook[prec]
                    diff_squared = real(diff*conj(diff))
                    dist[prec] = sqrt(diff_squared.sum())
                min_dist = min(dist)
                PMI_estimate = argwhere(dist == min_dist)
                PMI_sb_estimate[symb, sb] = PMI_estimate
                bits_sb_estimate[symb, sb] = self.dec2binary(PMI_estimate, self.bit_codebook)

        return PMI_sb_estimate, bits_sb_estimate