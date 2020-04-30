from numpy import conj, sqrt, convolve, zeros, var, diag, angle, dot, exp, array, real, argwhere
from numpy.random import normal
from numpy.fft import fft
from numpy.linalg import svd
import matplotlib.pyplot as plt


class PLSReceiver:
    def __init__(self, pls_params, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb,SNRdB, SNR_type):
        """
        Initialization of class
        :param pls_params: object from PLSParameters class containing basic parameters
        :param synch: object of SynchSignal class containing synch parameters and synch mask
        :param symb_pattern: List of 0s and 1s representing pattern of symbols - synch is represented by 0, data by 1
        :param total_num_symb: Total number of OFDM symbols (synch + data)
        :param num_data_symb: Number of data OFDM symbols
        :param num_synch_symb: Number of synch OFDM symbols
        :param SNRdB: Signal to Noise Ratio in dB
        :param SNR_type: Analog or Digital SNR
        """
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
        self.total_symb_len = self.total_num_symb*self.OFDMsymb_len

        self.num_data_symb = num_data_symb
        self.num_synch_symb = num_synch_symb
        self.SNRdB = SNRdB
        self.SNR_type = SNR_type

        self.codebook = pls_params.codebook

    def receive_sig_process(self, buffer_tx_time, ref_sig):
        """
        This method deals with all the receiver functions - generate rx signal (over channel + noise), synhronization,
        channel estimation, SVD, precoder detection, bit recovery
        :param buffer_tx_time: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        :param ref_sig: Matrix of QPSK reference signals. QPSK values for each frequency bin in each data symbol
        :return:
        """
        buffer_rx_time = self.rx_signal_gen(buffer_tx_time)

        # synchronization - return just data symbols with CP removed
        buffer_rx_data = self.synchronize(buffer_rx_time)

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
        buffer_rx_time = zeros((self.num_ant,self.total_symb_len + self.max_impulse - 1), dtype=complex)
        for rx in range(self.num_ant):
            rx_sig_ant = 0  # sum rx signal at each antenna
            for tx in range(self.num_ant):
                chan = self.channel_time[rx, tx, :]
                tx_sig = buffer_tx_time[tx, :]
                rx_sig_ant += convolve(tx_sig, chan)
                # print(self.buffer_data_tx_time[tx, :].shape)
                # print(self.channel_time[rx, tx, :].shape)

            buffer_rx_time[rx, :] = rx_sig_ant
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
        :return buffer_rx_data: Time domain rx signal at each receive antenna with CP removed
        """
        buffer_rx_data = zeros((self.num_ant, self.num_data_symb*self.NFFT), dtype=complex)

        total_symb_count = 0
        synch_symb_count = 0
        data_symb_count = 0
        for symb in self.symb_pattern.tolist():
            symb_start = total_symb_count * self.OFDMsymb_len
            symb_end = symb_start + self.OFDMsymb_len
            # print(symb_start, symb_end)
            if int(symb) == 0:
                synch_symb_count += 1
            else:
                # print(symb, symb_start, symb_end)
                data_start = data_symb_count * self.NFFT
                data_end = data_start + self.NFFT
                # print(data_start, data_end)
                # print(time_ofdm_data_symbols[:, data_start: data_end])
                data_with_CP = buffer_rx_time[:,  symb_start: symb_end]
                data_without_CP = data_with_CP[:, self.CP: ]
                buffer_rx_data[:, data_start: data_end] = data_without_CP
                data_symb_count += 1

            total_symb_count += 1
        # print(buffer_rx_data.shape)
        return buffer_rx_data

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
        # print(chan_est.shape)
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