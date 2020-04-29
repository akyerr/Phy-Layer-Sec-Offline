from numpy import conj, prod, sqrt, convolve, zeros, var
from numpy.random import normal
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

    def receive_sig_process(self, buffer_tx_time):
        """
        This method deals with all the receiver functions - generate rx signal (over channel + noise), synhronization,
        channel estimation, SVD, precoder detection, bit recovery
        :param buffer_tx_time: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        :return:
        """
        buffer_rx_time = self.rx_signal_gen(buffer_tx_time)

        plt.plot(buffer_rx_time[0, :].real)
        plt.plot(buffer_rx_time[0, :].imag)
        plt.show()

        # synchronization - return just data symbols with CP removed
        # buffer_rx_data = self.synchronize(buffer_rx_time)


    def rx_signal_gen(self, buffer_tx_time):
        """
        Generates the time domain rx signal at each receive antenna (Convolution with channel and add noise)
        :param buffer_tx_time: Time domain tx signal streams on each antenna (matrix)
        :return: Time domain rx signal at each receive antenna
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
        :return: Signal with AWGN added on per antenna basis
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
        :return: Time domain rx signal at each receive antenna with CP removed
        """
        buffer_rx_data = zeros((self.num_ant, self.num_data_symb*self.NFFT), dtype=complex)
        # for
