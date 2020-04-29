from numpy import conj, prod, sqrt, convolve, zeros, var
from numpy.random import normal
import matplotlib.pyplot as plt


class PLSReceiver:
    def __init__(self, pls_params, synch, symb_pattern, total_num_symb):
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

    def receive_sig_process(self, buffer_tx_time, SNRdB, SNR_type):
        #1. generate over the channel signal
        #2. Add noise
        #3. Synchronize
        #4. FFT
        #5. Channel estimate
        #6. Construct freq domain matrices
        #7. SVD
        pass
        tx_time_ofdm_symbols = buffer_tx_time
        buffer_rx_time = self.rx_signal_gen(tx_time_ofdm_symbols)
        buffer_rx_time = self.awgn(buffer_rx_time, SNRdB, SNR_type)

        plt.plot(buffer_rx_time[0, :].real)
        plt.plot(buffer_rx_time[0, :].imag)
        plt.show()

    def rx_signal_gen(self, tx_time_ofdm_symbols):
        rx_ofdm_symb_time = zeros((self.num_ant,self.total_symb_len + self.max_impulse - 1), dtype=complex)
        for rx in range(self.num_ant):
            rx_sig_ant = 0  # sum rx signal at each antenna
            for tx in range(self.num_ant):
                chan = self.channel_time[rx, tx, :]
                tx_sig = tx_time_ofdm_symbols[tx, :]
                rx_sig_ant += convolve(tx_sig, chan)
                # print(self.buffer_data_tx_time[tx, :].shape)
                # print(self.channel_time[rx, tx, :].shape)

            rx_ofdm_symb_time[rx, :] = rx_sig_ant
        return rx_ofdm_symb_time



    def awgn(self, in_signal, SNRdB, SNR_type):
        SNR_lin = 10 ** (SNRdB / 10)
        sig_pow = var(in_signal)  # Determine the expected value of tx signal power

        bits_per_symb = self.num_data_bins * self.bit_codebook
        samp_per_symb = self.OFDMsymb_len

        # Calculate noise variance
        if SNR_type == 'Digital':
            noise_var = (1 / bits_per_symb) * samp_per_symb * sig_pow * 10 ** (-SNRdB / 10)
        elif SNR_type == 'Analog':
            noise_var = sig_pow * 10 ** (-SNRdB / 10)
        else:
            exit(0)

        SNR_analog = sig_pow / noise_var
        # print(self.noise_var)
        # Add noise at each receive antenna

        for rx in range(self.num_ant):

            awg_noise = sqrt(noise_var/2)*(normal(0, 1, in_signal[rx,:].shape) + 1j*normal(0, 1, in_signal[rx, :].shape))

            in_signal[rx, :] += awg_noise

        noisy_signal = in_signal
        return noisy_signal