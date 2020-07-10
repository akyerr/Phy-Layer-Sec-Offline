from numpy import sum, zeros, array, exp, pi, concatenate, conj, sqrt, append
from numpy.fft import ifft
import matplotlib.pyplot as plt


class SynchSignal:
    def __init__(self, pls_params, num_synch_symb, num_data_symb, symb_pattern):
        self.bandwidth = pls_params.bandwidth
        self.bin_spacing = pls_params.bin_spacing
        self.num_ant = pls_params.num_ant
        self.bit_codebook = pls_params.bit_codebook
        self.synch_data_pattern = pls_params.synch_data_pattern
        self.num_synch_symb = num_synch_symb
        self.NFFT = pls_params.NFFT
        self.CP = pls_params.CP
        self.OFDMsymb_len = self.NFFT + self.CP
        self.symb_pattern = symb_pattern
        self.num_synch_bins = self.NFFT - 2
        # self.used_synch_bins

        DC_index = int(self.NFFT / 2)
        neg_synch_bins = list(range(DC_index - int(self.num_synch_bins / 2), DC_index))
        pos_synch_bins = list(range(DC_index + 1, DC_index + int(self.num_synch_bins / 2) + 1))
        self.used_synch_bins = array(neg_synch_bins + pos_synch_bins)

        self.num_data_symb = num_data_symb
        self.prime_no = [23, 41]
        self.prime_nos = self.prime_no * self.num_data_symb
        self.num_unique_synch = len(self.prime_no)
        assert len(self.prime_nos) == self.num_synch_symb

        self.synch_signals = zeros((self.num_synch_symb, self.OFDMsymb_len), dtype=complex)
        self.mimo_synch = zeros((self.num_ant, self.num_unique_synch * self.num_ant * self.OFDMsymb_len), dtype=complex)
        self.synch_freq = zeros((self.num_synch_symb, self.NFFT), dtype=complex)
        for symb in range(self.num_synch_symb):
            synch_symb = self.zadoff_chu_gen(self.prime_nos[symb])  # size NFFT - 2
            self.synch_freq[symb, self.used_synch_bins] = synch_symb # size NFFT

            synch_ifft = ifft(self.synch_freq[symb])  # size NFFT
            synch_cp = synch_ifft[-self.CP:] # size CP
            synch_time = concatenate((synch_cp, synch_ifft))  # size NFFT + CP

            power_est = sum(synch_time*conj(synch_time))/len(synch_time)
            norm_synch = synch_time/sqrt(power_est)

            self.synch_signals[symb, :] = norm_synch

        self.synch_mask = zeros((self.num_ant, self.OFDMsymb_len*len(self.symb_pattern)), dtype=complex)

        total_symb_count = 0
        synch_symb_count = 0
        self.synch_start = list()
        for symb in self.symb_pattern.tolist():
            if symb == 0:
                symb_start = total_symb_count * self.OFDMsymb_len
                symb_end = symb_start + self.OFDMsymb_len
                # print(symb_start, symb_end)
                self.synch_start.append(symb_start)
                self.synch_mask[0, symb_start: symb_end] = self.synch_signals[synch_symb_count]
                synch_symb_count += 1

            total_symb_count += 1
        # Presignal
        count = 0
        for ant in range(self.num_ant):
            start_symb = count * self.OFDMsymb_len * self.num_unique_synch
            end_symb = start_symb + self.OFDMsymb_len * self.num_unique_synch
            self.mimo_synch[ant, start_symb: end_symb] = self.synch_signals[0: self.num_unique_synch, :].flatten()
            count += 1
        plt.plot(self.synch_mask[1, :].real)
        plt.plot(self.synch_mask[1, :].imag)
        plt.show()
        self.synch_mask = append(self.mimo_synch, self.synch_mask, axis=1)
        plt.plot(self.synch_mask[0, :].real)
        plt.plot(self.synch_mask[0, :].imag)
        plt.show()
        plt.plot(self.synch_mask[1, :].real)
        plt.plot(self.synch_mask[1, :].imag)
        plt.show()
        dbg = 1

    def zadoff_chu_gen(self, prime):
        x0 = array(range(0, self.num_synch_bins))
        x1 = array(range(1, self.num_synch_bins + 1))
        if self.num_synch_bins % 2 == 0:
            zadoff_chu = exp(-1j * (2 * pi / self.num_synch_bins) * prime * (x0**2 / 2))
        else:
            zadoff_chu = exp(-1j * (2 * pi / self.num_synch_bins) * prime * (x0 * x1) / 2)

        return zadoff_chu
