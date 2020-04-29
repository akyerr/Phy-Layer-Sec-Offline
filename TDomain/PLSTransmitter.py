from numpy import pi, exp, array, zeros, dot, diag, concatenate
from numpy.fft import ifft
from numpy.random import choice, uniform
from numpy.linalg import qr
import matplotlib.pyplot as plt


class PLSTransmitter:
    def __init__(self, pls_params, synch, symb_pattern):
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

    def transmit_signal_gen(self, *args):
        tx_node = args[0]
        num_data_symb = args[1]
        ref_sig = self.ref_signal_gen(num_data_symb)

        if tx_node == 'Alice0' and len(args) == 2:
            precoders = self.unitary_gen(num_data_symb)
        elif tx_node == 'Bob':
            precoders = None
        elif tx_node == 'Alice1':
            precoders = None
        else:
            precoders = None
            print('Error')

        freq_bin_data = self.apply_precoders(precoders, ref_sig, num_data_symb)

        time_ofdm_data_symbols = self.ofdm_modulate(num_data_symb, freq_bin_data)
        # print(time_ofdm_data_symbols)
        time_ofdm_symbols = self.synch_data_mux(time_ofdm_data_symbols)

        return time_ofdm_symbols

    # QPSK reference signals
    def ref_signal_gen(self, num_data_symb):
        ref_sig = zeros((num_data_symb, self.num_data_bins), dtype=complex)
        for symb in range(num_data_symb):
            for fbin in range(self.num_data_bins):
                ref_sig[symb, fbin] = exp(1j * (pi / 4) * (choice(array([1, 3, 5, 7]))))

        return ref_sig

    def unitary_gen(self, num_data_symb):
        unitary_mats = zeros((num_data_symb, self.num_subbands), dtype=object)
        for symb in range(num_data_symb):
            for sb in range(0, self.num_subbands):
                Q, R = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                          + 1j * uniform(0, 1, (self.num_ant, self.num_ant)))

                unitary_mats[symb, sb] = dot(Q, diag(diag(R) / abs(diag(R))))
        return unitary_mats

    def ofdm_modulate(self, num_symb, freq_bin_data):
        time_ofdm_symbols = zeros((self.num_ant, num_symb * self.OFDMsymb_len), dtype=complex)
        for ant in range(self.num_ant):
            for symb in range(num_symb):
                freq_data_start = symb * self.num_data_bins
                freq_data_end = freq_data_start + self.num_data_bins

                ofdm_symb = zeros(self.NFFT, dtype=complex)
                ofdm_symb[self.used_data_bins] = freq_bin_data[ant, freq_data_start:freq_data_end]

                data_ifft = ifft(ofdm_symb, self.NFFT)
                cyclic_prefix = data_ifft[-self.CP:]
                data_time = concatenate((cyclic_prefix, data_ifft))  # add CP

                time_symb_start = symb * self.OFDMsymb_len
                time_symb_end = time_symb_start + self.OFDMsymb_len

                time_ofdm_symbols[ant, time_symb_start: time_symb_end] = data_time
        # print(time_ofdm_symbols)
        # print(time_ofdm_symbols.shape)
        # print(num_symb)
        return time_ofdm_symbols

    def apply_precoders(self, precoders, ref_sig, num_data_symb):
        print(num_data_symb)
        freq_bin_data = zeros((self.num_ant, num_data_symb * self.num_data_bins), dtype=complex)
        for symb in range(num_data_symb):
            # print(symb)
            symb_start = symb * self.num_data_bins
            symb_end = symb_start + self.num_data_bins

            fbin_val = zeros((self.num_ant, self.num_data_bins), dtype=complex)
            for sb in range(self.num_subbands):
                precoder = precoders[symb, sb]

                sb_start = sb * self.subband_size
                sb_end = sb_start + self.subband_size

                fbin_val[:, sb_start: sb_end] = precoder

            for fbin in range(self.num_data_bins):
                fbin_val[:, fbin] *= ref_sig[symb, fbin]
            # print(fbin_val.shape)
            # print(symb_start, symb_end)
            freq_bin_data[:, symb_start: symb_end] = fbin_val
            # dbg = 1
            # print(freq_bin_data)
        return freq_bin_data

    def synch_data_mux(self, time_ofdm_data_symbols):
        time_ofdm_symbols = self.synch.synch_mask # Add data into this

        total_symb_count = 0
        synch_symb_count = 0
        data_symb_count = 0
        for symb in self.symb_pattern.tolist():
            symb_start = total_symb_count*self.OFDMsymb_len
            symb_end = symb_start + self.OFDMsymb_len
            # print(symb_start, symb_end)
            if int(symb) == 0:
                synch_symb_count += 1
            else:
                # print(symb, symb_start, symb_end)
                data_start = data_symb_count*self.OFDMsymb_len
                data_end = data_start + self.OFDMsymb_len
                # print(data_start, data_end)
                # print(time_ofdm_data_symbols[:, data_start: data_end])
                time_ofdm_symbols[:, symb_start: symb_end] = time_ofdm_data_symbols[:, data_start: data_end]
                data_symb_count += 1

            total_symb_count += 1
        # print('synch ', synch_symb_count, 'data ', data_symb_count, 'total ', total_symb_count)
        plt.plot(time_ofdm_symbols[0, :].real)
        plt.plot(time_ofdm_symbols[0, :].imag)
        plt.show()
        return time_ofdm_symbols



