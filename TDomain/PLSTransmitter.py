from numpy import pi, exp, array, zeros, dot, diag, concatenate, conj, sqrt, var
from numpy.fft import ifft
from numpy.random import choice, uniform
from numpy.linalg import qr
import matplotlib.pyplot as plt


class PLSTransmitter:
    def __init__(self, pls_params, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb):
        """
        Initialization of class
        :param pls_params: object from PLSParameters class containing basic parameters
        :param synch: object of SynchSignal class containing synch parameters and synch mask
        :param symb_pattern: List of 0s and 1s representing pattern of symbols - synch is represented by 0, data by 1
        :param total_num_symb: Total number of OFDM symbols (synch + data)
        :param num_data_symb: Number of data OFDM symbols
        :param num_synch_symb: Number of synch OFDM symbols
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

        self.total_num_symb = total_num_symb
        self.num_data_symb = num_data_symb
        self.num_synch_symb = num_synch_symb

        self.codebook = pls_params.codebook

    def transmit_signal_gen(self, *args):
        """
        This method deals with all the transmitter functions - gen ref signals, precoding, OFDM mod, IFFT, CP
        :param args: 1. Which node is transmitting - Alice or Bob? 2. Total number of data symbols
        :return: Time domain buffer of OFDM symbols (synch + data). Ready to send over channel.
        """

        tx_node = args[0]
        num_data_symb = args[1]


        if tx_node == 'Alice0':
            precoders = self.unitary_gen()
        elif tx_node == 'Bob':
            pvt_info_bits = args[2]
            rotation_mat = args[3]
            bits_subband = self.map_bits2subband(pvt_info_bits)
            dft_precoders = self.codebook_select(bits_subband)
            precoders = self.rotated_preocder('Bob', dft_precoders, rotation_mat)
        elif tx_node == 'Alice1':
            precoders = None
        else:
            precoders = None
            print('Error')

        ref_sig = self.ref_signal_gen()
        freq_bin_data = self.apply_precoders(precoders, ref_sig, num_data_symb)
        time_ofdm_data_symbols = self.ofdm_modulate(num_data_symb, freq_bin_data)
        buffer_tx_time = self.synch_data_mux(time_ofdm_data_symbols)

        return buffer_tx_time, ref_sig

    # QPSK reference signals
    def ref_signal_gen(self):
        """
        Generate QPSK reference signals for each frequency bin in each data symbol
        :return: Matrix of QPSK reference signals
        Same ref signal on both antennas in a bin. (Can be changed later)
        """
        ref_sig = zeros((self.num_data_symb, self.num_data_bins), dtype=complex)
        for symb in range(self.num_data_symb):
            for fbin in range(self.num_data_bins):
                ref_sig[symb, fbin] = exp(1j * (pi / 4) * (choice(array([1, 3, 5, 7]))))

        return ref_sig

    def unitary_gen(self):
        """
        Generate random unitary matrices for each symbol for each sub-band
        :return: matrix of random unitary matrices
        """
        unitary_mats = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            for sb in range(0, self.num_subbands):
                Q, R = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                          + 1j * uniform(0, 1, (self.num_ant, self.num_ant)))

                unitary_mats[symb, sb] = dot(Q, diag(diag(R) / abs(diag(R))))
        return unitary_mats

    def ofdm_modulate(self, num_data_symb, freq_bin_data):
        """
        Takes frequency bin data and places them in appropriate frequency bins, on each antenna
        Then takes FFT and adds CP to each data symbol
        :param num_data_symb: Total number of data OFDM symbols
        :param freq_bin_data: Frequency domain input data
        :return: Time domain tx stream of data symbols on each antenna
        """
        min_pow = 1e-30
        time_ofdm_symbols = zeros((self.num_ant, num_data_symb * self.OFDMsymb_len), dtype=complex)
        for symb in range(num_data_symb):
            freq_data_start = symb * self.num_data_bins
            freq_data_end = freq_data_start + self.num_data_bins

            time_symb_start = symb * self.OFDMsymb_len
            time_symb_end = time_symb_start + self.OFDMsymb_len

            P = 0
            for ant in range(self.num_ant):

                ofdm_symb = zeros(self.NFFT, dtype=complex)
                ofdm_symb[self.used_data_bins] = freq_bin_data[ant, freq_data_start:freq_data_end]

                data_ifft = ifft(ofdm_symb, self.NFFT)
                cyclic_prefix = data_ifft[-self.CP:]
                data_time = concatenate((cyclic_prefix, data_ifft))  # add CP

                # sig_energy = abs(dot(data_time, conj(data_time).T))
                # # power scaling to normalize to 1
                # if sig_energy > min_pow and ant == 0:
                #     scale_factor = sqrt(len(data_time) / sig_energy)
                # else:
                #     scale_factor = 1
                # data_time *= scale_factor
                # P += var(data_time)
                time_ofdm_symbols[ant, time_symb_start: time_symb_end] = data_time

            # for ant in range(self.num_ant):
            #     time_ofdm_symbols[ant, time_symb_start: time_symb_end] *= (1 / sqrt(P))

        return time_ofdm_symbols

    def apply_precoders(self, precoders, ref_sig, num_data_symb):
        """
        Applies precoders in each frequency bin.
        Example: no of antenans = 2 => sub-band size = 2 bins. One precoder matrix is split into 2 columns. Each column
        is applied in a bin in that sub-band
        :param precoders: matrix of precoders for each sub-band and for each data OFDM symbol
        :param ref_sig: matrix of reference signals for each frequency bin in each each data OFDM symbol
        Same ref signal on both antennas in a bin. (Can be changed later)
        :param num_data_symb: Total number of OFDM Data symbols
        :return: frequency bin data for each symbol (precoder*ref signal) and on each antenna
        """
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

            freq_bin_data[:, symb_start: symb_end] = fbin_val

        return freq_bin_data

    def synch_data_mux(self, time_ofdm_data_symbols):
        """
        Multiplexes synch and data symbols according to the symbol pattern
        Takes synch mask from synch class and inserts dats symbols next to the synchs
        :param time_ofdm_data_symbols: NFFT+CP size OFDM data symbols
        :return: time domain tx symbol stream per antenna (contains synch and data symbols) (matrix)
        """

        buffer_tx_time = self.synch.synch_mask # Add data into this

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

                ### normalization tests
                # data_ant0 = time_ofdm_data_symbols[0, data_start: data_end]
                # data_ant1 = time_ofdm_data_symbols[1, data_start: data_end]
                # data_p0 = data_ant0*conj(data_ant0)/self.OFDMsymb_len
                # data_p1 = data_ant1 * conj(data_ant1) / self.OFDMsymb_len
                #
                # data
                ###
                buffer_tx_time[:, symb_start: symb_end] = time_ofdm_data_symbols[:, data_start: data_end]
                data_symb_count += 1

            total_symb_count += 1
        # print('synch ', synch_symb_count, 'data ', data_symb_count, 'total ', total_symb_count)
        print(buffer_tx_time.shape)
        plt.plot(buffer_tx_time[0, :].real)
        plt.plot(buffer_tx_time[0, :].imag)
        plt.show()
        return buffer_tx_time

    def map_bits2subband(self, pvt_info_bits):
        """
        Based on the number of bits in the codebook index, each sub-band (group of bins=num antennas) in each symbol
        is assigned bits from the private info bit stream
        :param pvt_info_bits: Private information to be transmitted
        :return bits_subband: Bits in each sub-band for each data symbol
        """
        bits_subband = zeros((self.num_data_symb, self.num_subbands), dtype=object)

        for symb in range(self.num_data_symb):
            # Map secret key to subbands
            for sb in range(self.num_subbands):
                start = sb * self.bit_codebook
                fin = start + self.bit_codebook

                bits_subband[symb, sb] = pvt_info_bits[start: fin]

        return bits_subband

    def codebook_select(self, bits_subband):
        """
        selects the DFT precoder from the DFT codebook based. Bits are converted to decimal and used as look up index.
        :param bits_subband: Bits in each sub-band for each data symbol
        :return dft_precoder: Selected DFT preocder from codebook for each sub-band in each data symbol
        """
        dft_precoder = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        for symb in range(self.num_data_symb):
            for sb in range(self.num_subbands):
                bits = bits_subband[symb, sb]
                start = self.bit_codebook - 1
                bi2dec_wts = 2**(array(range(start, -1, -1))) # convert bits to decimal index in each sub-band
                codebook_index = sum(bits*bi2dec_wts)
                dft_precoder[symb, sb] = self.codebook[codebook_index]

        return dft_precoder

    def rotated_preocder(self, tx_node, dft_precoders, rotation_mat):
        """
        applies channel-based rotation matrix (LSV from previous SVD) to the DFT precoder
        :param tx_node: Who is transmitting - Alice or Bob?
        :param dft_precoders: selected DFT precoders from codebook based on secret bits
        :param rotation_mat: channel-based rotation matrix (LSV from previous SVD)
        :return:
        """
        precoders = zeros((self.num_data_symb, self.num_subbands), dtype=object)
        if tx_node == 'Bob':
            for symb in range(self.num_data_symb):
                for sb in range(self.num_subbands):
                    precoders[symb, sb] = dot(conj(rotation_mat[symb, sb]), conj(dft_precoders[symb, sb]).T)

        return precoders


