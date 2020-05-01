from numpy import zeros, dot, conj, prod, sqrt, exp, pi, diag, angle, array, argwhere, real, pad, fromfile, unpackbits
from numpy.linalg import qr, multi_dot, svd
from numpy.random import uniform, normal, randint

class Node:
    def __init__(self, pls_params):
        """
        Initialization of class
        :param pls_params: object from PLSParameters class containing basic parameters
        """
        self.bandwidth = pls_params.bandwidth
        self.bin_spacing = pls_params.bin_spacing
        self.num_ant = pls_params.num_ant
        self.bit_codebook = pls_params.bit_codebook

        self.NFFT = pls_params.NFFT
        self.num_used_bins = pls_params.num_used_bins
        self.subband_size = self.num_ant

        self.num_subbands = pls_params.num_subbands
        self.num_PMI = self.num_subbands

        self.key_len = self.num_subbands * self.bit_codebook

    def unitary_gen(self):
        """
        Generate random nitary matrices for each sub-band
        :return GA: Unitary matrices in each sub-band at Alice
        """
        GA = zeros(self.num_subbands, dtype=object)
        for sb in range(0, self.num_subbands):
            Q, R = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                      +1j*uniform(0, 1, (self.num_ant, self.num_ant)))

            GA[sb] = dot(Q, diag(diag(R)/abs(diag(R))))
        return GA

    @staticmethod
    def awgn(in_signal, SNRdB):
        """
        Adds AWGN to the input signal. Maintains a given SNR.
        :param in_signal: input signal to which noise needs to be addded
        :param SNRdB: Signal to Noise Ratio in dB
        :return: noisy signal
        """
        S0 = in_signal*conj(in_signal)
        S = S0.sum() / prod(in_signal.shape)
        SNR = 10 ** (SNRdB / 10)
        N = S.real / SNR
        awg_noise = sqrt(N / 2) * normal(0, 1, in_signal.shape) + \
                    1j * sqrt(N / 2) * normal(0, 1, in_signal.shape)

        return in_signal + awg_noise

    def sv_decomp(self, rx_sig):
        """
        Perform SVD for the matrix in each sub-band
        :param rx_sig: Channel matrix at the receiver in each sub-band
        :return lsv, sval, rsv: Left, Right Singular Vectors and Singular Values for the matrix in each sub-band
        """
        lsv = zeros(self.num_subbands, dtype=object)
        sval = zeros(self.num_subbands, dtype=object)
        rsv = zeros(self.num_subbands, dtype=object)

        for sb in range(0, self.num_subbands):
            U, S, VH = svd(rx_sig[sb])
            V = conj(VH).T
            ph_shift_u = diag(exp(-1j * angle(U[0, :])))
            ph_shift_v = diag(exp(-1j * angle(V[0, :])))
            lsv[sb] = dot(U, ph_shift_u)
            sval[sb] = S
            rsv[sb] = dot(V, ph_shift_v)

        return lsv, sval, rsv

    def receive(self, *args):
        """
        Contains 3 cases for the 3 steps of the process depending who is the receiver (Alice or Bob)
        Generates the frequency domain rx signal in each sub-band which is of the form H*G*F
        H - channel, G - random unitary or LSV from SVD, F - DFT precoder
        :param args: 0 - who is receiving, 1 - signal to noise ratio in dB, 2 - freq domain channel,
        3 - random unitary or LSV from SVD, 4 - DFT precoder
        :return: frequency domain rx signal in each sub-band
        """
        rx_node = args[0]
        SNRdB = args[1]
        if rx_node == 'Bob' and len(args) == 4:
                HAB = args[2]
                GA = args[3]
                rx_sigB = zeros(self.num_subbands, dtype=object)
                for sb in range(0, self.num_subbands):
                    tx_sig = multi_dot([HAB[sb], GA[sb]])
                    # rx_sigB[sb] = tx_sig
                    rx_sigB[sb] = self.awgn(tx_sig, SNRdB)
                return rx_sigB
        if rx_node == 'Bob' and len(args) == 5:
            HAB = args[2]
            UA = args[3]
            FA = args[4]
            rx_sigB = zeros(self.num_subbands, dtype=object)
            for sb in range(0, self.num_subbands):
                tx_sig = multi_dot([HAB[sb], conj(UA[sb]), conj(FA[sb]).T])
                rx_sigB[sb] = self.awgn(tx_sig, SNRdB)
            return rx_sigB
        elif rx_node == 'Alice' and len(args) == 5:
            HBA = args[2]
            UB = args[3]
            FB = args[4]
            rx_sigA = zeros(self.num_subbands, dtype=object)
            for sb in range(0, self.num_subbands):
                tx_sig = multi_dot([HBA[sb], conj(UB[sb]), conj(FB[sb]).T])
                rx_sigA[sb] = self.awgn(tx_sig, SNRdB)
                # rx_sigA[sb] = tx_sig
            return rx_sigA


    def secret_key_gen(self, type_of_data):
        """
        Generate private info bits by converting image or text to bits
        :return bits_for_tx: bits to be transmitted
        """


        if type_of_data == 'image':
            in_name = 'tux.png'
            in_bytes = fromfile(in_name, dtype="uint8")
            in_bits = unpackbits(in_bytes)
            data = list(in_bits)
            bits_for_tx = data
            # bit_difference = abs(int(self.key_len - len(data)))
            # bits_for_tx = pad(data, (0, bit_difference), 'constant')

        else:
            bit_list = self.to_bits("Hello World")
            bits_for_tx = bit_list
            # Read data and convert to a list of bits
            # bit_difference = abs(int(self.key_len - len(bit_list)))
            # bits_for_tx = pad(bit_list, (0, bit_difference), 'constant')

        return bits_for_tx

    def map_key2subband(self, bits_for_tx):
        secret_key = bits_for_tx
        bit_difference = abs(int(self.key_len - len(bits_for_tx)))
        secret_key = pad(bits_for_tx, (0, bit_difference), 'constant')
        print(len(bits_for_tx))
        bits_subband = zeros(self.num_subbands, dtype=object)
        # Map secret key to subbands
        for sb in range(self.num_subbands):
            start = sb * self.bit_codebook
            fin = start + self.bit_codebook

            bits_subband[sb] = secret_key[start: fin]

        return bits_subband

    def precoder_select(self, bits_subband, codebook):
        """
        selects the DFT precoder from the DFT codebook based. Bits are converted to decimal and used as look up index.
        :param bits_subband: Bits in each sub-band
        :param codebook: DFT codebook of matrix precoders
        :return precoder: Selected DFT preocder from codebook for each sub-band
        """
        precoder = zeros(self.num_subbands, dtype=object)

        for sb in range(self.num_subbands):
            bits = bits_subband[sb]
            start = self.bit_codebook - 1
            bi2dec_wts = 2**(array(range(start, -1, -1)))
            codebook_index = sum(bits*bi2dec_wts)
            precoder[sb] = codebook[codebook_index]

        return precoder
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
        # print(x[0, 0], bits)
        return bits

    def PMI_estimate(self, rx_precoder, codebook):
        """
        Apply minumum distance to estimate the transmitted precoder, its index in the codebook and the binary equivalent
        of the index
        :param rx_precoder: observed precoder (RSV of SVD)
        :param codebook: DFT codebook of matrix precoders
        :return PMI_sb_estimate, bits_sb_estimate: Preocder matrix index and bits for each sub-band
        """
        PMI_sb_estimate = zeros(self.num_subbands, dtype=int)
        bits_sb_estimate = zeros(self.num_subbands, dtype=object)

        for sb in range(self.num_subbands):
            dist = zeros(len(codebook), dtype=float)

            for prec in range(len(codebook)):
                diff = rx_precoder[sb] - codebook[prec]
                diff_squared = real(diff*conj(diff))
                dist[prec] = sqrt(diff_squared.sum())
            min_dist = min(dist)
            PMI_estimate = argwhere(dist == min_dist)
            PMI_sb_estimate[sb] = PMI_estimate
            bits_sb_estimate[sb] = self.dec2binary(PMI_estimate, self.bit_codebook)

        return PMI_sb_estimate, bits_sb_estimate

    def to_bits(self, s):
        result = []
        for c in s:
            bits = bin(ord(c))[2:]
            bits = '00000000'[len(bits):] + bits
            result.extend([int(b) for b in bits])
        return result







