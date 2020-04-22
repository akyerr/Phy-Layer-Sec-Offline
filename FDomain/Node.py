from numpy import zeros, dot, conj, prod, sqrt, exp, pi, diag, angle, array, argwhere, real
from numpy.linalg import qr, multi_dot, svd
from numpy.random import uniform, normal, randint

class Node:
    def __init__(self, pls_params):
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
        GA = zeros(self.num_subbands, dtype=object)
        for sb in range(0, self.num_subbands):
            Q, R = qr(uniform(0, 1, (self.num_ant, self.num_ant))
                      +1j*uniform(0, 1, (self.num_ant, self.num_ant)))

            GA[sb] = dot(Q, diag(diag(R)/abs(diag(R))))
        return GA

    @staticmethod
    def awgn(in_signal, SNRdB):
        S0 = in_signal*conj(in_signal)
        S = S0.sum() / prod(in_signal.shape)
        SNR = 10 ** (SNRdB / 10)
        N = S.real / SNR
        awg_noise = sqrt(N / 2) * normal(0, 1, in_signal.shape) + \
                    1j * sqrt(N / 2) * normal(0, 1, in_signal.shape)

        return in_signal + awg_noise

    def sv_decomp(self, rx_sig):
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


    def secret_key_gen(self):
        bits_subband = zeros(self.num_subbands, dtype=object)

        secret_key = randint(0, 2, self.key_len)

        # Map secret key to subbands
        for sb in range(self.num_subbands):
            start = sb * self.bit_codebook
            fin = start + self.bit_codebook

            bits_subband[sb] = secret_key[start: fin]

        return bits_subband

    def precoder_select(self, bits_subband, codebook):
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
        bit_str = [char for char in format(x[0, 0], '0' + str(num_bits) + 'b')]
        bits = array([int(char) for char in bit_str])
        # print(x[0, 0], bits)
        return bits

    def PMI_estimate(self, rx_precoder, codebook):
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








