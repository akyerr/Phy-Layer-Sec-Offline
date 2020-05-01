from numpy import pi, sqrt, exp, floor, zeros
from numpy.random import normal

class PLSParameters:

    def __init__(self, prof):
        """
        Initialization of class
        :param prof: PLS profile containing basic parameters such as bandwidth, antennas bin spacing, bits in codebook
        index
        """
        self.bandwidth = prof['bandwidth']
        self.bin_spacing = prof['bin_spacing']
        self.num_ant = prof['num_ant']
        self.bit_codebook = prof['bit_codebook']

        if self.bandwidth == 20e6:
            self.NFFT = 2048
            self.num_used_bins = 1332
        else:
            self.NFFT = int(floor(self.bandwidth/self.bin_spacing))
            self.num_used_bins = self.NFFT - 2
        self.subband_size = self.num_ant

        self.num_subbands = int(floor(self.num_used_bins/self.subband_size))
        self.num_PMI = self.num_subbands

    def codebook_gen(self):
        """
        Generate DFT codebbok of matrix preocders
        :return: matrix of matrix preocders
        """
        num_precoders = 2**self.bit_codebook
        codebook = zeros(num_precoders, dtype=object)

        for p in range(0, num_precoders):
            precoder = zeros((self.num_ant, self.num_ant), dtype=complex)
            for m in range(0, self.num_ant):
                for n in range(0, self.num_ant):
                    w = exp(1j*2*pi*(n/self.num_ant)*(m + p/num_precoders))
                    precoder[n, m] = (1/sqrt(self.num_ant))*w

            codebook[p] = precoder

        return codebook

    def channel_gen(self):
        """
        Generate generic Rayleigh fading channels in the frequency domain
        :return:
        """
        HAB = zeros(self.num_subbands, dtype=object)
        HBA = zeros(self.num_subbands, dtype=object)

        for sb in range(0, self.num_subbands):
            H = normal(0, 1, (self.num_ant, self.num_ant)) \
                + 1j*normal(0, 1, (self.num_ant, self.num_ant))
            HAB[sb] = H
            HBA[sb] = H.T

        return HAB, HBA
