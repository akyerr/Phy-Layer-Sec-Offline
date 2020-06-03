from numpy import pi, sqrt, exp, floor, zeros, array
from numpy.fft import fft
from numpy.linalg import norm

class PLSParameters:

    def __init__(self, prof):
        """
        Initialization of class
        :param prof: PLS profile containing basic parameters such as bandwidth, antennas bin spacing, bits in codebook
        index and the synch data pattern (How many synchs vs how many datas)
        """
        self.bandwidth = prof['bandwidth']
        self.bin_spacing = prof['bin_spacing']
        self.num_ant = prof['num_ant']
        self.bit_codebook = prof['bit_codebook']
        self.synch_data_pattern = prof['synch_data_pattern']

        self.NFFT = int(floor(self.bandwidth/self.bin_spacing))
        self.CP = int(0.25*self.NFFT)
        # self.num_data_bins = int(0.75*self.NFFT)
        self.num_data_bins = 4
        # self.num_data_bins = 1
        self.subband_size = self.num_ant


        if self.num_data_bins == 1:
            self.used_data_bins = array([10])

        DC_index = int(self.NFFT / 2)
        neg_data_bins = list(range(DC_index - int(self.num_data_bins / 2), DC_index))
        pos_data_bins = list(range(DC_index + 1, DC_index + int(self.num_data_bins / 2) + 1))
        self.used_data_bins = array(neg_data_bins + pos_data_bins)


        self.num_subbands = int(floor(self.num_data_bins/self.subband_size))
        # print(self.num_subbands)
        self.num_PMI = self.num_subbands
        self.max_impulse = self.NFFT
        # self.max_impulse = 1
        self.channel_time = zeros((self.num_ant, self.num_ant, self.max_impulse), dtype=complex)
        self.channel_freq = zeros((self.num_ant, self.num_ant, self.NFFT), dtype=complex)

        # Channel matrices after FFT for each used bin
        self.h_f = zeros((self.num_ant, self.num_ant, self.num_data_bins), dtype=complex)

        self.codebook = self.codebook_gen()

        test_case = 0
        h = zeros((self.num_ant, self.num_ant), dtype=object)

        # Time domain channels between Alice and Bob
        if test_case == 0:
            # h = 1
            # h[0, 0] = array([1])
            # h[0, 1] = array([1])
            # h[1, 0] = array([1])
            # h[1, 1] = array([1])
            h[0, 0] = array([0.3977])
            h[0, 1] = array([0.8423j])
            h[1, 0] = array([0.1631])
            h[1, 1] = array([0.0572j])
            # h[0, 0] = array([0.3977, 0.7954 - 0.3977j, -0.1988, 0.0994, -0.0398])
            # h[0, 1] = array([0.8423j, 0.5391, 0, 0, 0])
            # h[1, 0] = array([0.1631, -0.0815 + 0.9784j, 0.0978, 0, 0])
            # h[1, 1] = array([0.0572j, 0.3659j, 0.5717 - 0.5717j, 0.4574, 0])
        else:
            print('# Load from MATLAB channel toolbox - currently not done')
            exit(0)

        for rx in range(self.num_ant):
            for tx in range(self.num_ant):
                if test_case == 0:
                    self.channel_time[rx, tx, 0:len(h[rx, tx])] = h[rx, tx] / norm(h[rx, tx])
                else:
                    print('# Load normalized channels from MATLAB toolbox - currently not done')
                    exit(0)

                # Take FFT of channels
                self.channel_freq[rx, tx, :] = fft(self.channel_time[rx, tx, 0:len(h[rx, tx])], self.NFFT)
                self.h_f[rx, tx, :] = self.channel_freq[rx, tx, self.used_data_bins.astype(int)]



    def codebook_gen(self):
        """
        Generate a DFT codebook.
        :return: Matrix of codebook entries. Each entry is a antenna x antenna matrix. Number of entries = 2^n where n
        is the number of bits in each codebook index (self.bit_codebook)
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

