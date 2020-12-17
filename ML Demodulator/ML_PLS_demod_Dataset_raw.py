from numpy import zeros, conj, sqrt, exp, pi, array
from numpy.random import normal, randint
import pickle
import time


start = time.process_time()

max_iter = 100000
SNR_dB = [10]
# SNR_dB = [0, 10, 20, 30]
SNR_lin = 10**(array(SNR_dB)/10)
bit_codebook = 2
num_ant = 2
num_classes = 2**bit_codebook

def codebook_gen(num_ant, bit_codebook):
    """
    Generate DFT codebbok of matrix preocders
    :return: matrix of matrix preocders
    """
    num_precoders = 2 ** bit_codebook
    codebook = zeros(num_precoders, dtype=object)

    for p in range(0, num_precoders):
        precoder = zeros((num_ant, num_ant), dtype=complex)
        for m in range(0, num_ant):
            for n in range(0, num_ant):
                w = exp(1j * 2 * pi * (n / num_ant) * (m + p / num_precoders))
                precoder[n, m] = (1 / sqrt(num_ant)) * w

        codebook[p] = precoder

    return codebook

codebook = codebook_gen(num_ant, bit_codebook)

for s in range(len(SNR_lin)):

    precoder_img = list()
    tx_PMI = list()
    for i in range(max_iter):
        print(SNR_dB[s], ' dB', i)
        PMI = randint(0, num_classes)  # generate random precoder index
        tx_PMI.append(PMI)

        precoder = codebook[PMI]

        prec_power = sum(sum(precoder * conj(precoder))) / (num_ant ** 2)
        #     print(prec_power)
        noise_var = abs(prec_power) / SNR_lin[s]

        noise = normal(0, sqrt(noise_var), (num_ant, num_ant)) + 1j * normal(0, sqrt(noise_var), (num_ant, num_ant))

        # Add noise
        noisy_precoder = precoder + noise

print(time.process_time() - start)