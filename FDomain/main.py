from numpy import concatenate, bitwise_xor, zeros
import matplotlib.pyplot as plt
from FDomain.PLSParameters import PLSParameters
from FDomain.Node import Node


def bin_array2dec(bin_array):
    arr_reversed = bin_array[::-1]
    dec = 0
    for j in range(len(arr_reversed)):
        dec += (2 ** j) * arr_reversed[j]
    return dec



max_SNR = 45
SNR_dB = range(0, max_SNR, 5)
# SNR_dB = [45, 45]
max_iter = 200

pls_profiles = {
               0: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               1: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               }



for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob
    KER_A = zeros(len(SNR_dB), dtype=float)
    for s in range(len(SNR_dB)):
        num_errorsA = zeros(max_iter, dtype=int)
        num_errorsB = zeros(max_iter, dtype=int)
        transmitted_PMI = zeros((len(SNR_dB), max_iter), dtype=int)
        observed_precoder = zeros((len(SNR_dB), max_iter), dtype=object)
        for i in range(max_iter):
            HAB, HBA = pls_params.channel_gen()

            ## 1. Alice to Bob
            GA = N.unitary_gen()
            rx_sigB0 = N.receive('Bob', SNR_dB[s], HAB, GA)

            ## 1. At Bob
            UB0 = N.sv_decomp(rx_sigB0)[0]
            bits_subbandB = N.secret_key_gen()
            transmitted_PMI[s, i] = bin_array2dec(bits_subbandB[0])


            FB = N.precoder_select(bits_subbandB, codebook)

            ## 2. Bob to Alice
            rx_sigA = N.receive('Alice', SNR_dB[s], HBA, UB0, FB)

            ## 2. At Alice
            UA, _, VA = N.sv_decomp(rx_sigA)

            observed_precoder[s, i] = VA[0]

            plt.plot(observed_precoder[s, i].real, observed_precoder[s, i].imag, 'o', color='black')
            
            # plt.savefig('foo.png')






#     print(KER_A)
#     plt.semilogy(SNR_dB, KER_A, label=f'{pls_params.bit_codebook} bit codebbok')
# plt.legend()
# plt.show()










