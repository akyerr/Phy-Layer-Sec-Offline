from numpy import concatenate, bitwise_xor, zeros
import matplotlib.pyplot as plt
from FDomain.PLSParameters import PLSParameters
from FDomain.Node import Node

max_SNR = 45
SNR_dB = range(0, max_SNR, 5)
max_iter = 200

pls_profiles = {
               0:{'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 1},
               1:{'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               }

for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob
    KER_A = zeros(len(SNR_dB), dtype=float)
    for s in range(len(SNR_dB)):
        num_errorsA = zeros(max_iter, dtype=int)
        num_errorsB = zeros(max_iter, dtype=int)
        for i in range(max_iter):
            HAB, HBA = pls_params.channel_gen()

            ## 1. Alice to Bob
            GA = N.unitary_gen()
            rx_sigB0 = N.receive('Bob', SNR_dB[s], HAB, GA)

            ## 1. At Bob
            UB0 = N.sv_decomp(rx_sigB0)[0]
            bits_subbandB = N.secret_key_gen()
            FB = N.precoder_select(bits_subbandB, codebook)

            ## 2. Bob to Alice
            rx_sigA = N.receive('Alice', SNR_dB[s], HBA, UB0, FB)

            ## 2. At Alice
            UA, _, VA = N.sv_decomp(rx_sigA)
            bits_sb_estimateB = N.PMI_estimate(VA, codebook)[1]
            actual_keyB = concatenate(bits_subbandB)
            observed_keyB = concatenate(bits_sb_estimateB)
            num_errorsA[i] = bitwise_xor(actual_keyB, observed_keyB).sum()

            bits_subbandA = N.secret_key_gen()
            FA = N.precoder_select(bits_subbandA, codebook)

            ## 3. Alice to Bob
            rx_sigB1 = N.receive('Bob', SNR_dB[s], HAB, UA, FA)

            ## 3. At Bob
            VB1 = N.sv_decomp(rx_sigB1)[2]
            bits_sb_estimateA = N.PMI_estimate(VB1, codebook)[1]
            actual_keyA = concatenate(bits_subbandA)
            observed_keyA = concatenate(bits_sb_estimateA)
            num_errorsB[i] = bitwise_xor(actual_keyA, observed_keyA).sum()

        ## Calculate KER at Alice
        total_key_len = max_iter*pls_params.num_subbands*pls_params.bit_codebook*2
        KER_A[s] = num_errorsA.sum()/total_key_len
    plt.semilogy(SNR_dB, KER_A, label=f'{pls_params.bit_codebook} bit codebbok')
plt.legend()
plt.show()










