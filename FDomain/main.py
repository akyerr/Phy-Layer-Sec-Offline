from numpy import concatenate, bitwise_xor, zeros, packbits, ceil
import matplotlib.pyplot as plt
from PLSParameters import PLSParameters
from Node import Node

SNR_dB = 60
max_iter = 1

pls_profiles = {
               0: {'bandwidth': 20e6, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 3},
               # 1: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               }

for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob

    # set up data to be transmitted - 'text' or 'image'
    bits_to_tx = N.secret_key_gen('image')

    num_symbols = int(ceil(len(bits_to_tx)/(pls_params.num_subbands * pls_params.bit_codebook)))

    bits_recovered = zeros(len(bits_to_tx), dtype=int)
    for symb in range(num_symbols):
        bits_start = symb*pls_params.num_subbands*pls_params.bit_codebook
        bits_end = bits_start + (pls_params.num_subbands*pls_params.bit_codebook)
        bits_in_symb = bits_to_tx[bits_start: bits_end]
        bits_subbandB = N.map_key2subband(bits_in_symb)

        HAB, HBA = pls_params.channel_gen()

        ## 1. Alice to Bob
        GA = N.unitary_gen()
        rx_sigB0 = N.receive('Bob', SNR_dB, HAB, GA)

        ## 1. At Bob - private info transmission starts here
        UB0 = N.sv_decomp(rx_sigB0)[0]

        FB = N.precoder_select(bits_subbandB, codebook)

        ## 2. Bob to Alice
        rx_sigA = N.receive('Alice', SNR_dB, HBA, UB0, FB)

        ## 2. At Alice
        UA, _, VA = N.sv_decomp(rx_sigA)
        bits_sb_estimateB = N.PMI_estimate(VA, codebook)[1]
        actual_keyB = concatenate(bits_subbandB)
        observed_keyB = concatenate(bits_sb_estimateB)
        num_errorsA = bitwise_xor(actual_keyB, observed_keyB).sum()
        print(num_errorsA)

        bits_recovered[bits_start: bits_end] = observed_keyB[0: len(bits_in_symb)]

    out_bits = bits_recovered
    out_bytes = packbits(out_bits)
    out_name = 'tux_out.png'
    out_bytes.tofile(out_name)











