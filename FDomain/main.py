from numpy import concatenate, bitwise_xor, zeros, packbits, ceil, asarray
import cv2
import matplotlib.pyplot as plt
from PLSParameters import PLSParameters
from Node import Node
from plot_diagnostics import plots
import time
start = time.process_time()

SNR_dB = [60, 6]
max_iter = 1

pls_profiles = {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 4}


for s in range(len(SNR_dB)):
    pls_params = PLSParameters(pls_profiles)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob

    # set up data to be transmitted - 'text' or 'image'
    bits_to_tx = N.secret_key_gen('image')
    print('Number of bits in the image:', len(bits_to_tx))
    num_symbols = int(ceil(len(bits_to_tx)/(pls_params.num_subbands * pls_params.bit_codebook)))
    print('-----System Parameters-----')
    print(f'Bandwidth: {pls_params.bandwidth}')
    print(f'FFT: {pls_params.NFFT}')
    print(f'Used freq bins: {pls_params.num_used_bins}')
    print(f'Sub-band size: {pls_params.subband_size}')
    # Groups of 2 frequency bins form a sub-band (This is because the precoders are 2x2 matrices and are split across 2 bins
    print(f'Number of sub-bands: {pls_params.num_subbands}')
    print(f'Total number of OFDM symbols: {num_symbols}')

    bits_recovered = zeros(len(bits_to_tx), dtype=int)
    for symb in range(num_symbols):
        print(f'symbol {symb} of {num_symbols}')
        bits_start = symb*pls_params.num_subbands*pls_params.bit_codebook
        bits_end = bits_start + (pls_params.num_subbands*pls_params.bit_codebook)
        bits_in_symb = bits_to_tx[bits_start: bits_end]
        bits_subbandB = N.map_key2subband(bits_in_symb)

        HAB, HBA = pls_params.channel_gen()

        ## 1. Alice to Bob
        GA = N.unitary_gen()
        rx_sigB0 = N.receive('Bob', SNR_dB[s], HAB, GA)

        ## 1. At Bob - private info transmission starts here
        UB0 = N.sv_decomp(rx_sigB0)[0]

        FB = N.precoder_select(bits_subbandB, codebook)

        ## 2. Bob to Alice
        rx_sigA = N.receive('Alice', SNR_dB[s], HBA, UB0, FB)

        ## 2. At Alice
        UA, _, VA = N.sv_decomp(rx_sigA)
        bits_sb_estimateB = N.PMI_estimate(VA, codebook)[1]
        actual_keyB = concatenate(bits_subbandB)
        observed_keyB = concatenate(bits_sb_estimateB)
        num_errorsA = bitwise_xor(actual_keyB, observed_keyB).sum()
        # print(num_errorsA)

        bits_recovered[bits_start: bits_end] = observed_keyB[0: len(bits_in_symb)]

        if SNR_dB[s] == 60 and symb == 0:
            plots(pls_params, GA, rx_sigB0, rx_sigA)
            dbg = 1

    out_bits = bits_recovered
    out_bytes = packbits(out_bits)
    out_name = f'tux_out_{SNR_dB[s]}dB.png'
    out_bytes.tofile(out_name)

    plt.hist(out_bytes)
    plt.title(f'Histogram of Rx image byte array, SNR: {SNR_dB[s]} dB')
    plt.xlabel('Bytes')
    plt.ylabel('Freq. of occurrence')
    plt.show()

print(f'Time to run: {time.process_time() - start} seconds')










