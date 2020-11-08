import io
import cv2
from numpy import zeros, frombuffer, uint8, where
import matplotlib.pyplot as plt
import pandas as pd
from FDomain.PLSParameters import PLSParameters
from FDomain.Node import Node


def bin_array2dec(bin_array):
    arr_reversed = bin_array[::-1]
    dec = 0
    for j in range(len(arr_reversed)):
        dec += (2 ** j) * arr_reversed[j]
    return dec

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = frombuffer(buf.getvalue(), dtype=uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

max_SNR = 20
SNR_dB = range(0, max_SNR, 5)
# SNR_dB = [45, 45]
max_iter = 2

pls_profiles = {
               0: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               1: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               }

# dbg = 1
# for prof in pls_profiles.keys():
#     df = pd.DataFrame(list(pls_profiles[prof].items()),columns = ['column 1', 'column 2'])
# dbg = 1
df = pd.DataFrame(columns = ['Bandwidth', 'Bin Spacing', 'Antennas',
                             'Bit Codebook', 'SNR', 'Obs Precoder', 'Correct PMI'])
dbg = 1
for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob

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
            transmitted_PMI = bin_array2dec(bits_subbandB[0])

            FB = N.precoder_select(bits_subbandB, codebook)

            ## 2. Bob to Alice
            rx_sigA = N.receive('Alice', SNR_dB[s], HBA, UB0, FB)

            ## 2. At Alice
            UA, _, VA = N.sv_decomp(rx_sigA)

            observed_precoder[s, i] = VA[0]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(observed_precoder[s, i].real, observed_precoder[s, i].imag, 'o', color='black')
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))
            plt.show()
            plot_img_np = get_img_from_fig(fig)

            df = df.append({'Bandwidth': pls_params.bandwidth,
                            'Bin Spacing': pls_params.bin_spacing,
                            'Antennas': pls_params.num_ant,
                             'Bit Codebook': pls_params.bit_codebook,
                            'SNR': SNR_dB[s],
                            'Obs Precoder': plot_img_np,
                            'Correct PMI': transmitted_PMI},
                           ignore_index=True)
            dbg = 1

dbg = 1










