from numpy import array, zeros
import matplotlib.pyplot as plt




def plots(pls_params, GA, rx_sigB0, rx_sigA):
    DC_index = int(pls_params.NFFT / 2)
    neg_data_bins = list(range(DC_index - int(pls_params.num_used_bins / 2), DC_index))
    pos_data_bins = list(range(DC_index + 1, DC_index + int(pls_params.num_used_bins / 2) + 1))
    used_data_bins = array(neg_data_bins + pos_data_bins)







    rxA = list()
    gA = list()
    for sb in range(pls_params.num_subbands):
        rxA0 = rx_sigA[sb]
        rxA.extend(rxA0[0, :].tolist())
        gA0 = GA[sb]
        gA.extend(gA0[0, :])
    rxA = array(rxA)
    gA = array(gA)

    xax = array(range(-int(pls_params.NFFT/2), int(pls_params.NFFT/2)))

    rxA_used = zeros(pls_params.NFFT, dtype=complex)
    rxA_used[used_data_bins] = rxA
    gA_used = zeros(pls_params.NFFT, dtype=complex)
    gA_used[used_data_bins] = gA

    plt.subplot(2, 1, 1)
    plt.stem(xax, gA_used.real)
    plt.xlabel('Subcarrer index')
    plt.ylabel('Real')
    plt.title('Tx Signal on Ant 1 in Symb 1 With Rand Unitary (Uniformly distributed)')
    plt.subplot(2, 1, 2)
    plt.stem(xax, gA_used.imag)
    plt.xlabel('Subcarrer index')
    plt.ylabel('Imag')
    plt.title('Tx Signal on Ant 1 in Symb 1 With Rand Unitary (Uniformly distributed)')
    plt.show()


    # print(rxA)
    # print(rxA.shape)
    plt.subplot(2, 1, 1)
    plt.stem(xax, rxA_used.real)
    plt.xlabel('Subcarrer index')
    plt.ylabel('Real')
    plt.title('Rx Signal on Antenna 1 in Symbol 1 at Alice (Step 2)')
    plt.subplot(2, 1, 2)
    plt.stem(xax, rxA_used.imag)
    plt.xlabel('Subcarrer index')
    plt.ylabel('Imag')
    plt.title('Rx Signal on Antenna 1 in Symbol 1 at Alice (Step 2)')
    plt.show()
    dbg = 1