from numpy import ceil, tile, concatenate, zeros, ones
from numpy.random import randint
# from TDomain.PLSParameters import PLSParameters
# from TDomain.PLSTransmitter import PLSTransmitter

from PLSParameters import PLSParameters
from PLSTransmitter import PLSTransmitter
from PLSReceiver import PLSReceiver
from SynchSignal import SynchSignal

SNRdB = 40
SNR_type = 'Analog'

pls_profiles = {
    0: {'bandwidth': 960e3,
        'bin_spacing': 15e3,
        'num_ant': 2,
        'bit_codebook': 2,
        'synch_data_pattern': [2, 1]},
    # 1:{'bandwidth': 960e3,
    #    'bin_spacing': 15e3,
    #    'num_ant': 2,
    #    'bit_codebook': 2,
    #    'synch_data_pattern': [4, 2]},
}
pvt_info_len = 144  # bits
for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    # print(pls_params.num_subbands)
    # if length of pvt info is not a perfect multiple of num sub-bands*bit codebook then append some bits to make it
    # a multiple
    num_data_symb = int(ceil(pvt_info_len / (pls_params.num_subbands * pls_params.bit_codebook)))
    # 2, 1, 2, 1, 2, 1, 2, 1, 2, 1
    num_synch_symb = pls_params.synch_data_pattern[0] * num_data_symb

    total_num_symb = num_synch_symb + num_data_symb
    num_synchdata_patterns = int(total_num_symb / sum(pls_params.synch_data_pattern))

    symb_pattern0 = concatenate((zeros(pls_params.synch_data_pattern[0]), ones(pls_params.synch_data_pattern[1])))
    symb_pattern = tile(symb_pattern0, num_synchdata_patterns)


    # Generate synch
    synch = SynchSignal(pls_params, num_synch_symb, num_data_symb, symb_pattern)

    pls_tx = PLSTransmitter(pls_params, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb)
    pls_rx = PLSReceiver(pls_params, synch, symb_pattern, total_num_symb, num_data_symb, num_synch_symb, SNRdB, SNR_type)

    # 1. Alice to Bob first transmission
    buffer_tx_time_A, ref_sig_A = pls_tx.transmit_signal_gen('Alice0', num_data_symb)

    # 1. Bob first reception
    lsv_B0, rsv_B0 = pls_rx.receive_sig_process(buffer_tx_time_A, ref_sig_A)

    # 2. Bob to Alice - pvt info transfer starts here
    pvt_info_bits = randint(0, 2, pvt_info_len)  # private info bits
    buffer_tx_time_B, ref_sig_B = pls_tx.transmit_signal_gen('Bob', num_data_symb, pvt_info_bits, lsv_B0)

    #2. Alice reception
    lsv_A, rsv_A = pls_rx.receive_sig_process(buffer_tx_time_B, ref_sig_B)



