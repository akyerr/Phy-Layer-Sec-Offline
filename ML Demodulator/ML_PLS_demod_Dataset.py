from numpy import zeros, dot, conj, prod, sqrt, exp, pi, diag, angle, array, argwhere, real, floor, frombuffer, uint8, where, stack, asarray, expand_dims
from numpy.linalg import qr, multi_dot, svd
from numpy.random import uniform, normal, randint
import matplotlib.pyplot as plt
import pandas as pd
import io
import cv2
import pickle
import time

start = time.process_time()


max_iter = 10000
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
        num_precoders = 2**bit_codebook
        codebook = zeros(num_precoders, dtype=object)

        for p in range(0, num_precoders):
            precoder = zeros((num_ant, num_ant), dtype=complex)
            for m in range(0, num_ant):
                for n in range(0, num_ant):
                    w = exp(1j*2*pi*(n/num_ant)*(m + p/num_precoders))
                    precoder[n, m] = (1/sqrt(num_ant))*w

            codebook[p] = precoder

        return codebook

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = frombuffer(buf.getvalue(), dtype=uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    scaled_image = cv2.resize(img, (400, 400))  
#     scaled_image = cv2.resize(img, (200, 200))
#     print(scaled_image.shape)
    crop_dim = 256
    cropped_img = crop_center(scaled_image, crop_dim, crop_dim)
#     cropped_img = crop_center(scaled_image, 128, 128)
    cropped_img = cv2.bitwise_not(cropped_img)
    cropped_img = expand_dims(cropped_img, axis=0)
    return cropped_img, crop_dim

codebook = codebook_gen(num_ant, bit_codebook)

for i in range(len(SNR_lin)):
    
    precoder_img = list()
    tx_PMI = list()
    for iter in range(max_iter):
        print(SNR_dB[i],' dB', iter)
#         print(iter)
        PMI = randint(0, num_classes) # generate random precoder index
        tx_PMI.append(PMI)

        precoder = codebook[PMI]


        prec_power = sum(sum(precoder*conj(precoder)))/(num_ant**2)
    #     print(prec_power)
        noise_var = abs(prec_power)/SNR_lin[i]


        noise = normal(0, sqrt(noise_var), (num_ant, num_ant)) + 1j*normal(0, sqrt(noise_var), (num_ant, num_ant))

    #     print(noise)


        # Add noise
        noisy_precoder = precoder + noise

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(noisy_precoder.real, noisy_precoder.imag, 'o', color='black')
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.close(fig)
        
        cropped_image, crop_dimensions = get_img_from_fig(fig)
        precoder_img.append(cropped_image) 
    

    

    precoder_img_data = stack(precoder_img, axis=0)
    precoder_labels = array(tx_PMI)

    file_name = 'img_data_' + str(max_iter) + '_' + str(SNR_dB[i]) + 'dB_' + str(crop_dimensions) + 'x' + str(crop_dimensions) + '.pckl'
    with open(file_name, 'wb') as f:
        pickle.dump([precoder_img_data, precoder_labels], f)



print(time.process_time() - start)      
