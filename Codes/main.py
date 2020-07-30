###############################################################################
## Author: Team Supply Bot
## Edition: eYRC 2019-20
## Instructions: Do Not modify the basic skeletal structure of given APIs!!!
###############################################################################


######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import csv
import cv2.aruco as aruco
from aruco_lib import *
import copy



########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Videos'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))




############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################
def process(ip_image):
    ###########################
    ## Your Code goes here
    print("in loop")
    b, g, r = cv2.split(ip_image)

    # defining an array consisting of b,g,r channles of the blur image
    img_array = [b, g, r]

    # defining an empty array to store the final image
    final_img = np.empty((840, 1600, 3))

    # defining the angel, dis, load/noise values to be used in the program
    # below for restoring the image of different channels
    angle = [87, 90, 93]
    dis = [20, 22, 20]
    load = [22, 17, 18]
    i = 0
    # now applying the restoration algo on each channel one by one
    for color in img_array:
        img_channel = np.float32(color) / 255.0


        # forming a blur image from the given image
        d=31
        # returns a tuple of number of rows, columns and channels of the image
        height, width = img_channel.shape[:2]
        # Creating a border around the image
        img_pad = cv2.copyMakeBorder(img_channel, d, d, d, d, cv2.BORDER_WRAP)
        # Blurring bordered image using a gaussian function and storing it in img_blur
        img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]
        # Returning an a array representing the indices of matrix
        y, x = np.indices((height, width))
        # returning an array formed by stacking x and y and concatenating along 3rd dimension
        dist = np.dstack([x, width - x - 1, y, height - y - 1]).min(-1)
        # Comparing both the arrays and returning the element wise minimum value to array elements
        w = np.minimum(np.float32(dist) / d, 1.0)
        blur_img =  img_channel * w + img_channel * (1 - w)
        #----------------------------------------------


        # Applying discrete fourier transform on blur image since images are collection of discrete values in time domain
        # so we are converting them into frequency domain

        dft_image = cv2.dft(blur_img, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Defining angle ang and converting it from degree to radian assigning variable d and noise with respective values

        ang = np.deg2rad(angle[i])
        d = dis[i]
        noise = 10 ** (-0.1 * load[i])

        # Defining degradation_kernel using motion kernel

        # making a kernel of size 20x20
        sz = 20
        # Creating an array of the given parameters with ones
        kernel = np.ones((1, d), np.float32)
        # Defining trigonometric cosine as c and trigonoetric sine as s
        c, s = np.cos(ang), np.sin(ang)
        # Assigning matrix A with given parameters defind above with 32-bit floating values
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = sz // 2
        A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
        # transforming the kern image using the specified matrix
        deg_kernel = cv2.warpAffine(kernel, A, (sz, sz), flags=cv2.INTER_CUBIC)
        #--------------------------------------------------------

        deg_kernel /= deg_kernel.sum()
        # creating a zero matrix of the size of img that will be used for padding
        deg_pad = np.zeros_like(img_channel)
        kh, kw = deg_kernel.shape
        # padding the kernel to get the degradation function of the same size as that of the image
        deg_pad[:kh, :kw] = deg_kernel

        # Performing the DFT on the psf_pad and saving it in deg_kernel_dft
        # First channel will have the real part of the result and second channel will have the imaginary part of the result
        deg_dft = cv2.dft(deg_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        # taking magnitude of the complex values
        deg_dft_2 = (deg_dft ** 2).sum(-1)
        i_deg_dft_2 = deg_dft / (deg_dft_2 + noise)[..., np.newaxis]

        # Performing element wise multiplication of the two matrix IMG and iPSF that are results of a real or complex fourier transform
        Restored_img = cv2.mulSpectrums(dft_image, i_deg_dft_2, 0)
        # RES is our restored image in frequency domain now converting it in time domain by performing idft
        restored_img = cv2.idft(Restored_img, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        # Rolling res array elements along a given axis i.e 0 and 1
        restored_img = np.roll(restored_img, -kh // 2, 0)
        restored_img = np.roll(restored_img, -kw // 2, 1)

        final_img[:, :, i] = restored_img
        i += 1



    final_img = (final_img/np.max(final_img))*255
    final_img = final_img.astype(np.uint8)


    brightness = 95
    contrast = 85
    highlight = 255
    shadow = brightness
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow

    buf = cv2.addWeighted(final_img, alpha_b, final_img, 0, gamma_b)

    f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)

    final_img = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)





    det_aruco_list = detect_Aruco(final_img)
    # print(det_aruco_list)
    if det_aruco_list:
        ip_image = mark_Aruco(final_img, det_aruco_list)
        robot_state = calculate_Robot_State(final_img, det_aruco_list)
        print(robot_state)

        ###########################
    id_list = robot_state[25]



    return ip_image, id_list


    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main(val):
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(images_folder_path+"/"+"ArUco_bot.mp4")
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## getting the frame sequence
    frame_seq = int(val)*fps
    ## setting the video counter to frame sequence
    cap.set(1,frame_seq)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)
    ## display to see if the frame is correct
    cv2.imshow("window", frame)
    cv2.waitKey(0);
    ## calling the algorithm function
    op_image, aruco_info = process(frame)
    ## saving the output in  a list variable
    line = [str(i), "Aruco_bot.jpg" , str(aruco_info[0]), str(aruco_info[3])]
    ## incrementing counter variable
    i+=1
    ## verifying all data
    print(line)
    os.chdir('..')
    path = generated_folder_path
    print(path)
    cv2.imwrite(path + "/"+"aruco_with_id.png",op_image)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'output.csv', 'w') as writeFile:
        print("About to write csv")
        writer = csv.writer(writeFile)
        writer.writerow(line)

    ## closing csv file    
    writeFile.close()



    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main(input("time value in seconds:"))
