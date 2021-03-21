import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import copy
#......IMPORT .........
import argparse

'''
Task 1: Iso-data Intensity Thresholding
'''
def task1(img_pass):
    # Read the gray scale image from the folder
    image = cv2.imread(img_pass, 0) 
    
    # Check visually if the image is bimodal or not
    f1 = plt.figure()
    plt.hist(image.ravel(), bins = 256)
    plt.grid(True)
    plt.xlabel("Intensity -------> ")
    plt.ylabel("Count -------->")
    plt.title('Histogram plot_Task1')
    plt.show()
    f1.savefig(path+"/histogram_task1_"+img_pass)
    
    
    # Algorithm for ISO-data intensity implementation
    # Point 1: Select an arbitrary initial threshold t
    # Choose an appropriate threshold at the beginning
    min_intensity = np.min(image)
    max_intensity =  np.max(image)
    #print(min_intensity, max_intensity)
    th = (min_intensity + max_intensity)/2
    
    th_new = 0
    # Check the threshold value for various epsilon values
    epsilon = [0.001, 0.01, 0.05, 0.1, 0.5]
    final_threshold = []
    collect_threshold_all = {0:[], 1:[], 2:[], 3:[], 4:[]}
    
    count = 0
    for e in epsilon:
        #print("epsilon =", e)
        temp = [th]     # to collect updates in the threshold value for agiven epsilon
        while 1:
            # Point 2: 
            mu_0 = image[image < th].mean()
            mu_1 = image[image >= th].mean()
            #print(mu_0, mu_1)
            # Point 3: 
            th_new = (mu_0 + mu_1)/2
            
            # Point 4:
            if abs(th_new - th) < e:
                final_threshold.append(th_new)
                th = th_new
                th_new = 0            
                break 
            
            temp.append(th_new)
            th = th_new
        # collect all threshold values changes for its plot    
        collect_threshold_all[count] = temp
        count += 1
    
    # Choose the value of threshold    
    th = max(set(final_threshold), key = final_threshold.count)
    
    # Calculate threshold vaule and epsilon value for later use
    th_pos = final_threshold.index(th)
    epsilon_pick = epsilon[th_pos]
    
    
    #  Plot of threshold value t at every iterations on a graph for the choosen threshold
    f2 = plt.figure()
    plt.plot(collect_threshold_all[th_pos], '-o', color = 'black', mfc = 'r')
    plt.grid(True)
    plt.xlabel("Iteration -------> ")
    plt.ylabel("Threshold value -------->")
    plt.title(f'Threshold value w.r.t. iterations: epsilon ={epsilon_pick}_Task1')
    plt.show()
    f2.savefig(path+"/threshold_"+img_pass)
    
    
    
    # Print the choosen threshold value 
    print("Threshold value = ", th)
    
    
    # Change into binary image with the background encoded as white [pixel value: 255] and the rice kernels as black [pixel value: 0]. 
    img = np.zeros(image.shape) #dinmension height*width
    img[image > th] = 1
    img[image <= th] = 2
    
    image[img == 1 ] = 0
    image[img == 2] = 255
    
    th =round(th, 2)
    
    #image_withtitle = image
    img_name_t2 = path+"/"+img_pass+"_Task1.png"
    #cv2.imwrite(img_name_t2, image_withtitle) 
    #cv2.imshow("Binary image", image_withtitle)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    f3 = plt.figure()
    plt.imshow(image, cmap = "gray")
    plt.title(f'Threshold Value = {th}')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    f3.savefig(img_name_t2)
    
    
    
    
    return image


#%%
'''
Task 2: Counting the rice kernels
'''
# Helper function --------------

# To calculate all neighbors based on 8 connectivity --------
# Neighbours have lables and are not background pixels ----
def neighbor_list(i, j, modified_img, img_with_labels):
    all_neighb = []
    
    # four directions of eight_conn_neighbors -----
    north = i - 1
    south = i + 1
    west = j - 1
    east = j + 1
    
    # coordinates of neighbors 8 connectivity ---------
    coordinates_neighbours = [(north,west), (north,j), (north,east), (i,west), (i,east), (south,west), (south,j), (south,east)]
 
    for c in coordinates_neighbours:
        if img_with_labels[c] != 0 and modified_img[c] != 255: # Not background and have labels
            all_neighb.append(c)
    
    return all_neighb


def task2(image):
    
    # Step 1: Preprocessing -- applying a median filter to remove noisy pixels
    image_filter=cv2.medianBlur(image, 7)
    
    image_filter_temp = image_filter
  
    #=============================================================
    # Preprocesing the image 
    # Change it to binary------ background to 255 and foreground to 0
    label_collect = [] 
    image_filter[image_filter <= 127] = 0
    image_filter[image_filter > 127] = 255
    
    # Padding the image------------
    modified_img = np.ones((image_filter.shape[0]+2, image_filter.shape[1]+2))
    modified_img*255 # all white
    modified_img[1:-1,1:-1] = image_filter  # fit the image to the modified image
      
    
    # Change the padded pixels to background 255
    modified_img[modified_img == 1] = 255
    # print(modified_img)
    # Initialize image with labels with all zero (black) of modified_img size
    img_with_labels = np.zeros_like(modified_img)
    
    #================================================================
    #--------Two-pass connected componenet algorithm--------------
    #================================================================
       
    height, width = image_filter.shape
    label_count = 1.0
    connected_components = defaultdict(set)
    
    #=================================================================
    # First Pass ---------------------------------------------------
    # Step1: Constructing the image with labels ------
    for i in range(1, height+1):
        for j in range(1, width+1):
            
            # Do only when the pixel is not background ---
            if modified_img[i,j] != 255:
                
                eight_conn_neighbors = neighbor_list(i,j, modified_img, img_with_labels)
                
                # if no nieghbors with labels and not background ----
                if len(eight_conn_neighbors) == 0:
                    # add neighbours as connected componenets
                    connected_components[label_count].add(label_count)
                    img_with_labels[i,j] = label_count
                    label_count += 1
                # otherwise    
                else:
                    label_collect = [img_with_labels[p] for p in eight_conn_neighbors]
                    # assign the min value among the neighbours
                    img_with_labels[i,j] = min(label_collect)
                    for labels in list(set(label_collect)):
                        for labels_ in list(set(label_collect)):
                            connected_components[labels].add(labels_)

    # Step2: Now for equivalence --------------
    list_equivalene = list()
    for i in connected_components:
        list_equivalene.append(list(connected_components[i]))
    
    temp_t1 = [] 
    # for all elements in list_equivalence ----
    while len(list_equivalene) > 0:
        
        first_element, remaining_elements = list_equivalene[0], list_equivalene[1:]
        # make it a set for set operations later
        first_element = set(first_element)
        check_cond = -1
        while len(first_element) > check_cond:
            check_cond = len(first_element)
            temp_t2 = []
            for ele in remaining_elements:
                if len(first_element.intersection(set(ele))) > 0:
                    first_element |= set(ele)  # Union operation
                else:
                    temp_t2.append(ele) 
                    
            remaining_elements = temp_t2

        temp_t1.append(first_element)
        list_equivalene = remaining_elements
    
    
    label_ccount = 1
    # equivalence list to the img_with_labels list
    eqv_list_labels = defaultdict(list) 
    for i in temp_t1:
        eqv_list_labels[label_ccount] = i
        label_ccount += 1
    
    
    # Second Pass ---------------------------------------------------
    # All labels finding
    def label_fix(img_with_labels):
        for label in eqv_list_labels:
            
            if img_with_labels in eqv_list_labels[label]:
                # return label
                return label
    
    
    for i in range(1, height+1):
        for j in range(1, width+1):
            if modified_img[i,j] != 255:   # if not background ---
                img_with_labels[i,j] = label_fix(img_with_labels[i,j])
    
    #=================================================================
            
    # Total rice kernels count    
    total_grain = np.max(img_with_labels)         
    print(f'Number of rice kernels in the image_{file_name} = {total_grain}')
    
    
    f4 = plt.figure()
    plt.imshow(image_filter_temp, cmap = "gray")
    plt.title(f'No. of rice kernels = {total_grain}')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    f4.savefig(path+"/"+file_name+"_Task2.png")
    
      
    return img_with_labels
    
#connected_comp_label = task2(img_task1)
#%%    

'''
Task 3: Percentage of damaged rice kernels
'''    
def task3(img_with_labels, min_area = None):
    
    
    #----------------- count undamaged kernels ----------
    labels_disjoint, all_count = np.unique(img_with_labels, return_counts = True)
    temp_dict = dict(zip(labels_disjoint, all_count))
    #print(temp_dict)
    
    # if min area is not given then choose one appropriate
    if min_area == None:
        temp_temp = copy.copy(temp_dict)
        del temp_temp[0]  # first element of temp_dict has the sum of all
        max_val = max(temp_temp.values())
        #print("max_val = ", max_val)
        #------ min_area threshold as the 50% of the max area
        min_area = max_val*0.5
    
    #min_area = int(min_area)
    #print("Area_min = ", min_area)    
    area_with_max = [i for i in temp_dict if temp_dict[i] >= int(min_area)][:]
   
    img_with_fine_kernels = np.zeros_like(img_with_labels)
       
    count_labels = 1 
    for area in area_with_max:
        img_with_fine_kernels[img_with_labels == area] = count_labels
        count_labels += 1
    
    # number of good grains
    good_grains = int(np.max(img_with_fine_kernels))
    print("Undamaged rice kernels = ", good_grains)
    
    total_grain = int(np.max(img_with_labels))     
    print("Damaged rice kernels = ", total_grain - good_grains)
    print("Total rice kernels in the image =", total_grain)
    
    b_p = ((total_grain-good_grains)/total_grain)*100
    print("Percentage of damaged rice kernels = %.2f"%b_p, "%")
    
    # removing paddings ---------
    img_with_fine_kernels = img_with_fine_kernels[1:-1,1:-1] 
    
    #cv2.imshow("Img_with_labels", img_with_labels)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #cv2.imshow("Img_with_fine_kernels", img_with_fine_kernels)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
  
  
    # Image conversion and savings ------------
    img_ = np.uint8(179*img_with_fine_kernels/np.max(img_with_fine_kernels))
    ch_temp = 255*np.ones_like(img_)
    final_img = cv2.merge([img_, ch_temp, ch_temp])  
    final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2BGR)
    final_img[img_ == 0] = 255
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    
   
    
    # change to binary 
    final_img[final_img <= 240] = 0
    final_img[final_img > 240] = 255
    
       
    b_p =round(b_p, 2)
    f5 = plt.figure()
    plt.imshow(final_img, cmap = "gray")
    plt.title(f'Damaged kernels = {b_p}%')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    f5.savefig(path+"/"+file_name+"_Task3.png")
    
#task3(connected_comp_label)
#%%
    
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered temp_dict whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()


file_name = args.input_filename
path = args.OP_folder

if not os.path.exists(path):
    print("Specified folder does not exist, images will be saved in current directory!!")
    path = os.getcwd()
    
# Function calls ----------
# TASK1
img_task1 = task1(file_name)

# TASK2
connected_comp_label = task2(img_task1)

# TASK 3
task3(connected_comp_label, min_area= int(args.min_area))





