import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """

    """
    STUDENT CODE BEGINS
    """
    y = np.linspace(0, image.shape[0], image.shape[0])
    x = np.linspace(0, image.shape[1], image.shape[1])
    

    ind_y = np.where(confidence > threshmin)[0]
    ind_x = np.where(confidence > threshmin)[1]

    flow_y = flow_image[ind_y, ind_x, 1]
    flow_x = flow_image[ind_y, ind_x, 0]
    
    
    
    """
    STUDENT CODE ENDS
    """
    
    plt.imshow(image, cmap='gray')
    plt.quiver(ind_x, ind_y, (-flow_x*10).astype(int), (-flow_y*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    

