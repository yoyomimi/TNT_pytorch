import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats


#TODO what does this function mean?
def pred_bbox_by_F(bbox, F, show_flag, img1, img2):
    # Create figure and axes
    if show_flag==1:
        fig1, ax1 = plt.subplots(1)
    # Display the image
    if show_flag==1:
        ax1.imshow(img1)
    
    pred_bbox = np.zeros((len(bbox),4))
    for n in range(len(bbox)):
        xmin = bbox[n,0]
        ymin = bbox[n,1]
        xmax = bbox[n,2]+bbox[n,0]
        ymax = bbox[n,3]+bbox[n,1]
        w = bbox[n,2]
        h = bbox[n,3]
        if show_flag==1:
            rect = patches.Rectangle((xmin,ymin), w, h, linewidth=1, edgecolor='#FF0000', facecolor='none')
            ax1.add_patch(rect)

    if show_flag==1:
        plt.show()
        
    # Display the image
    if show_flag==1:
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(img2)
        
    for n in range(len(bbox)):
        xmin = bbox[n,0]
        ymin = bbox[n,1]
        xmax = bbox[n,2] + bbox[n,0]
        ymax = bbox[n,3] + bbox[n,1]
        w = bbox[n,2]
        h = bbox[n,3]
        
        temp_A = np.zeros((4,2))
        temp_b = np.zeros((4,1))
        temp_pt = np.zeros((1,3))
        temp_pt[0,:] = np.array([xmin,ymin,1])
        A1 = np.matmul(temp_pt, np.transpose(F))

        temp_A[0,0] = A1[0,0]
        temp_A[0,1] = A1[0,1]
        temp_b[0,0] = -A1[0,2]
        
        temp_pt[0,:] = np.array([xmax,ymin,1])
        A2 = np.matmul(temp_pt, np.transpose(F))
        temp_A[1,0] = A2[0,0]
        temp_A[1,1] = A2[0,1]
        temp_b[1,0] = -w * A2[0,0] - A2[0,2]
        
        temp_pt[0,:] = np.array([xmin,ymax,1])
        A3 = np.matmul(temp_pt, np.transpose(F))
        temp_A[2,0] = A3[0,0]
        temp_A[2,1] = A3[0,1]
        temp_b[2,0] = -h * A3[0,1] - A3[0,2]
        
        temp_pt[0,:] = np.array([xmax,ymax,1])
        A4 = np.matmul(temp_pt, np.transpose(F))
        temp_A[3,0] = A4[0,0]
        temp_A[3,1] = A4[0,1]
        temp_b[3,0] = -w * A4[0,0] - h * A4[0,1] - A4[0,2]
        
        new_loc = np.matmul(np.linalg.pinv(temp_A),temp_b)
        xmin = new_loc[0,0]
        ymin = new_loc[1,0]
        xmax = new_loc[0,0] + w
        ymax = new_loc[1,0] + h
        
        pred_bbox[n,0] = xmin
        pred_bbox[n,1] = ymin
        pred_bbox[n,2] = w
        pred_bbox[n,3] = h

        if show_flag==1:
            rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#FF0000', facecolor='none')
            ax2.add_patch(rect)
    
    if show_flag==1:
        plt.show()

    return pred_bbox


def linear_pred(x):
    if len(x)==1:
        return x
    else:
        y = np.array(range(len(x)))
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return slope * len(y) + intercept