import numpy as np

#bbox = [x, y, w, h]
def get_IOU(bbox1, bbox2): 
    area1 = bbox1[2] * bbox1[3] 
    area2 = bbox2[2] * bbox2[3] 
    x1 = max(bbox1[0], bbox2[0]) 
    y1 = max(bbox1[1], bbox2[1]) 
    x2 = min(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) 
    y2 = min(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1)

    overlap_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
    ratio = overlap_area / (area1 + area2 - overlap_area)
    return ratio, overlap_area, area1, area2


def get_overlap(bbox1, bbox2): 
    num1 = bbox1.shape[0] 
    num2 = bbox2.shape[0] 
    overlap_mat = np.zeros((num1, num2)) 
    overlap_area = np.zeros((num1, num2)) 
    area1 = np.zeros(num1)
    area2 = np.zeros(num2)
    for n in range(num1): 
        for m in range(num2):
            overlap_mat[n,m], overlap_area[n,m], area1[n], area2[m] = get_IOU(bbox1[n,:], bbox2[m,:])

    return overlap_mat, overlap_area, area1, area2
