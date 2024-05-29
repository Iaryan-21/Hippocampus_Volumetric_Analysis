import numpy as np

def Dice3d(x,y):

    if len(x.shape) != 3 or len(y.shape) !=3:
        raise Exception(f"Need 3D input instead got {x.shape} and {y.shape}")
    if x.shape != y.shape:
        raise Exception(f"Expecting inputs of the same shape, got {x.shape} and {y.shape}")

    x = (x>0)
    y = (y>0)
    intersection = np.sum(x*y)
    volumes = np.sum(x) + np.sum(y)
    if volumes == 0:
        return -1
    
    return 2.*float(intersection)/float(volumes)

def Jaccard3d(x,y):
    
    if len(x.shape) !=3 or len(y.shape) !=3:
        raise Exception(f"Need 3D input instead got {x.shape} and {y.shape}")
    if x.shape != y.shape:
        raise Exception(f"Expecting inputs of the same shape, got {x.shape} and {y.shape}")

    x = (x>0)
    y = (y>0)
    intersection = np.sum(x*y)
    union = np.sum(x) + np.sum(y) - intersection
    if union == 0:
        return -1
    
    return float(intersection)/ float(union)