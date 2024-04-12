import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    """
    STUDENT CODE BEGINS
    """
    x = np.arange(x - 2, x + 3)
    x_p = x[x >= 0]
    x_p = x_p[x_p < Ix.shape[0]]

    y = np.arange(y - 2, y + 3)
    y_p = y[y >= 0]
    y_p = y_p[y_p < Iy.shape[0]]
    
    stack = np.column_stack((np.repeat(x_p, y_p.size), np.tile(y_p, x_p.size)))

    Ix = Ix[stack[:, 1], stack[:, 0]].reshape(stack.shape[0], 1)
    Iy = Iy[stack[:, 1], stack[:, 0]].reshape(stack.shape[0], 1)
    It = It[stack[:, 1], stack[:, 0]].reshape(stack.shape[0], 1)

    A = np.hstack((Ix, Iy))
    x = np.linalg.lstsq(A, -It, rcond = -1)

    flow = x[0].reshape(2,)
    conf = min(x[3])
    """
    STUDENT CODE ENDS
    """
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

