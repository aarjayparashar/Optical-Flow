import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """
    """
    STUDENT CODE BEGINS
    """
    x = np.arange(up[1], down[1])
    y = np.arange(up[0], down[0])
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()


    flow_x = flow_x[up[0]:down[0], up[1]:down[1]].ravel()
    flow_y = flow_y[up[0]:down[0], up[1]:down[1]].ravel()

    calib_pic = np.linalg.inv(K) @ np.vstack((xx, yy, np.ones(xx.shape))) #3xn
    calib_flow = np.linalg.inv(K) @ np.vstack((flow_x, flow_y, np.zeros(flow_x.shape))) #3xn

    A = np.zeros((2, 8))
    B = np.zeros((2, 1))

    for i in range(calib_pic.shape[1]):
        a = np.array([[(calib_pic[0][i])**2, calib_pic[0][i]*calib_pic[1][i], calib_pic[0][i], calib_pic[1][i], 1, 0, 0, 0],
                 [calib_pic[0][i]*calib_pic[1][i], (calib_pic[1][i])**2, 0, 0, 0, calib_pic[1][i], calib_pic[0][i], 1]])
        A = np.vstack((A, a))

        b = np.array([[calib_flow[0][i]],
                     [calib_flow[1][i]]])
        B = np.vstack((B, b))

    A = A[2:, :]
    B = B[2:, :]

    x = np.linalg.lstsq(A, B)[0]
    sol = x.reshape(-1)

   
    return sol
    """
    STUDENT CODE ENDS
    """
    
    
