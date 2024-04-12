import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
        @ep: np.array(3,) the epipole you found epipole.py note it is uncalibrated and you need to calibrate it in this function!
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)

    """
    STUDENT CODE BEGINS
    """
    x = np.arange(flow.shape[1])
    y = np.arange(flow.shape[0])
    xx, yy = np.meshgrid(x, y)

    cond = confidence > thres

    xx_thresh = xx[cond]
    yy_thresh = yy[cond]
    
    u_thres = flow[:, :, 0][cond]
    v_thres = flow[:, :, 1][cond]


    calib_pic = np.linalg.inv(K) @ np.vstack((xx_thresh, yy_thresh, np.ones(xx_thresh.shape)))
    calib_flow = np.linalg.inv(K) @ np.vstack((u_thres, v_thres, np.zeros(u_thres.shape)))
    calib_ep = np.linalg.inv(K) @ ep.reshape(3,1)

    depth = np.linalg.norm(calib_pic - calib_ep, axis=0) / np.linalg.norm(calib_flow, axis=0) #shape is (n,)

    depth_map[yy_thresh, xx_thresh] = depth


    """
    STUDENT CODE ENDS
    """

    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    # print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    

    return truncated_depth_map
