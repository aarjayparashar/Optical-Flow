import numpy as np
import pdb
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold
        (for both sampling and finding inliers)
        u, v and smin are (w,h), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,)
    '''

    """YOUR CODE HERE -- You can do the thresholding outside the RANSAC loop here
    """
    x = np.arange(u.shape[1])
    y = np.arange(u.shape[0])
    
    xx, yy = np.meshgrid(x, y)
    
    xx = xx - int(u.shape[0]/2)
    yy = yy - int(u.shape[1]/2)

    res = np.arange(u.shape[0])
    res, _ = np.meshgrid(res, res)
    

    smin = smin.ravel()

    cond = smin>thresh
    inds = cond

    res_thresh = np.arange(len(smin))[inds]
    
    u_thres = u.ravel()[inds]
    v_thres = v.ravel()[inds]
    xx_thres = xx.ravel()[inds]
    yy_thres = yy.ravel()[inds]

    """ END YOUR CODE
    """
    sample_size = 2
    eps = 10**-2
    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations):
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        
        """YOUR CODE HERE
        """
        
        x_p = np.vstack((xx_thres[sample_indices], yy_thres[sample_indices], np.ones(xx_thres[sample_indices].shape)))
        up = np.vstack((u_thres[sample_indices], v_thres[sample_indices], np.zeros(u_thres[sample_indices].shape)))
        A = np.cross(x_p, up, axisa=0, axisb=0)
        
        u_svd, s_svd, vt_svd = np.linalg.svd(A)
        ep = vt_svd.T[:, -1].reshape(3, 1)

        x_p_test = np.vstack((xx_thres[test_indices], yy_thres[test_indices], np.ones(xx_thres[test_indices].shape)))
        
        up_test = np.vstack((u_thres[test_indices], v_thres[test_indices], np.zeros(u_thres[test_indices].shape)))
        
        mul = np.cross(x_p_test, up_test, axisa=0, axisb=0) 
        
        
        delta = np.abs(mul @ ep)
        
        cond = (delta < eps).reshape(-1,)
        inliers = res_thresh[np.hstack((sample_indices, test_indices[cond]))]


       

        """ END YOUR CODE
        """

        #NOTE: inliers need to be inds in original input (unthresholded), 
        #sample inds before test inds for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep.reshape(-1), best_inliers