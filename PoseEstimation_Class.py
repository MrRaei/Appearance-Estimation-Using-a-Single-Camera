import cv2
import numpy as np
import scipy.linalg as la

cornerDots = np.array([[[ 350.0 , 607.0 ]] , [[ 412.0 , 609.0 ]] , [[ 472.0 , 609.0 ]] , [[ 370.0 , 570.0 ]] , [[ 429.0 , 571.0 ]] , 
    [[ 486.0 , 573.0 ]] , [[ 389.0 , 537.0 ]] , [[ 443.0 , 538.0 ]] , [[ 498.0 , 539.0 ]]])

def PoseEstimationFirstValues():
    # Load previously saved data
    with np.load('Camera Calibration/Calibre Files/HTC/HTC KMatrix.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    TileSize = 20 #(cm)
    m = 3 # left to right (number of tiles)
    n = 3 # up to down (number of tiles)

    objp = np.zeros((m*n,3), np.float32)
    objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)*TileSize

    # Choose m*n corners
    corners = cornerDots

    # Find the rotation and translation vectors.
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

    return mtx, rvecs, tvecs, dist, objp

def PoseEstimation(H, KInv, RTOrginal_4_4, r1, r2, r3, tvecs, topY, bottomY, topBottomX):
    ## Formula:
    ## Finding X & Y:
    ## landa * [[xBottom],[yBottom],[1]] = K * [R T] * [[X],[Y],[0],[1]]
    ## landa * [[xBottom],[yBottom],[1]] = K * [r1 r2 T] * [[X],[Y],[1]]
    ## landa * [[xBottom],[yBottom],[1]] = H * [[X],[Y],[1]]
    ## landa * H^(-1) * [[xBottom],[yBottom],[1]] = [[X],[Y],[1]]
    ## landa * HBot = [[X],[Y],[1]]
    ## X = landa * HBot[0]
    ## Y = landa * HBot[1]
    ## 1 = landa * HBot[2]

    ## Finding Z (Height):
    ## landa * [[xTop],[yTop],[1]] = K * [R T] * [[X],[Y],[Z],[1]]
    ## landa * K^(-1) * [[xTop],[yTop],[1]] = [R T] * [[X],[Y],[Z],[1]]
    ## landa * K^(-1) * [[xTop],[yTop],[1]] = X*r1 + Y*r2 + Z*r3 + T
    ## landa * V = X*r1 + Y*r2 + Z*r3 + T
    ## [V | -r3] * [[landa],[Z]] = X*r1 + Y*r2 + T
    ## A * [[landa],[Z]] = B
    ## A * [[landa],[Z]] = B
    ## Z = linalg.lstsq(A,B)[0][1]

    Top = np.array([[topBottomX], [topY], [1]])
    Bottom = np.array([[topBottomX], [bottomY], [1]])
    
    # H*Bottom
    HBot = np.dot(H,Bottom)
    # landa, X, Y
    landa = 1/HBot[2]
    X = HBot[0]*landa
    Y = HBot[1]*landa

    distansFromCamera = np.dot(RTOrginal_4_4, np.array([[X[0]], [Y[0]], [0], [1]]))[2]
    # print (distansFromCamera)

    # Z
    V = np.dot(KInv,Top);
    A = np.concatenate((V,-r3),axis=1)

    Xr1 = X*r1
    Yr2 = Y*r2
    B = Xr1 + Yr2 + tvecs
    
    Z = la.lstsq(A,B)

    Z = Z[0][1]
    
    # print (Z)
    if Z < 0:
        return 0, 0
    return int(Z), (distansFromCamera)[0]


# # Render a Cube
def newCorners(objp, mtx, dist):
    # Choose m*n corners
    newCoordinates = cornerDots

    # Find the rotation and translation vectors.
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, newCoordinates, mtx, dist)
    return rvecs, tvecs


def Cube(frame, Height, mtx, dist, objp):
    #####
    GreenLine = 40
    BlueLine = -Height
    RedLine = 40
    axis = np.float32([[0,0,0], [0,GreenLine,0], [GreenLine,GreenLine,0], [GreenLine,0,0],
        [0,0,-BlueLine],[0,RedLine,-BlueLine],[RedLine,RedLine,-BlueLine],[RedLine,0,-BlueLine]])

    rvecs, tvecs = newCorners(objp, mtx, dist)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    #####

    imgpts = np.int32(imgpts).reshape(-1,2)   

    # draw ground floor in green
    cv2.drawContours(frame, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    cv2.drawContours(frame, [imgpts[4:]],-1,(0,0,255),-1)

    return frame