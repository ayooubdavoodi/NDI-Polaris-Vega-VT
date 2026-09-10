# transformation_fun.py

import numpy as np
import os
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import csv
import glob
from   datetime import datetime
import json
import copy
from scipy.interpolate import interp1d
from matplotlib import gridspec
import json
import  math
from scipy import optimize
from scipy.spatial.transform import Rotation as Rott


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def tranMat_to_pose(T):

    x = T[0][3]
    y = T[1][3]
    z = T[2][3]

    R = T[0:3,0:3]
    euler= rotationMatrixToEulerAngles(R)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return np.array([x, y, z, roll , pitch, yaw])

def euler_to_rotVec(yaw, pitch, roll):
    # compute the rotation matrix
    Rmat = euler_to_rotMat(yaw, pitch, roll)
    
    theta = math.acos(((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        rx, ry, rz = 0.0, 0.0, 0.0
    else:
        multi = 1 / (2 * math.sin(theta))
        rx = multi * (Rmat[2, 1] - Rmat[1, 2]) * theta
        ry = multi * (Rmat[0, 2] - Rmat[2, 0]) * theta
        rz = multi * (Rmat[1, 0] - Rmat[0, 1]) * theta
    return rx, ry, rz

def euler_to_rotMat(yaw, pitch, roll):
    # Rz_yaw = np.array([
    #     [np.cos(yaw), -np.sin(yaw), 0],
    #     [np.sin(yaw),  np.cos(yaw), 0],
    #     [          0,            0, 1]])
    # Ry_pitch = np.array([
    #     [ np.cos(pitch), 0, np.sin(pitch)],
    #     [             0, 1,             0],
    #     [-np.sin(pitch), 0, np.cos(pitch)]])
    # Rx_roll = np.array([
    #     [1,            0,             0],
    #     [0, np.cos(roll), -np.sin(roll)],
    #     [0, np.sin(roll),  np.cos(roll)]])
    # # R = RzRyRx
    # rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))

    # %     The rotation matrix R can be constructed as follows by
    # %     ct = [cx cy cz] and st = [sx sy sz]
    # %
    # %     R = [            cy*cz,           -cy*sz,     sy]
    # %         [ cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx]
    # %         [ sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy]
    # %       = Rx(tx) * Ry(ty) * Rz(tz)
    ct1 = np.cos(roll)
    st1 = np.sin(roll)
    ct2 = np.cos(pitch)
    st2 = np.sin(pitch)
    ct3 = np.cos(yaw)
    st3 = np.sin(yaw)

    R11 = ct2*ct3;
    R12 = -ct2*st3;
    R13 = st2;
    R21 = ct1*st3 + ct3*st1*st2;
    R22 = ct1*ct3 - st1*st2*st3;
    R23 = -ct2*st1;
    R31 = st1*st3 - ct1*ct3*st2;
    R32 = ct3*st1 + ct1*st2*st3;
    R33 = ct1*ct2;

    rotMat = np.array([
        [R11, R12, R13],
        [R21 ,R22, R23],
        [R31, R32, R33]])

    # R = RzRyRx
    return rotMat


def pose_to_tranMat(x , y ,z , yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0 , 0],
        [np.sin(yaw),  np.cos(yaw), 0 , 0],
        [          0,            0, 1 , 0],
        [          0,            0, 0 , 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch) , 0],
        [             0, 1,             0 , 0],
        [-np.sin(pitch), 0, np.cos(pitch) , 0],
        [          0,            0,     0 , 1]])
    Rx_roll = np.array([
        [1,            0,             0 , 0],
        [0, np.cos(roll), -np.sin(roll) , 0],
        [0, np.sin(roll),  np.cos(roll) , 0],
        [          0,            0,   0 , 1]])
    Trans_xyz = np.array([
        [1,            0,             0 , x],
        [0,            1,             0 , y],
        [0,            0,             1 , z],
        [0,            0,             0 , 1]])
    # R = RzRyRx
    TranMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    # TranMat[0][3] =x
    # TranMat[1][3] =y
    # TranMat[2][3] =z
    TranMat = np.dot(Trans_xyz,TranMat)
    return TranMat

def RPY_to_normal_vector(fn):
    f = open(fn,"r")
    data = json.load(f)

    x = []
    y = []
    z = []
    nx = []
    ny = []
    nz = []

    for i in data['probe_frames']:
        x.append(i['x'])
        y.append(i['y'])
        z.append(i['z'])

        Rot =  euler_to_rotMat(i['Y'],i['P'],i['R'])
        normal_vector = -1 * Rot[:,2]
        
        nx.append(normal_vector[0])
        ny.append(normal_vector[1])
        nz.append(normal_vector[2])
    f.close()
    return  x,y,z,nx,ny,nz


def RPY_to_normal_vector2(fn):
    f = open(fn,"r")
    data = json.load(f)

    x = []
    y = []
    z = []
    nx = []
    ny = []
    nz = []

    for i in data['probe_frames']:
        x.append(i['x'])
        y.append(i['y'])
        z.append(i['z'])

        Rot =  euler_to_rotMat(i['Y'] , i['P'], i['R'])
        
        normal_vector = -1 * Rot[:,2]
        
        nx.append(normal_vector[0])
        ny.append(normal_vector[1])
        nz.append(normal_vector[2])
    f.close()
    return  x,y,z,nx,ny,nz



# def plot_multiple_frames(T_P1,T_P2,T_P3,T_P4,vec_length):

#     fig_1 = plt.figure(1)
#     ax3d_1 = fig_1.add_subplot(projection='3d')


#     ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][0], T_P1[1][0], T_P1[2][0], length=vec_length, normalize=True,color='red')
#     ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][1], T_P1[1][1], T_P1[2][1], length=vec_length, normalize=True,color='green')
#     ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][2], T_P1[1][2], T_P1[2][2], length=vec_length, normalize=True,color='blue')



#     ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][0], T_P2[1][0], T_P2[2][0], length=vec_length, normalize=True,color='red')
#     ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][1], T_P2[1][1], T_P2[2][1], length=vec_length, normalize=True,color='green')
#     ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][2], T_P2[1][2], T_P2[2][2], length=vec_length, normalize=True,color='blue')



#     ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][0], T_P3[1][0], T_P3[2][0], length=vec_length, normalize=True,color='red')
#     ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][1], T_P3[1][1], T_P3[2][1], length=vec_length, normalize=True,color='green')
#     ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][2], T_P3[1][2], T_P3[2][2], length=vec_length, normalize=True,color='blue')



#     ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][0], T_P4[1][0], T_P4[2][0], length=vec_length, normalize=True,color='red')
#     ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][1], T_P4[1][1], T_P4[2][1], length=vec_length, normalize=True,color='green')
#     ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][2], T_P4[1][2], T_P4[2][2], length=vec_length, normalize=True,color='blue')



#     ax3d_1.set_xlabel('x [m]')
#     ax3d_1.set_ylabel('y [m]')
#     ax3d_1.set_zlabel('z [m]')


#     plt.axis('equal')

#     plt.show()


def plot_multiple_frames(frames, vec_length = 10):

    colors = ['red', 'green', 'blue']

    fig_1 = plt.figure(1)
    ax3d_1 = fig_1.add_subplot(projection='3d')

    for T_P in frames:
        for i, color in enumerate(colors):
            ax3d_1.quiver(
                T_P[0][3], T_P[1][3], T_P[2][3],
                T_P[0][i], T_P[1][i], T_P[2][i],
                length=vec_length, normalize=True, color=color
            )

    ax3d_1.set_xlabel('x [m]')
    ax3d_1.set_ylabel('y [m]')
    ax3d_1.set_zlabel('z [m]')
    
    plt.axis('equal')
    plt.show()


# p1 = [0.0, 0.0, 0.06, 0.0, 0.0, 0.0]
# p2 = [0.06, 0.0, 0.0, 0.0, 0.0, 0.0]
# p3 = [0.0, 0.0, 0.0, 0.0, -1*math.pi/3, 0.0]
# p4 = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0]
# p5 = [0.0, 0.0, 0.21, 0.0, 0.0, 0.0]

# T1 = pose_to_tranMat(p1[0],p1[1],p1[2],p1[5],p1[4],p1[3]);
# T2 = pose_to_tranMat(p2[0],p2[1],p2[2],p2[5],p2[4],p2[3]);
# T3 = pose_to_tranMat(p3[0],p3[1],p3[2],p3[5],p3[4],p3[3]);
# T4 = pose_to_tranMat(p4[0],p4[1],p4[2],p4[5],p4[4],p4[3]);
# T5 = pose_to_tranMat(p5[0],p5[1],p5[2],p5[5],p5[4],p5[3]);

# T21 = np.dot(T1 ,T2);
# T31 = np.dot(T21,T3);
# T41 = np.dot(T31,T4);
# T51 = np.dot(T41,T5);


# pose_tip = tranMat_to_pose(T51);
# print(pose_tip)

# T51_inv = np.linalg.inv(T51)
# pose_tip_inv = tranMat_to_pose(T51_inv);
# print(pose_tip_inv)



# p3 = [-0.09325, 0.0, 0.17844, 0.0, -1*math.pi/3, 0.0]
# print(p3)
# T3 = pose_to_tranMat(p3[0],p3[1],p3[2],p3[5],p3[4],p3[3]);
# T3_inv = np.linalg.inv(T3)
# p3_inv = tranMat_to_pose(T3_inv);
# print(p3_inv)

# ee_pose =[-0.6354139,  0.20465067,  0.44357022,  -2.53944458,  -0.79878963,  -2.22477228]
# tip_pose=[-0.60890432,  -0.03403331,  0.27484129,  -2.75262867,  -0.22631686,  -0.80139669]


# sx = 0.0946673
# sy = 0.09228649
# image_sensetivity =  np.array([
#             [ sx,    0.0,   0.0,   0.0],
#             [0.0,    sy,   0.0,   0.0],
#             [0.0,   0.0,   1.0,   0.0],
#             [0.0,   0.0,   0.0,   1.0]])

# probe_image_to_ee = np.array([
#             [-1.94602703e-01,   -1.06590609e-01,   9.95073449e-01,   1.52880127e-01],
#             [9.980098276e-01,   1.86038343e-02,   1.97639232e-01,   -4.0544410e+01],
#             [-3.92065910e-02,   9.94128935e-01,   1.00848916e-01,   2.64340645e+02],
#             [0.00000000e+00,    0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

# T_ee_w = pose_to_tranMat(ee_pose[0]*1000,ee_pose[1]*1000,ee_pose[2]*1000,ee_pose[5],ee_pose[4],ee_pose[3])
# T_tip_w = pose_to_tranMat(tip_pose[0]*1000,tip_pose[1]*1000,tip_pose[2]*1000,tip_pose[5],tip_pose[4],tip_pose[3])

# T_I_w = np.dot(T_ee_w, probe_image_to_ee)
# T_I_w_inv = np.linalg.inv(T_I_w)

# T_tip_I = np.dot(T_I_w_inv, T_tip_w)

# print(T_tip_I)
# w = -1*T_tip_I[2][3] /T_tip_I[2][2]
# P_I = np.array([
#     [T_tip_I[0][3] + w* T_tip_I[0][2]],
#     [T_tip_I[1][3] + w* T_tip_I[1][2]],
#     [0],
#     [1]])
# print(P_I[0]/image_sensetivity[0][0])
# print(P_I[1]/image_sensetivity[1][1])







# R = euler_to_rotMat(0.4, 0.5, 0.3)
# P = [[0.7], [0.8], [0.9]]
# T= np.append(R, P, axis=1)
# T = np.append(T, [np.array([0,0,0,1])], axis=0)
# print(T)


# T_tip  =  np.array([[  1.,    0.,    0.,    0.  ],[  0.,    1.,    0.,   -74.71],[  0.,    0.,    1.,   247.69],[  0.,    0.,    0.,    1.  ]])
# T_pointer  =  np.array([[  1.,        0.,        0.,        2.002234],[  0.,        1.,        0.,        2.4414  ],[  0.,        0.,        1.,      166.0548  ],[  0.,        0.,        0.,        1.     ]])

# T_pointer_c  =  np.array([[ 9.13466394e-01,   4.02260572e-01,  -6.13645948e-02,  -2.15519135e+02], 
#  [ 9.17408466e-02,  -5.66682220e-02,   9.94169176e-01,   9.98162994e+01], 
#  [ 3.96437645e-01,  -9.13769722e-01,  -8.86682421e-02,   1.98686926e+03], 
#  [ 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])


# T_drill_c  =  np.array([[ 7.87189364e-01,  3.76907974e-01,  4.88132417e-01, -3.12211243e+02],
#  [-4.78571028e-01, -1.25892192e-01,  8.68976951e-01,  7.43617783e+01],
#  [ 3.88976395e-01, -9.17655468e-01,  8.12762454e-02,  1.91426245e+03],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# T_pointer  =  np.array([[ 9.13466394e-01,  4.02260572e-01, -6.13645948e-02, -2.22897968e+02],
#  [ 9.17408466e-02, -5.66682220e-02,  9.94169176e-01,  2.64948200e+02],
#  [ 3.96437645e-01, -9.13769722e-01, -8.86682421e-02,  1.97070836e+03],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# T_tip  =  np.array([[ 7.87189364e-01,  3.76907974e-01,  4.88132417e-01, -2.19464519e+02],
#  [-4.78571028e-01, -1.25892192e-01,  8.68976951e-01,  2.99004085e+02],
#  [ 3.88976395e-01, -9.17655468e-01,  8.12762454e-02,  2.00295180e+03],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# # pose  =  [ 1.61081296e+02 -2.02181017e+02  2.55261034e+01 -1.52766966e-01
# #   7.56896740e-02  5.67688245e-01]


# # T_1 = np.dot(T_pointer_c,T_pointer)
# # T_2 = np.dot(T_drill_c,T_tip)

# T_pointer_tip   = np.dot(T_tip , np.linalg.inv(T_pointer))
# pose = tranMat_to_pose(T_pointer_tip)
# print(pose)

# Pose_world = [0,0,0,0,0,0]
# T_world = pose_to_tranMat(Pose_world[0],Pose_world[1],Pose_world[2],Pose_world[5],Pose_world[4],Pose_world[3])
# fig_1 = plt.figure(1)
# ax3d_1 = fig_1.add_subplot(projection='3d')
# ax3d_1.quiver(T_world[0][3], T_world[1][3], T_world[2][3], T_world[0][0], T_world[1][0], T_world[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_world[0][3], T_world[1][3], T_world[2][3], T_world[0][1], T_world[1][1], T_world[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_world[0][3], T_world[1][3], T_world[2][3], T_world[0][2], T_world[1][2], T_world[2][2], length=10, normalize=True,color='blue')



# ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][0], T_P1[1][0], T_P1[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][1], T_P1[1][1], T_P1[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_P1[0][3], T_P1[1][3], T_P1[2][3], T_P1[0][2], T_P1[1][2], T_P1[2][2], length=10, normalize=True,color='blue')



# ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][0], T_P2[1][0], T_P2[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][1], T_P2[1][1], T_P2[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_P2[0][3], T_P2[1][3], T_P2[2][3], T_P2[0][2], T_P2[1][2], T_P2[2][2], length=10, normalize=True,color='blue')



# ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][0], T_P3[1][0], T_P3[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][1], T_P3[1][1], T_P3[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_P3[0][3], T_P3[1][3], T_P3[2][3], T_P3[0][2], T_P3[1][2], T_P3[2][2], length=10, normalize=True,color='blue')



# ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][0], T_P4[1][0], T_P4[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][1], T_P4[1][1], T_P4[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_P4[0][3], T_P4[1][3], T_P4[2][3], T_P4[0][2], T_P4[1][2], T_P4[2][2], length=10, normalize=True,color='blue')


# # ax3d_1.quiver(T_drill_c[0][3], T_drill_c[1][3], T_drill_c[2][3], T_drill_c[0][0], T_drill_c[1][0], T_drill_c[2][0], length=10, normalize=True,color='red')
# # ax3d_1.quiver(T_drill_c[0][3], T_drill_c[1][3], T_drill_c[2][3], T_drill_c[0][1], T_drill_c[1][1], T_drill_c[2][1], length=10, normalize=True,color='green')
# # ax3d_1.quiver(T_drill_c[0][3], T_drill_c[1][3], T_drill_c[2][3], T_drill_c[0][2], T_drill_c[1][2], T_drill_c[2][2], length=10, normalize=True,color='blue')


# ax3d_1.quiver(T_pointer[0][3], T_pointer[1][3], T_pointer[2][3], T_pointer[0][0], T_pointer[1][0], T_pointer[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_pointer[0][3], T_pointer[1][3], T_pointer[2][3], T_pointer[0][1], T_pointer[1][1], T_pointer[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_pointer[0][3], T_pointer[1][3], T_pointer[2][3], T_pointer[0][2], T_pointer[1][2], T_pointer[2][2], length=10, normalize=True,color='blue')


# ax3d_1.quiver(T_tip[0][3], T_tip[1][3], T_tip[2][3], T_tip[0][0], T_tip[1][0], T_tip[2][0], length=10, normalize=True,color='red')
# ax3d_1.quiver(T_tip[0][3], T_tip[1][3], T_tip[2][3], T_tip[0][1], T_tip[1][1], T_tip[2][1], length=10, normalize=True,color='green')
# ax3d_1.quiver(T_tip[0][3], T_tip[1][3], T_tip[2][3], T_tip[0][2], T_tip[1][2], T_tip[2][2], length=10, normalize=True,color='blue')


# ax3d_1.set_xlabel('x [m]')
# ax3d_1.set_ylabel('y [m]')
# ax3d_1.set_zlabel('z [m]')


# save_path = '/home/ayoob/fusion_track/data'
# file_name = "geometry001_lamb_compensated.ini"
# completeName = os.path.join(save_path, file_name)
# with open(completeName, 'w') as f:
#     f.write('[fiducial0]'+'\n')
#     f.write('x='+ str(P1[0]) +'\n')
#     f.write('y='+ str(P1[1]) +'\n')
#     f.write('z='+ str(P1[2]) +'\n')
#     f.write('[fiducial1]'+'\n')
#     f.write('x='+ str(P2[0]) +'\n')
#     f.write('y='+ str(P2[1]) +'\n')
#     f.write('z='+ str(P2[2]) +'\n')
#     f.write('[fiducial2]'+'\n')
#     f.write('x='+ str(P3[0]) +'\n')
#     f.write('y='+ str(P3[1]) +'\n')
#     f.write('z='+ str(P3[2]) +'\n')
#     f.write('[fiducial3]'+'\n')
#     f.write('x='+ str(P4[0]) +'\n')
#     f.write('y='+ str(P4[1]) +'\n')
#     f.write('z='+ str(P4[2]) +'\n')
#     f.write('[geometry]'+'\n')
#     f.write('count=4'+'\n')
#     f.write('id=1'+'\n')
#     f.close()




# # ax3d_1.legend()

# # plt.xlim([-0.8, 0.1])
# # plt.ylim([-0.3, 0.3])
# plt.axis('equal')
# # ax3d_1.set_aspect('equal', adjustable='box')
# fig_1.axis('square')

# # fig_1.set_aspect('equal')


# # fig_2= plt.figure(2)
# # ax3d_2 = fig_2.add_subplot(projection='3d')
# # ax3d_2.scatter(pose_x, pose_y,pose_z, color='green',label=" drill base points" )
# # ax3d_2.scatter(Tip_point[0], Tip_point[1],Tip_point[2], color='blue', label=" pivot _point" )
# # # ax3d_2.scatter(M_x, M_y,M_z, color='black',label=" Marker points" )
# # # ax3d_1.scatter(C_x2, C_y2,C_z2, color='blue', label=" Corner points" )
# # # ax3d_2.quiver(pose_x, pose_y,pose_z, n_x,n_y,n_z, length=0.2, normalize=True,color='blue')


# # ax3d_2.set_xlabel('x [m]')
# # ax3d_2.set_ylabel('y [m]')
# # ax3d_2.set_zlabel('z [m]')

# # plt.xlim([-0.8, 0.1])
# # plt.ylim([-0.3, 0.3])



# plt.show()



