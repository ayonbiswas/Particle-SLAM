import numpy as np
from scipy import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numpy.linalg import inv
import cv2
from skimage.draw import line

#compute map correlation 
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                                            np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

#convert from lidar frame to world frame
def sensor_transform(current_particle,head_angles):
    
    h_T_l = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.15],
                    [0, 0, 0, 1]])
    
    p, y = head_angles[1], head_angles[0]    
    
    r11 = np.cos(y) * np.cos(p)
    r12 = - np.sin(y) 
    r13 = np.cos(y) * np.sin(p)  

    r21 = np.sin(y) * np.cos(p)
    r22 = np.cos(y) 
    r23 = np.sin(y) * np.sin(p)

    r31 = -np.sin(p)
    r32 =  0
    r33 = np.cos(p)
    
    b_T_h = np.array([[r11, r12, r13, 0.],
                    [r21, r22, r23, 0.],
                    [r31, r32, r33, 0.33],
                    [0., 0., 0., 1.]])
    x , y , alpha = current_particle
    
    w_T_b = np.array([[np.cos(alpha), -np.sin(alpha),0, x],
                    [np.sin(alpha), np.cos(alpha),0, y],
                    [0, 0, 1, 0.93],
                    [0, 0, 0, 1]])
    
    w_T_l =  w_T_b@ b_T_h@ h_T_l

    return w_T_l

#create an occupied map for each particle
def lidar2world(particles, head_angle, lidar_scan, angles, MAP):
    

    ranges = np.double(lidar_scan).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    world_map_ = {}
    phy_units_ = {}
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    Y = np.concatenate([np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0), np.ones(xs0.shape)], axis =0)
    for i in range(len(particles)):
        w_T_l = sensor_transform(particles[i],head_angle)
        
        lidar_pts_w = w_T_l@Y
        lidar_pts_w = lidar_pts_w[:,lidar_pts_w[2,:] > 0.1]
        xis = np.ceil((lidar_pts_w[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((lidar_pts_w[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        world_map_[i] = np.unique(np.stack((xis,yis),axis = 1),axis = 0).T
        phy_units_[i] = lidar_pts_w
    return world_map_, phy_units_

#compute free space and update log-odds
def update_log_odds(particle, occupied, Map):
    particle_pos_x = (np.ceil((particle[0] - Map['xmin']) / Map['res']).astype(np.int16) - 1)
    particle_pos_y = (np.ceil((particle[1] - Map['ymin']) / Map['res']).astype(np.int16) - 1)
    Map["log"][occupied[0,:], occupied[1,:]] += np.log(8)
    Map['log'][particle_pos_x,particle_pos_y] += -np.log(4)

    for i in range(occupied.shape[1]):
        start =  (particle_pos_x, particle_pos_y)
        end = (occupied[0][i], occupied[1][i])
        free = line(*start, *end)
        Map["log"][free[0][1:-1],free[1][1:-1]] += -np.log(4)
        
    obstacles = Map['log'] > 0
    freespace = Map['log'] < 0
    Map['map'][obstacles] = 1
    Map['display'][obstacles, :] = 0
    Map['display'][freespace] = 1
    
    return Map 