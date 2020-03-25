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
from utils import *
from load_data import*

#main script for running SLAM

#set dataset
j0 = get_joint("joint/train_joint4")
l0 = get_lidar("lidar/train_lidar4")

#set parameters
N = 100
particle = np.zeros((N,3))
part_w = (1/N)*np.ones((N,1))
timesteps = len(l0)
threshold = 30

MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -40  #meters
MAP['ymin']  = -40
MAP['xmax']  =  40
MAP['ymax']  =  40 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']), dtype = np.int8) #DATA TYPE: char or int8
MAP['log'] = np.zeros((MAP['sizex'],MAP['sizey']))
MAP['display'] = 0.4*np.ones((MAP['sizex'],MAP['sizey'],3),dtype=np.int8)

angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T

ts = j0['ts']
head_angles = j0['head_angles']
start_head_idx = np.argmin(np.abs(ts - l0[0]['t']))

head_cur_ = head_angles[:,start_head_idx]
obstacles_map, phy_map = lidar2world(particle,head_cur_,l0[0]['scan'],angles,MAP)
MAP= update_log_odds(particle[0],obstacles_map[0],MAP)


x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

#set kernel for correlation
x_range = np.arange(-0.1,0.1+0.05,0.05)
y_range = np.arange(-0.1,0.1+0.05,0.05)

pathx = []
pathy = []

for t in range(1,timesteps):
    print("timestep:", t)
    
    ut = l0[t]['delta_pose']

    noise =  np.tile(np.random.normal(0, 1e-3, (N, 1)),(1,3))
    noise[:,2] = noise[:,2]*5

    particle = particle + ut + noise
    
    next_scan = l0[t]['scan']
    head_idx = np.argmin(np.abs(ts - l0[t]['t']))
    
    head_cur_ = head_angles[:,head_idx]
    updated_obstacles_map, updated_phy_units = lidar2world(particle,head_cur_,next_scan,angles, MAP)
    corr = np.zeros((N, 1))
    
    for i in range(N):
        c = mapCorrelation(MAP['map'],x_im,y_im,updated_phy_units[i],x_range,y_range)
        ind = np.unravel_index(np.argmax(c, axis=None), c.shape)

        corr[i] = c[ind[0],ind[1]]
        particle[i, 0] += x_range[ind[0]]
        particle[i, 1] += y_range[ind[1]]
        
    log_part_w = np.log(part_w) + corr
    max_num = np.max(log_part_w)
    log_part_w = log_part_w -max_num
    part_w = np.exp(log_part_w)
    part_w = part_w/np.sum(part_w)

    best_particle_idx = np.argmax(part_w)
    
    x_best = np.ceil((particle[best_particle_idx,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_best = np.ceil((particle[best_particle_idx,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    pathx.append(x_best)
    pathy.append(y_best)
    MAP = update_log_odds(particle[best_particle_idx],updated_obstacles_map[best_particle_idx],MAP)
    
    N_eff = 1 / np.sum(np.square(part_w))

    if(N_eff < threshold):
        particle = particle[np.random.choice(np.arange(N),N, p = part_w.reshape(-1))]
        
        part_w  = (1/N)*np.ones((N,1)) 
    if(t%2000==0):
        plt.figure(figsize=(9,12))
        plt.imshow(MAP['display'])
        plt.scatter(pathy,pathx,s = 1, c = 'r')
        plt.savefig("./lid4/lid4_{}.png".format(t))

    
plt.figure(figsize=(9,12))
plt.imshow(MAP['display'])
plt.scatter(pathy,pathx,s = 1, c = 'r')
plt.savefig("./lid4/lid4_final.png")




