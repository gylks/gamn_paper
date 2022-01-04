import os
import numpy as np
import json 
import math 
import random
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, LeakyReLU, ELU
from tensorflow.keras.regularizers import l1_l2, l1,l2
from tensorflow.keras.optimizers import Adam

import plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio
offline.init_notebook_mode(connected=True) 

CURRENT_MODEL_SAVE_DIR = './save/'
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']= '-1' #-1 for CPU, 0 for GPU
tf.__version__ , tf.keras.__version__, tf.config.list_physical_devices('GPU')

def flatten(sArray):
    '''
    take stroke array and create flat structure for network training
    [stroke[0],stroke[1],stroke[2],stroke[3]]
    
    (4, 52, 2) 4 strokes, time ,xy
    (52, 2, 4)
    (52, 8)
    
    '''
    x = np.transpose(sArray,[1,2,0])
    x = np.reshape(x,[-1,8])
    return x

def unFlatten(fArray):
    '''
    take stroke array and create flat structure for network training
    [stroke[0],stroke[1],stroke[2],stroke[3]]
    only using first four strokes
    
    (52, 8) <T,xy>
    (52, 2, 4)
    (4, 52, 2) 4 strokes, time ,xy
    
    '''
    x = np.reshape(fArray[:,:8], [-1,2,4]) #(52, 2, 4)
    x = np.transpose(x,[2,0,1]) #(4, 52, 2)
    return x


def graphCurves(xyCurves, layout=go.Layout(),labels=[], dashRange=range(0,0), mode='lines+markers'):
    #(n,points,3) where n  is the number of curves, points is the number of points on the circle, 3 is xyz
    traces = []
    for i in range (0, len(xyCurves)):
        xyCa = np.array(xyCurves[i])
        if i < len(labels):
            label = labels[i]
        else:
            label = str(i)
            
        #create a line/spline for each curve   
        sc = go.Scatter(
            x =  xyCa[0]
            ,y = xyCa[1]
            ,name = label
            ,line = dict(shape='spline')  
            )
        #add dash attributes if within dashRange
        sc.mode=mode
        if i in dashRange:
            sc.line['dash'] = '2px,6px'
            sc.marker['symbol'] = 'star'
            
        traces.append(sc)
    return go.Figure(data=traces, layout=layout)

def get_circle_data(num_obs=30, radius=.5, center=[.5,.5]):
    a = np.linspace(0,2*math.pi,num_obs)
    a = np.dstack([np.sin(a),np.cos(a)])[0]
    return radius*a+center 

def get_circle_input_target(char_data, r=.5, r_offset=.05, step_size=1):
    num_obs = char_data.shape[0]
    z_input1 = get_circle_data(num_obs=num_obs, radius=r-r_offset)
    z_input2 = get_circle_data(num_obs=num_obs, radius=r+r_offset)
    z_center = get_circle_data(num_obs=num_obs, radius=r)
    z_target = get_data_next(z_center, step_size=step_size)

    z_input=np.concatenate([z_input1,z_input2])
    z_target=np.concatenate([z_target,z_target])
    x_target=np.concatenate([char_data,char_data])
    
    return z_input, z_target, x_target

def get_data_next(data, step_size=1):
    '''
    Rotate the data a number of steps
    
    '''   
    num_obs = data.shape[0]
    data_next = np.concatenate([data[:-1],data])[step_size:num_obs+step_size]
    
    return data_next
def get_helix_points(nPoints=30, num_twists=10, r=.25,R=.5, phi_offset=0.,x_offset=.5,y_offset=.5,z_offset=.5,t_offset=0.):
    '''
    Parameter definitions:
    r=.25,R=.5 small and large donut slices
    phi_offset moves the starting point of the helix forward along the torus
    x_offset=.5,y_offset=.5,z_offset=.5 centers data around [.5,.5,.5]
    t_offset = time offset moving the value along the helix path
    
    '''    
    theta = np.linspace(0+t_offset,1+t_offset,nPoints)*2*math.pi
    phi = theta*num_twists+phi_offset

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)  
    z = r * np.sin(phi)

    x += x_offset
    y += y_offset
    z += z_offset

    return np.stack([x,y,z],1)

def get_helix_input_target_points(nPoints=100, r=.25, num_twists=5, R=.35, r_offset=.05, t_offset = 0,phi_offset=0):
    '''
    generate teh input and output dataset used for training
    '''
    if phi_offset<0:
        phi_offset = 2*math.pi/num_twists*random.random()
    d0 = get_helix_points(nPoints=nPoints,R=R,r=r-r_offset, num_twists=num_twists, phi_offset=phi_offset,t_offset=0)
    dc = get_helix_points(nPoints=nPoints,R=R, r=r, num_twists=num_twists, phi_offset=phi_offset, t_offset=t_offset)
    d1 = get_helix_points(nPoints=nPoints,R=R, r=r+r_offset, num_twists=num_twists, phi_offset=phi_offset,t_offset=0)

    x_input = np.concatenate([d0,d1])
    x_target = np.concatenate([dc,dc])
    
    return x_input,x_target

def get_helix(
    nPoints=30, num_twists=10, r=.25,R=.5, phi_offset=0.,x_offset=.5,y_offset=.5,z_offset=.5,t_start=0,t_end=1):
    '''
    r=.25,R=.5 small and large donut slices
    phi_offset rotates the starting point of the helix forward
    x_offset=.5,y_offset=.5,z_offset=.5 centers data around [.5,.5,.5]
    t_offset = time offset moving the value along the helix path
    
    '''    
    theta = np.linspace(t_start,t_end,nPoints)*2*math.pi
    phi =   (theta*num_twists)+phi_offset

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)  
    z = r * np.sin(phi)

    x += x_offset
    y += y_offset
    z += z_offset

    return np.stack([x,y,z],1)

def get_helix_input_target(
    char_target, num_twists=10, r=.25,R=.5, phi_offset=0.,
    x_offset=.5,y_offset=.5,z_offset=.5,r_offset=.05, t_offset=.015, t_start=0,t_end=1):
    '''
    Build a dataset with  3 sets of xyz columns for each offset (num points,3*num offsets)

    '''   
    nPoints = char_target.shape[0]
    d0 = get_helix(nPoints=nPoints,R=R,r=r-r_offset, num_twists=num_twists, z_offset=.5,t_start=t_start,t_end=t_end)
    dc = get_helix(nPoints=nPoints,R=R, r=r, num_twists=num_twists, z_offset=.5,
        t_start=t_start+t_offset,t_end=t_end+t_offset)
    d1 = get_helix(nPoints=nPoints,R=R, r=r+r_offset, num_twists=num_twists,z_offset=.5,t_start=t_start,t_end=t_end)

    z_input = np.concatenate([d0,d1])
    z_target = np.concatenate([dc,dc])
    x_target = np.concatenate([char_target,char_target])
    
    return z_input,z_target,x_target

def graph_char(x_stroke, seList, layout=[]):
    rList = []
    for se in seList:
        s,e = se
        gs0 = np.array([x_stroke[0,s:e,0], x_stroke[0,s:e,1]]) #x,y Stroke 0
        gs1 = np.array([x_stroke[1,s:e,0], x_stroke[1,s:e,1]]) #x,y
        gs2 = np.array([x_stroke[2,s:e,0], x_stroke[2,s:e,1]]) #x,y
        gs3 = np.array([x_stroke[3,s:e,0], x_stroke[3,s:e,1]]) #x,y
        rList.append(graphCurves([gs0,gs1,gs2,gs3,],mode='lines',layout=layout))
        
    return rList

def add_trace(subplot,trace,row,column):
    for d in trace.data:
        subplot.add_traces(d, rows=[row], cols=[column])
    return

def follow_z_path_2d(enc_Model,dec_Model,z_start_list=[[.5,.5],],num_obs=10):
    '''
    Start at a point z_point and recursively build a path through the latent z space
    and create graph
    
    '''
    p_list = []
    z_start_list=np.array(z_start_list)
    for zi in z_start_list:
        #start point
        z_point = np.array([zi],dtype=np.float32)
        zList = [z_point[0]]
        for i in range(num_obs-1):
            ph = dec_Model(z_point)
            z_point = enc_Model(ph)
            zList.append(z_point[0].numpy()) #paths are a list of single points passed through the network
        p_list.append(zList)
        zList = []
    
    return np.array(p_list)

def follow_z_path_3d(enc_Model,dec_Model,z_start_list=[[.5,.5,.5],],num_obs=10):
    '''
    Start at a point z_point and recursively build a path through the latent z space
    and create graph
    
    '''
    p_list = []
    z_start_list=np.array(z_start_list)
    for zi in z_start_list:
        #start point
        z_point = np.array([zi],dtype=np.float32)
        zList = [z_point[0]]
        for i in range(num_obs-1):
            ph = dec_Model(z_point)
            z_point = enc_Model(ph)
            zList.append(z_point[0].numpy()) #paths are a list of single points passed through the network
        p_list.append(zList)
        zList = []
    
    return np.array(p_list)

def graph_path_list_3d(path_array):
    g_list = []
    for path in path_array:
        trace = go.Scatter3d(
            x = path[:,0],
            y = path[:,1],
            z = path[:,2],
            name = str(path[0,:]),
            mode='lines', #'lines+markers'
        )
        g_list.append(trace) 
    return g_list

def graph_path_list_2d(path_array):
    g_list = []
    for path in path_array:
        trace = go.Scatter(
            x = path[:,0],
            y = path[:,1],
            name = str(path[0,:]),
            mode='lines', #'lines+markers'
        )
        g_list.append(trace) 
    return g_list

def mlpModel(dims=[]):
    activation = ELU(alpha=1)    
    kernel_initializer = 'he_uniform'

    input_layer = Input((dims[0],))
    X = input_layer
    for i in range(1,len(dims)-1):
        X = Dense(dims[i], activation=activation, kernel_initializer=kernel_initializer )(X)
    output_layer = Dense(dims[-1], activation=activation)(X)

    encoder = Model(input_layer, output_layer)
    return encoder

def pairModel(enCode,deCode):
    inputDim = enCode.layers[0].input_shape[0][1]
    input_layer = Input((inputDim,))
    X = enCode(input_layer)
    output_layer = deCode(X)   
    return Model(input_layer, output_layer)

def compile_model(model,learning_rate):
        model.compile(Adam(learning_rate=learning_rate, beta_1=0.5),
        loss='mse',
        metrics=['mae'])