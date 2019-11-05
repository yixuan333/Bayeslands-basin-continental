import os
import sys
import random
import operator
import math 
import copy
import fnmatch
import shutil
import time
import numpy as np
from pyBadlands.model import Model as badlandsModel

def problem_setup(problem = 1):
    random.seed(time.time()) 

    inittopo_estimated = np.array([])

    if problem == 1: #this will have region and time rainfall of Problem 1
        problemfolder = 'Examples/etopo_extended/'
        datapath = problemfolder + 'data/final_elev.txt'
        
        groundtruth_elev = np.loadtxt(datapath)
        init_elev = np.loadtxt(problemfolder+ 'data/pred_mean_0.0_.txt')
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')
        groundtruth_elev_pts = np.loadtxt(problemfolder + 'data/final_elev_pts.txt')
        res_summaryfile = '/results.txt'
        #inittopo_expertknow = np.loadtxt(problemfolder + 'data/inittopo_groundtruthfine.txt') #  expert knowledge  20 x 20
        inittopo_expertknow = np.loadtxt(problemfolder + 'data/init_topo_20_20.txt') #  expert knowledge 10 x 10
 
        inittopo_estimated = []

        inittopo_expertknow = inittopo_expertknow 

        
         
        simtime = 1000000
        resolu_factor = 1
        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True
        real_rain = 1.5 #m/a
        real_erod = 5.e-6 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 8.e-1 #aerial diffusion
        rain_min = 1.5
        rain_max = 1.5 
        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 

        if rain_timescale ==4:
            xmlinput = problemfolder + 'etopo.xml'
        elif rain_timescale ==8:
            xmlinput = problemfolder + 'etopo_t8.xml' 
        elif rain_timescale ==16:
            xmlinput = problemfolder + 'etopo_t16.xml'

        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale)
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale)
        minlimits_others = [3.e-6, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [7.e-6, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]

        #minlimits_others = [5.e-6, 0.5, 1, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        #maxlimits_others = [5.e-6, 0.5, 1, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]
        #minlimits_vec = np.append(rain_minlimits,minlimits_others)
        #maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)
        #print(maxlimits_vec, ' maxlimits ')


        epsilon = 0.5 

        inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        inittopo_gridwidth = 20


        len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
        wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over





         
        inittopo_minlimits = np.repeat( -200   , inittopo_gridlen*inittopo_gridwidth)
        inittopo_maxlimits = np.repeat(200 , inittopo_gridlen*inittopo_gridwidth)
 

        #--------------------------------------------------------


        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)



        temp_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        minlimits_vec = np.append(temp_vec, inittopo_minlimits)

        temp_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
        maxlimits_vec = np.append(temp_vec, inittopo_maxlimits)

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters, 'vec_parameters')

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters) 
        erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem

    elif problem == 2: # Aus: 149 MILLION YEARS with INIT TOPO

        problemfolder = 'Examples/australia/'
        xmlinput = problemfolder + 'AUSP1306.xml'

        inittopo_expertknow = np.loadtxt(problemfolder + 'data/init_topo_fused.txt')  # no expert knowledge as simulated init topo



        print(inittopo_expertknow, ' * **************** inittopo_expertknow ********************** ')
         
        simtime = -1.49e06
        resolu_factor = 1 

        init_elev = np.loadtxt(problemfolder+ 'data/initial_elev.txt')
        groundtruth_elev = np.loadtxt(problemfolder +'data/final_elev_filtered_ocean.txt')
        groundtruth_erodep = np.loadtxt(problemfolder +'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder +'data/final_erdp_pts_.txt')
        # groundtruth_elev_pts = np.loadtxt(problemfolder +'data/final_elev_pts_.txt')
        groundtruth_elev_pts = np.loadtxt(problemfolder +'data/elev_pts_updated.txt')
 
        inittopo_estimated = [] #np.loadtxt(problemfolder + 'init_expertknowlegeprocess/init_estimated.txt') 

        res_summaryfile = '/results.txt'


        inittopo_expertknow = inittopo_expertknow  # no expert knowledge as simulated init topo

        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

        real_rain = 1.5 #m/a
        real_erod = 1.e-6 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 0.005 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 0.001 #aerial diffusion

        rain_min = 0 
        rain_max = 3

        #rain_min = 1.45 
        #rain_max = 1.55

        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 
        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

        #--------------------------------------------------------
        #minlimits_others = [8.e-7, 0.50, 0.8, 0.003 ,  0.0008, 0.0008, 0.4, 4, 24001, 4, 0.005]  # used for Bayeslands initopo (stage 1) 
        #maxlimits_others = [1.2e-6, 0.55 , 1.2, 0.006, 0.002, 0.0012, 0.6, 6, 24002, 6, 0.02] # from Carmen
 
        minlimits_others = [9.e-7, 0, 0 , 0  ,  0 , 0 , 0 , 0, 23001, 4, 0 ]  # used for Bayeslands environmental params  (stage 2) 
        maxlimits_others = [2.e-6, 1 ,  2, 0.2, 0.01, 0.01, 1, 10, 25002, 6, 0.05]


        #variables[:15] = [1.0, 1.0, 1.0, 1.0, 1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01]
 
 
 
 
 
 
        #----------------------------------------InitTOPO

        #inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        #inittopo_gridwidth = 20

        epsilon = 0.5 

        inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        inittopo_gridwidth = 20


        len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
        wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over

        print(len_grid, wid_grid, groundtruth_elev.shape[0], groundtruth_elev.shape[1] ,'  sub_gridlen, sub_gridwidth   ------------ ********')

         
        inittopo_minlimits = np.repeat( -200  , inittopo_gridlen*inittopo_gridwidth)
        inittopo_maxlimits = np.repeat(200 , inittopo_gridlen*inittopo_gridwidth)
 

        #--------------------------------------------------------


        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)



        temp_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        minlimits_vec = np.append(temp_vec, inittopo_minlimits)

        temp_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
        maxlimits_vec = np.append(temp_vec, inittopo_maxlimits)

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters, 'vec_parameters') 

        # erodep_coords = np.loadtxt(problemfolder +"data/erdp_coords.txt", ) #np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
        # print('erdp_coords', erodep_coords.shape)
        erodep_coords = np.loadtxt(problemfolder +"data/coord_final_elev.txt", ) #np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
        print('erdp_coords', erodep_coords.shape)        
        erodep_coords = np.array(erodep_coords, dtype = 'int')

    return (problemfolder, xmlinput, simtime, resolu_factor, init_elev ,groundtruth_elev, groundtruth_erodep,
    groundtruth_erodep_pts, groundtruth_elev_pts,  res_summaryfile, inittopo_expertknow, len_grid, wid_grid, simtime, 
    resolu_factor, likelihood_sediment, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erodep_coords, inittopo_estimated, vec_parameters,minlimits_vec, maxlimits_vec)