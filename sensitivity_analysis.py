##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##

#Main Contributer: Danial Azam  Email: dazam92@gmail.com

"""
This script is intended to implement functionality to  generate the likelihood surface of the free parameters.

"""
import os
import numpy as np
import random
import time
import math
import copy
import fnmatch
import shutil
import plotly
import collections
import plotly.plotly as py
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from copy import deepcopy
from pylab import rcParams
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from sklearn.preprocessing import normalize
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import argparse

parser=argparse.ArgumentParser(description='Sensitivity')

parser.add_argument('-vstart','--vstart', required=True, dest="vstart",type=int)
parser.add_argument('-vend', '-vend', required =True, dest="vend", type=int)

args = parser.parse_args()
vstart = args.vstart
vend = args.vend

class BayesLands():
    def __init__(self, muted, simtime, sim_interval, samples, real_elev , real_erodep, real_erodep_pts, real_elev_pts, elev_coords, erodep_coords, filename, xmlinput, minlimits_vec, maxlimits_vec, vec_parameters, run_nb, likl_sed, ocean_t):
        self.filename = filename
        self.input = xmlinput
        self.real_elev = real_elev
        self.real_erodep = real_erodep
        
        self.real_erodep_pts = real_erodep_pts
        self.real_elev_pts = real_elev_pts
        self.elev_coords = elev_coords
        self.erodep_coords = erodep_coords
        self.likl_sed = likl_sed

        self.simtime = simtime
        self.samples = samples
        self.run_nb = run_nb
        self.muted = muted
        self.minlimits_vec = minlimits_vec
        self.maxlimits_vec = maxlimits_vec
        self.vec_parameters = vec_parameters

        self.simtime = simtime
        self.sim_interval = sim_interval
        self.burn_in = 0.0
        self.ocean_t = ocean_t

    def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs
 
        rain_regiontime = 1 # number of parameters for rain based on  region and time 

        #Create a badlands model instance
        model = badlandsModel()

        #----------------------------------------------------------------
        # Load the XmL input file
        model.load_xml(str(self.run_nb), self.input, muted=True)

        init = False

        if init == True:

            geoparam  = rain_regiontime+11  # note 10 parameter space is for erod, c-marine etc etc, some extra space ( taking out time dependent rainfall)
            inittopo_vec = input_vector[geoparam:]
            filename=self.input.split("/")
            problem_folder=filename[0]+"/"+filename[1]+"/"

            #Update the initial topography
            #Use the coordinates from the original dem file
            xi=int(np.shape(model.recGrid.rectX)[0]/model.recGrid.nx)
            yi=int(np.shape(model.recGrid.rectY)[0]/model.recGrid.ny)
            #And put the demfile on a grid we can manipulate easily
            elev=np.reshape(model.recGrid.rectZ,(xi,yi)) 

            inittopo_estimate = self.process_inittopo(inittopo_vec)  
            inittopo_estimate = inittopo_estimate[0:  elev.shape[0], 0:  elev.shape[1]]  # bug fix but not good fix - temp @ 

            #Put it back into 'Badlands' format and then re-load the model
            filename=problem_folder+str(self.run_nb)+'/demfile_'+ str(int(self.temperature*10)) +'_demfile.csv' 

            elev_framex = np.vstack((model.recGrid.rectX,model.recGrid.rectY,inittopo_estimate.flatten()))
            np.savetxt(filename, elev_framex.T, fmt='%1.2f' ) 
            model.input.demfile=filename 
            model.build_mesh(model.input.demfile, verbose=False)

        # Adjust precipitation values based on given parameter
        #print(input_vector[0:rain_regiontime] )
        # print('model.force.rainVal', model.force.rainVal)

        # model.force.rainVal  = input_vector[0:rain_regiontime] 

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[rain_regiontime]  
        model.flow.erodibility.fill(input_vector[rain_regiontime ] )

        # Adjust m and n values
        model.input.SPLm = input_vector[rain_regiontime+1]  
        model.input.SPLn = input_vector[rain_regiontime+2] 

        #Check if it is the etopo extended problem
        #if problem == 4 or problem == 3:  # will work for more parameters
        model.input.CDm = input_vector[rain_regiontime+3] # submarine diffusion
        model.input.CDa = input_vector[rain_regiontime+4] # aerial diffusion

        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()
        elev_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            # print ('Interval ', x )
            # print('model.force.rainVal', model.force.rainVal, 'simtime', self.simtime)
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted=True)

            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros(self.erodep_coords.shape[0])
            elev_pts = np.zeros(self.elev_coords.shape[0])

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]

            for count, val in enumerate(self.elev_coords):
                elev_pts[count] = elev[val[0], val[1]]

            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts
            elev_pts_vec[self.simtime] = elev_pts
            
 
        return elev_vec, erodep_vec, erodep_pts_vec, elev_pts_vec

    def interpolateArray(self, coords=None, z=None, dz=None):
        """
        Interpolate the irregular spaced dataset from badlands on a regular grid.
        """
        x, y = np.hsplit(coords, 2)
        dx = (x[1]-x[0])[0]
        nx = int((x.max() - x.min())/dx+1 - 2)
        ny = int((y.max() - y.min())/dx+1 - 2)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))

        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if len(z[indices].shape) == 3:
            z_vals = z[indices][:,:,0]
            dz_vals = dz[indices][:,:,0]
        else:
            z_vals = z[indices]
            dz_vals = dz[indices]

        zi = np.average(z_vals,weights=(1./distances), axis=1)
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            dzi[onIDs] = dz[indices[onIDs,0]]
        zreg = np.reshape(zi,(ny,nx))
        dzreg = np.reshape(dzi,(ny,nx))
        return zreg,dzreg

    def plotFunctions(self, fname, pos_rmse, pos_params):
        font = 9
        width = 1

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title(' Likelihood', fontsize=  font+2)#, y=1.02)
        
        ax1 = fig.add_subplot(211, projection = '3d')
        ax1.set_facecolor('#f2f2f3')
        
        ax1.plot(pos_params, pos_rmse)
        ax.set(xlabel='Parameter', ylabel='Likelihood', title='Pos Param vs Likelihood')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
        plt.show()

    def storeParams(self, count, w , pos_rmse, var):
        """
        
        """
        pos_rmse = str(pos_rmse)
        w = str(w)

        if not os.path.isfile(('%s/prediction_data/%s.txt' % (self.filename, var))):
            with file(('%s/prediction_data/%s.txt' % (self.filename, var)),'w') as outfile:
                # outfile.write('\n# {0}\t'.format(naccept))
                outfile.write(pos_rmse)
                outfile.write('\n')
        else:
            with file(('%s/prediction_data/%s.txt' % (self.filename, var)),'a') as outfile:
                outfile.write(pos_rmse)
                outfile.write('\n')

        if not os.path.isfile(('%s/prediction_data/w%s.txt' % (self.filename, var))):
            with file(('%s/prediction_data/w%s.txt' % (self.filename, var)),'w') as outfile:
                # outfile.write('\n# {0}\t'.format(naccept))
                outfile.writelines(w)
                outfile.write('\n')
        else:
            with file(('%s/prediction_data/w%s.txt' % (self.filename, var)),'a') as outfile:
                outfile.writelines(w)
                outfile.write('\n')


    def likelihood_func(self,input_vector): 
        
        pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec, pred_elev_pts_vec = self.run_badlands(input_vector)

        likelihood_elev_ocean = 0
        rmse_ocean = np.zeros(self.sim_interval.size)
        i = 0

        for i, time in enumerate(self.sim_interval):
            p_elev_ocean = pred_elev_vec[time]
            r_elev_ocean = self.ocean_t[-i,:,:]

            # r_elev_ocean[r_elev_ocean<0] = 0 
            # r_elev_ocean[r_elev_ocean>0] = 1

            p_elev_ocean[p_elev_ocean>0] = 0
            p_elev_ocean[p_elev_ocean<0] = 1 

            matches = np.count_nonzero(p_elev_ocean==r_elev_ocean)
            non_matches = p_elev_ocean.size - matches
            
            print('\n time ', time, ' matches : ', matches ,'  non matches : ', non_matches, 'percentage non match', (float(non_matches)/float(p_elev_ocean.size))*100)

            # fig = plt.figure()
            # plt.imshow(p_elev_ocean, cmap='hot', interpolation='nearest')
            # plt.savefig(self.filename +'/pred_plots/'+ str(time) +'p_elev_ocean_original.png')
            # plt.close()

            # fig = plt.figure()
            # plt.imshow(r_elev_ocean, cmap='hot', interpolation='nearest')
            # plt.savefig(self.filename +'/pred_plots/'+ str(time) +'r_elev_ocean.png')
            # plt.close()

            tausq_ocean = np.sum(np.square(p_elev_ocean - r_elev_ocean))/self.real_elev.size  
            rmse_ocean[i] = tausq_ocean
            likelihood_elev_ocean  += np.sum(-0.5 * np.log(2 * math.pi * tausq_ocean) - 0.5 * np.square(p_elev_ocean - r_elev_ocean) /  tausq_ocean )
            i = i+ 1

        print('rmse_ocean', rmse_ocean)

        tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 
        tau_elev =  np.sum(np.square(pred_elev_pts_vec[self.simtime] - self.real_elev_pts)) / self.real_elev_pts.shape[0]
        tau_erodep  =  np.sum(np.square(pred_erodep_pts_vec[self.simtime] - self.real_erodep_pts))/ self.real_erodep_pts.shape[0]
        
        likelihood_elev  = np.sum(-0.5 * np.log(2 * math.pi * tau_elev ) - 0.5 * np.square(pred_elev_pts_vec[self.simtime] - self.real_elev_pts) / tau_elev )
        likelihood_erodep  = np.sum(-0.5 * np.log(2 * math.pi * tau_erodep ) - 0.5 * np.square(pred_erodep_pts_vec[self.sim_interval[len(self.sim_interval)-1]] - self.real_erodep_pts[0]) / tau_erodep ) # only considers point or core of erodep        
        likelihood =  (likelihood_elev/4) +  (likelihood_erodep/8) + (likelihood_elev_ocean/10) 
         
        rmse_elev = np.sqrt(tausq)
        rmse_elev_ocean = np.average(rmse_ocean)
        rmse_erodep = np.sqrt(tau_erodep) 
        rmse_elev_pts = np.sqrt(tau_elev)
        avg_rmse_er = 0#np.average(rmse_erodep)
        avg_rmse_el = 0#np.average(rmse_elev_pts)
        rmse_total = rmse_elev_pts + rmse_erodep + rmse_elev_ocean

        # print(likelihood_elev, likelihood_erodep, likelihood, tau_elev, rmse_elev, tau_erodep, rmse_erodep, '   likelihood_elev, likelihood_erodep, self.sedscalingfactor')

        return likelihood, rmse_elev_pts, rmse_erodep, rmse_elev_ocean, rmse_total

    def likelihoodSurface(self):
        
        # Initializing variables
        samples = self.samples

        real_elev = self.real_elev
        real_erodep = self.real_erodep
        real_erodep_pts = self.real_erodep_pts

        count_list = []
        font = 9
        width = 1
        variables = np.zeros((self.vec_parameters.size,samples))
        pos_rmse = np.zeros((variables.shape[0], variables.shape[1]))
        pos_likl = np.zeros((variables.shape[0], variables.shape[1]))
        pos_rmse_elev = np.zeros((variables.shape[0], variables.shape[1]))
        pos_rmse_erodep = np.zeros((variables.shape[0], variables.shape[1]))
        pos_rmse_elev_ocean = np.zeros((variables.shape[0], variables.shape[1]))
        pos_rmse_total = np.zeros((variables.shape[0], variables.shape[1]))

        print ('variables', variables.shape)

        for x in range(vstart,vend):
            variables[x,:] = np.linspace(self.minlimits_vec[x], self.maxlimits_vec[x], num = samples, endpoint = False)

        for i, v in enumerate(variables[vstart:vend,:]):
            for j, w in enumerate(v):
                start = time.time()
                print ('i ', i,' v',v)

                v_prop = self.vec_parameters
                v_prop[i] = w

                print ('j', j, 'w', w)

                # print ('v_prop', v_prop)
                likelihood, rmse_elev, rmse_erodep, rmse_elev_ocean, rmse_total = self.likelihood_func(v_prop)
                print ('rmse_elev ', rmse_elev)
                pos_rmse_elev[vstart + i,int(j)] = rmse_elev
                pos_rmse_erodep[vstart + i,int(j)] = rmse_erodep
                pos_rmse_elev_ocean[vstart + i,int(j)] = rmse_elev_ocean
                pos_rmse_total[vstart + i,int(j)] = rmse_total
                
                self.storeParams(i, w, pos_rmse_elev[vstart + i,int(j)], 'elev_%s' %(i))
                self.storeParams(i, w, pos_rmse_erodep[vstart + i,int(j)], 'erodep%s' %(i))
                # self.storeParams(i, self.vec_parameters, pos_rmse_elev_ocean[vstart + i,int(j)], 'elev_ocean%s' %(i))
                # self.storeParams(i, self.vec_parameters, pos_rmse_total[vstart + i,int(j)], 'total%s' %(i))

                end = time.time()
                total_time = end - start
                print '\nTime elapsed:', total_time

            font = 20
            fig = plt.figure(figsize=(15,15))
            
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            # ax.tick_params(labelcolor='w')
            
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.set_facecolor('#f1f1f1')
            ax1.plot(variables[vstart + i,:], pos_rmse_elev[vstart + i,:])
            ax1.set_title(r'Elevation',size=font+2)
            ax1.set_xlabel('Parameter value', size = font)
            ax1.set_ylabel('RMSE elev', size = font)
            ax1.tick_params(axis='both', which='major', labelsize=20)
            ax1.tick_params(axis='both', which='minor', labelsize=20)
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))
            # ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))


            ax2 = fig.add_subplot(212)
            ax2.grid(True)
            ax2.set_facecolor('#f1f1f1')
            ax2.plot(variables[vstart + i,:], pos_rmse_erodep[vstart + i,:])
            ax2.set_title(r'Erosion Deposition',size=font+2)
            ax2.set_xlabel('Parameter value', size = font)
            ax2.set_ylabel('RMSE erodep', size = font)
            ax2.tick_params(axis='both', which='major', labelsize=20)
            ax2.tick_params(axis='both', which='minor', labelsize=20)
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))
            # ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%1.3f'))

            plt.savefig('%s/plot_%s_elev_erdp.png'% (self.filename,vstart+ i), bbox_inches='tight', dpi=300, transparent=False)
            plt.close()


        font = 20
        fig = plt.figure(figsize=(15,15))
        
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        # ax.tick_params(labelcolor='w')
        
        ax1 = fig.add_subplot(211)
        ax1.grid(True)
        ax1.set_facecolor('#f1f1f1')

        x = np.arange(pos_rmse_elev.shape[1])
        print('pos_rmse_elev.shape', pos_rmse_elev.shape, 'y', pos_rmse_elev[1,:].shape)
        for y in range(vend):
            ax1.plot(x, pos_rmse_elev[y,:])
        
        ax1.set_title(r'Elevation',size=font+2)
        ax1.set_xlabel('Parameter value', size = font)
        ax1.set_ylabel('RMSE elev', size = font)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))
        # ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))


        ax2 = fig.add_subplot(212)
        ax2.grid(True)
        ax2.set_facecolor('#f1f1f1')
        
        x = np.arange(pos_rmse_elev.shape[1])
        for y in range(vend):
            ax2.plot(x, pos_rmse_erodep[y,:])

        ax2.set_title(r'Erosion Deposition',size=font+2)
        ax2.set_xlabel('Parameter value', size = font)
        ax2.set_ylabel('RMSE erodep', size = font)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%1.2f'))
        # ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%1.3f'))

        plt.savefig('%s/plot_sensitivity.png'% (self.filename), bbox_inches='tight', dpi=300, transparent=False)
        plt.close()

        print 'counter', i, '\nTime elapsed:', total_time, '\npos_rmse.shape', pos_rmse.shape
        
        return

def main():

    random.seed(time.time())
    muted = True
    run_nb = 0
    directory = ""
    likl_sed = False
    choice = 1#input("Please choose a Badlands example to run the likelihood surface generator on:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")

    samples = 100 #input("Please enter number of samples (Make sure it is a perfect square): \n")

    directory = 'Examples/australia'
    xmlinput = '%s/AUSP1306.xml' %(directory)
    num_successive_topo = 4
    simtime = -1.49E+08


    rain_min = 0  
    rain_max = 2
    # assume 4 regions and 4 time scales
    rain_regiongrid = 1  # how many regions in grid format 
    rain_timescale = 4  # to show climate change 
    rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
    rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 
    minlimits_others = [5.e-7, 0, 0, 0,  0 , 0 , 0 , 0, 23001, 4, 0 ]  # used for Bayeslands environmental params  (stage 2) 
    maxlimits_others = [2.e-6, 1, 2, 0.2, 0.2, 0.2, 1, 10, 25002, 6, 0.2]
    minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
    maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)
    
    vec_parameters = np.array([1.16, 0.9, 1.092, 1.5, 1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01])
    print('vec_parameters', vec_parameters)

    num_param = vec_parameters.size
    print('num_param', num_param)

    likl_sed = True

    elev_coords = np.loadtxt('%s/data/coord_final_elev_mt.txt' %(directory)) #np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
    elev_coords = np.array(elev_coords, dtype = 'int')

    erodep_coords = np.loadtxt('%s/data/erdp_coords.txt' %(directory)) #np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
    erodep_coords = np.array(erodep_coords, dtype = 'int')

    final_elev = np.loadtxt('%s/data/final_elev.txt' %(directory))
    final_erodep = np.loadtxt('%s/data/final_erdp.txt' %(directory))
    final_elev_pts = np.loadtxt('%s/data/elev_pts_updated_mt.txt' %(directory)) 
    final_erodep_pts = np.loadtxt('%s/data/final_erdp_pts_.txt' %(directory)) 

    while os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
        run_nb+=1
    if not os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
        os.makedirs('%s/liklSurface_%s' % (directory,run_nb))
        os.makedirs('%s/liklSurface_%s/plots' % (directory,run_nb))
        os.makedirs('%s/liklSurface_%s/prediction_data' % (directory,run_nb))
        filename = ('%s/liklSurface_%s' % (directory,run_nb))

    print '\nInput file shape', final_elev.shape, '\n'
    run_nb_str = 'liklSurface_' + str(run_nb)


    sim_interval = np.array([0, -5.0e06 , -25.0e06, -30.0e06,  -40.0e06, -50.0e06 , -75.0e06 , -100.0e06,  -115.0e06, -125.0e06, -1.40e08,  -1.49e08])
    filename_ocean = np.array([0, 5 , 25 , 30, 40, 50, 75, 100, 115,  125, 140, 149])

    # ## 1 MA 
    # sim_interval = np.array([0, -5.0e04 , -25.0e04, -50.0e04 , -75.0e04 , -100.0e04, -125.0e04, -1.49e06])
    # filename_ocean = np.array([0, 5, 25, 50, 75, 100, 125, 149])
 
    print ('Simulation time interval before',sim_interval)
    if simtime < 0:
        sim_interval = sim_interval[::-1]
        filename_ocean = filename_ocean[::-1]

    print("Simulation time interval", sim_interval)
    print()

    ocean_t = np.zeros((sim_interval.size,final_elev.shape[0], final_elev.shape[1]))

    for i, val in enumerate(filename_ocean): 
        temp = np.loadtxt(directory+ '/data/ocean/marine_%s.txt' %(val))
        ocean_t[i,:,:] = temp


    bLands = BayesLands(muted, simtime, sim_interval, samples, final_elev, final_erodep, final_erodep_pts,final_elev_pts, elev_coords, erodep_coords, filename, xmlinput, minlimits_vec, maxlimits_vec, vec_parameters, run_nb_str, likl_sed, ocean_t)
    bLands.likelihoodSurface()

    print 'Results are stored in ', filename

    print 'Finished producing Likelihood Surface'

if __name__ == "__main__": main()