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
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class BayesLands():
    def __init__(self, muted, simtime, samples, real_elev , real_erdp, real_erdp_pts, real_elev_pts, erodep_coords, filename, xmlinput, minlimits_vec, maxlimits_vec, vec_parameters, run_nb, likl_sed):
        self.filename = filename
        self.input = xmlinput
        self.real_elev = real_elev
        self.real_erdp = real_erdp
        
        self.real_erdp_pts = real_erdp_pts
        self.real_elev_pts = real_elev_pts
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
        self.burn_in = 0.0

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
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted=True)

            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))
            elev_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]
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

    def plotFunctions(self, fname, pos_likl, pos_params):
        nb_bins=30
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
        
        ax1.plot(pos_params, pos_likl)
        ax.set(xlabel='Parameter', ylabel='Likelihood', title='Pos Param vs Likelihood')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
        plt.show()

    def storeParams(self, pos_params , pos_likl):
        """
        
        """
        pos_likl = str(pos_likl)
        pos_params = str(pos_rain)

        if not os.path.isfile(('%s/exp_data.txt' % (self.filename))):
            with file(('%s/exp_data.txt' % (self.filename)),'w') as outfile:
                # outfile.write('\n# {0}\t'.format(naccept))
                outfile.write(pos_params)
                outfile.write('\t')
                outfile.write(pos_likl)
                outfile.write('\n')
        else:
            with file(('%s/exp_data.txt' % (self.filename)),'a') as outfile:
                outfile.write(pos_params)
                outfile.write('\t')
                outfile.write(pos_likl)
                outfile.write('\n')

    def likelihood_func(self,input_vector): 

        pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec, pred_elev_pts_vec = self.run_badlands(input_vector )
           
        tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 
        # tau_erodep =  np.zeros(self.sim_interval.size)  
        
        tau_erodep  =  np.sum(np.square(pred_erodep_pts_vec[self.sim_interval[len(self.sim_interval)-1]] - self.real_erodep_pts[0]))/ self.real_erodep_pts.shape[1]
       
        tau_elev =  np.sum(np.square(pred_elev_pts_vec[self.simtime] - self.real_elev_pts[0]))/ self.real_elev_pts.shape[1]

        likelihood_elev  = np.sum(-0.5 * np.log(2 * math.pi * tau_elev ) - 0.5 * np.square(pred_elev_pts_vec[self.simtime] - self.real_elev_pts[0]) / tau_elev )
        #likelihood_elev_  = np.sum(-0.5 * np.log(2 * math.pi * tausq ) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq )

        likelihood_erodep  = np.sum(-0.5 * np.log(2 * math.pi * tau_erodep ) - 0.5 * np.square(pred_erodep_pts_vec[self.sim_interval[len(self.sim_interval)-1]] - self.real_erodep_pts[0]) / tau_erodep ) # only considers point or core of erodep

        likelihood = np.sum(likelihood_elev) +  (likelihood_erodep  )

        rmse_elev = np.sqrt(tausq)
        rmse_erodep = np.sqrt(tau_erodep) 
        rmse_elev_pts = np.sqrt(tau_elev)
        avg_rmse_er = 0#np.average(rmse_erodep)
        avg_rmse_el = 0#np.average(rmse_elev_pts)

        print(likelihood_elev, likelihood_erodep, likelihood, tau_elev, rmse_elev, tau_erodep, rmse_erodep, '   likelihood_elev, likelihood_erodep, self.sedscalingfactor')

        print(likelihood ,  self.adapttemp,     ' ----    *** ------------------')

        return likelihood, rmse_elev_pts, rmse_erodep

    def likelihoodSurface(self):
        
        # Initializing variables
        samples = self.samples

        real_elev = self.real_elev
        real_erdp = self.real_erdp
        real_erdp_pts = self.real_erdp_pts

        count_list = []

        variables = np.zeros((self.vec_parameters.size,int(math.sqrt(samples))))
        print ('variables', variables.shape)
        # for i in range(len(pos_params))

        for x in range(len(self.vec_parameters)):
            
            variables[x,:] = np.linspace(self.minlimits_vec[x], self.maxlimits_vec[x], num = int(math.sqrt(samples)), endpoint = False)

        for i, v in enumerate(variables):
            print ('i and v', i , v)
            for j in v:
                vec_parameters[i] = j
                likelihood, sq_error, tau_elev, tau_erdp_pts = self.likelihoodFunc(vec_parameters)


        pos_likl = np.zeros((dimx, dimy))
        pos_sq_error = np.zeros((dimx, dimy))
        pos_tau_elev = np.zeros((dimx, dimy))
        pos_tau_erdp_pts = np.zeros((dimx, dimy))
        # print 'pos_likl', pos_likl.shape, 'pos_rain', pos_rain, 'pos_erod', pos_erod

        # Storing RMSE, tau values and adding initial run to accepted list
        start = time.time()

        i = 0
        
        for r in range(len(rain)):
            for e in range(len(erod)):
                print '\n'
                print 'Rain : ', rain[r], '  Erod : ', erod[e]
                print 'Simtime', self.simtime
                
                # Updating rain parameter and checking limits
                p_rain = rain[r]
                
                # Updating edodibility parameter and checking limits
                p_erod = erod[e]

                p_m = np.random.normal(0.5, 0.05)
                p_n = np.random.normal(1.0, 0.05)
                p_marinediff = np.random.normal(np.mean(self.marinelimit), np.std(self.marinelimit)/2)
                p_aerialdiff = np.random.normal(np.mean(self.aeriallimit), np.std(self.aeriallimit)/2)

                # Creating storage for parameters to be passed to blackBox model
                v_proposal = []
                v_proposal.append(p_rain)
                v_proposal.append(p_erod)
                v_proposal.append(p_m)
                v_proposal.append(p_n)
                v_proposal.append(p_marinediff)
                v_proposal.append(p_aerialdiff)

                # Passing paramters to calculate likelihood and rmse with new tau
                likelihood, sq_error, tau_elev, tau_erdp_pts = self.likelihoodFunc(v_proposal)
                print 'sq_error : ', sq_error, 'tau_elev :', tau_elev, 'tau_erdp_pts: ',tau_erdp_pts
                pos_erod[i] = p_erod
                pos_rain[i] = p_rain
                pos_m[i] = p_m
                pos_n[i] = p_n
                pos_marinediff[i] = p_marinediff
                pos_aerialdiff[i] = p_aerialdiff

                pos_likl[r,e] = likelihood
                pos_sq_error[r,e] = sq_error
                pos_tau_elev[r,e] = tau_elev
                pos_tau_erdp_pts[r,e] = tau_erdp_pts
                self.storeParams(i, pos_params, pos_likl[r,e])

                i += 1

        # self.plotFunctions(self.filename, pos_likl, rain, erod)
        self.viewGrid('Log_likelihood ',self.filename, pos_likl, rain, erod)
        self.viewGrid('Sum Squared Error',self.filename, pos_sq_error, rain, erod)
        end = time.time()
        total_time = end - start
        print 'counter', i, '\nTime elapsed:', total_time, '\npos_likl.shape', pos_likl.shape
        
        return (pos_rain, pos_erod, pos_likl)

def main():

    random.seed(time.time())
    muted = True
    run_nb = 0
    directory = ""
    likl_sed = False
    choice = 1#input("Please choose a Badlands example to run the likelihood surface generator on:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")
    samples = 100#input("Please enter number of samples (Make sure it is a perfect square): \n")

    directory = 'Examples/australia'
    xmlinput = '%s/AUSP1306.xml' %(directory)
    num_successive_topo = 4
    simtime = -1.49E+08
    sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    print ('Simulation time interval before',sim_interval)
    if simtime < 0:
        sim_interval = sim_interval[::-1]
    print("Simulation time interval", sim_interval)

    rain_min = 0 
    rain_max = 3
    # assume 4 regions and 4 time scales
    rain_regiongrid = 1  # how many regions in grid format 
    rain_timescale = 4  # to show climate change 
    rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
    rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 
    minlimits_others = [5.e-7, 0, 0 , 0  ,  0 , 0 , 0 , 0, 23001, 4, 0 ]  # used for Bayeslands environmental params  (stage 2) 
    maxlimits_others = [2.e-6, 1 ,  2, 0.2, 0.2, 0.2, 1, 10, 25002, 6, 0.2]
    minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
    maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)
    
    vec_parameters = [1.16, 0.9, 1.092, 1.5, 1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01]
    print('vec_parameters', vec_parameters)

    num_param = vec_parameters.size
    print('num_param', num_param)

    likl_sed = True
    erodep_coords = np.loadtxt('%s/data/erdp_coords.txt' %(directory)) #np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
    erodep_coords = np.array(erodep_coords, dtype = 'int')

    final_elev = np.loadtxt('%s/data/final_elev.txt' %(directory))
    final_erdp = np.loadtxt('%s/data/final_erdp.txt' %(directory))
    final_elev_pts = np.loadtxt('%s/data/final_elev_pts_.txt' %(directory)) 
    final_erdp_pts = np.loadtxt('%s/data/final_erdp_pts_.txt' %(directory)) 

    while os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
        run_nb+=1
    if not os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
        os.makedirs('%s/liklSurface_%s' % (directory,run_nb))
        os.makedirs('%s/liklSurface_%s/plots' % (directory,run_nb))
        os.makedirs('%s/liklSurface_%s/prediction_data' % (directory,run_nb))
        filename = ('%s/liklSurface_%s' % (directory,run_nb))

    print '\nInput file shape', final_elev.shape, '\n'
    run_nb_str = 'liklSurface_' + str(run_nb)

    bLands = BayesLands(muted, simtime, samples, final_elev, final_erdp, final_erdp_pts,final_elev_pts, erodep_coords, filename, xmlinput, minlimits_vec, maxlimits_vec, vec_parameters, run_nb_str, likl_sed)
    [pos_rain, pos_erod, pos_likl] = bLands.likelihoodSurface()

    print 'Results are stored in ', filename

    print 'Finished producing Likelihood Surface'

if __name__ == "__main__": main()