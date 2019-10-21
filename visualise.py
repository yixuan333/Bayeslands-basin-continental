

#Main Contributers:   Rohitash Chandra and Ratneel Deo  Email: c.rohitash@gmail.com, deo.ratneel@gmail.com

# Bayeslands II: Parallel tempering for multi-core systems - Badlands

from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math 
from pylab import rcParams
import copy
from copy import deepcopy 
from pylab import rcParams
import collections
from scipy import special
import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os
import shutil
import sys
import matplotlib.mlab as mlab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
from IPython.display import HTML
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas
import argparse
import cmocean as cmo
import pandas as pd
import seaborn as sns
from scipy.ndimage import filters 
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from problem_setup import problem_setup
from ptBayeslands import interpolateArray

import glob
# import lavavu
import viewBadlands as visu
import stratalArchitecture as strata
import matplotlib as mpl
from matplotlib import mlab, cm
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
import pandas as pd
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)


#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True,   dest="problem",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.02,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)  
parser.add_argument('-rain_intervals','--rain_intervals', help='rain_intervals', dest="rain_intervals",default=4,type=int)
parser.add_argument('-epsilon','--epsilon', help='epsilon for inital topo', dest="epsilon",default=0.5,type=float)
parser.add_argument('-cov','--covariance', help='flag for covariance', dest="covariance",default=0,type=int)


args = parser.parse_args()
    
#parameters for Parallel Tempering
problem = args.problem
samples = args.samples #10000  # total number of samples by all the chains (replicas) in parallel tempering
num_chains = args.num_chains
swap_ratio = args.swap_ratio
burn_in=args.burn_in
#maxtemp = int(num_chains * 5)/args.mt_val
maxtemp =   args.mt_val 
swap_interval = int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
num_successive_topo = 4
pt_samples = args.pt_samples
epsilon = args.epsilon
rain_intervals = args.rain_intervals

method = 1 # type of formaltion for inittopo construction (Method 1 showed better results than Method 2)

class results_visualisation:

    def __init__(self, vec_parameters, inittopo_expertknow, inittopo_estimated, rain_regiongrid, rain_timescale, len_grid,  wid_grid, num_chains, maxtemp, samples,swap_interval,fname, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, simtime, sim_interval, resolu_factor,  xmlinput,  run_nb_str, init_elev ):

   
        self.swap_interval = swap_interval
        self.folder = fname
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = samples
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
        self.real_erodep_pts  = groundtruth_erodep_pts
        self.real_elev = groundtruth_elev
        self.resolu_factor =  resolu_factor
        self.num_param = num_param
        self.erodep_coords = erodep_coords
        self.simtime = simtime
        self.sim_interval = sim_interval 
        self.xmlinput = xmlinput
        self.run_nb_str =  run_nb_str
        self.vec_parameters = vec_parameters
        #self.realvalues  =  realvalues_vec 
        self.burn_in = burn_in
        self.input = xmlinput
        # create queues for transfer of parameters between process chain
        self.geometric =  True
        self.total_swap_proposals = 0
        self.rain_region = rain_regiongrid  
        self.rain_time = rain_timescale
        self.len_grid = len_grid
        self.wid_grid = wid_grid
        self.inittopo_expertknow =  inittopo_expertknow 
        self.inittopo_estimated = inittopo_estimated
        self.init_elev = init_elev

    def results_current (self ):

        #pos_param, likelihood_rep, accept_list, pred_topo,  combined_erodep, accept, pred_topofinal, list_xslice, list_yslice, rmse_elev, rmse_erodep = self.show_results('chain_')

        posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts  = self.show_results('chain_')


        self.view_crosssection_uncertainity(xslice, yslice)

        optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_vec, posterior)
        np.savetxt(self.folder+'/optimal_percentile_para.txt', np.array([optimal_para, para_5thperc, para_95thperc]) )

        #for s in range(self.num_param): 
        for s in range(45): # change this if you want to see all pos plots
            self.plot_figure(posterior[s,:], 'pos_distri_'+str(s) ) 

    

        rain_regiontime = self.rain_region * self.rain_time # number of parameters for rain based on  region and time  
        geoparam  = rain_regiontime+11  # note 10 parameter space is for erod, c-marine etc etc, some extra space ( taking out time dependent rainfall) 
 
        mean_pos = posterior.mean(axis=1)
        std_pos = posterior.std(axis=1)

        np.savetxt(self.folder+'/mean_pos.txt', mean_pos) 
        np.savetxt(self.folder+'/std_pos.txt', std_pos) 

        percentile_95th = np.percentile(posterior, 95, axis=1) 

        percentile_5th = np.percentile(posterior, 5, axis=1) 

        #if problem == 0 or problem ==1 or problem ==2 or problem==3: # problem is global variable
         #   init = False
        #else:
        init = True # when you need to estimate initial topo

        if init == True: 
            init_topo_mean = self.process_inittopo(mean_pos[geoparam:])
            init_topo_95th = self.process_inittopo(percentile_95th[geoparam:])
            init_topo_5th = self.process_inittopo(percentile_5th[geoparam:])

            self.plot3d_plotly(init_topo_mean, 'mean_init')
            self.plot3d_plotly(init_topo_95th, 'percentile95_init')
            self.plot3d_plotly(init_topo_5th, 'percentile5_init')


        #   cut the slice in the middle to show cross section of init topo with uncertainity
            '''synthetic_initopo = self.get_synthetic_initopo()

            '''


            init_topo_mean = init_topo_mean[0:self.real_elev.shape[0], 0:self.real_elev.shape[1]]  # just to ensure that the size is exact 
            init_topo_95th = init_topo_95th[0:self.real_elev.shape[0], 0:self.real_elev.shape[1]]  # just to ensure that the size is exact 
            init_topo_5th = init_topo_5th[0:self.real_elev.shape[0], 0:self.real_elev.shape[1]]  # just to ensure that the size is exact  

            xmid = int(init_topo_mean.shape[0]/2) 
            inittopo_real =  init_topo_mean[xmid, :]  # ground-truth init topo mid (synthetic) 
            #inittopo_real = self.inittopo_estimated[xmid, :]  # ground-truth init topo mid (synthetic) 

            lower_mid = init_topo_5th[xmid, :]
            higher_mid = init_topo_95th[xmid, :]
            mean_mid = init_topo_mean[xmid, :] 
            x = np.linspace(0, self.real_elev.shape[1] * self.resolu_factor, num= self.real_elev.shape[1])
            rmse_slice_init = self.cross_section(x, mean_mid, inittopo_real, lower_mid, higher_mid, 'init_x_ymid_cross') # not needed in Australia problem 
             
            #print(self.real_elev, ' real_elev')

            

        else:
 
            rmse_slice_init =  0

        return  posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts, rmse_slice_init 




    def full_crosssection(self,  simulated_topo, real_elev):

        ymid = int( real_elev.shape[1]/2)  

        x = np.linspace(0, real_elev.shape[0], num=real_elev.shape[0])

        

        x_m = np.arange(0,real_elev.shape[0], 10)

 

        for i in x_m:
            xmid = i 

            real = real_elev[0:real_elev.shape[0], i]  
            pred = simulated_topo[0:real_elev.shape[0], i]
 

            size = 15

            plt.tick_params(labelsize=size)
            params = {'legend.fontsize': size, 'legend.handlelength': 2}
            plt.rcParams.update(params)
            plt.plot(x, real, label='Ground Truth') 
            plt.plot(x, pred, label='Badlands Pred.')
            #plt.plot(x, init, label = 'Initial Topo')
            plt.grid(alpha=0.75)
            plt.legend(loc='best')  
            plt.title("Topography cross section   ", fontsize = size)
            plt.xlabel(' Distance (x 50 km)  ', fontsize = size)
            plt.ylabel(' Height (m)', fontsize = size)
            plt.tight_layout()
              
            plt.savefig(self.folder+'/cross_section/'+str(i)+'_cross-sec_postcompare.pdf')
            plt.clf()

        fnameplot = self.folder +  '/cross_section/realmap_postcompare.png' 
        # im = plt.imshow(real_elev, cmap='hot', interpolation='nearest')
        plt.colorbar(im) 
        plt.savefig(fnameplot)
        plt.clf()

        fnameplot = self.folder +  '/cross_section/predmap_postcompare.png' 
        # im = plt.imshow(simulated_topo, cmap='hot', interpolation='nearest')
        plt.colorbar(im) 
        plt.savefig(fnameplot)
        plt.clf()

        fnameplot = self.folder +  '/cross_section/diffmap_postcompare.png' 
        # im = plt.imshow(real_elev- simulated_topo, cmap='hot', interpolation='nearest')
        plt.colorbar(im) 
        plt.savefig(fnameplot)
        plt.clf()

 



    def plot3d_plotly(self, zData, fname): # same method from previous class - ptReplica
        zmin =  zData.min() 
        zmax =  zData.max()

        tickvals= [0,50,75,-50]
        height=1000
        width=1000
        #title='Topography'
        resolu_factor = 1

        xx = (np.linspace(0, zData.shape[0]* resolu_factor, num=zData.shape[0]/10 )) 
        yy = (np.linspace(0, zData.shape[1] * resolu_factor, num=zData.shape[1]/10 )) 

        xx = np.around(xx, decimals=0)
        yy = np.around(yy, decimals=0)
         
        # range = [0,zData.shape[0]* self.resolu_factor]
        #range = [0,zData.shape[1]* self.resolu_factor],

        data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        layout = Layout(title='' , autosize=True, width=width, height=height,scene=Scene(
                    zaxis=ZAxis(title = ' Elev.(m) ', range=[zmin,zmax], autorange=False, nticks=6, gridcolor='rgb(255, 255, 255)',
                                gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
                    xaxis=XAxis(title = ' x ',  tickvals= xx,      gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
                    yaxis=YAxis(title = ' y ', tickvals= yy,    gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
                    bgcolor="rgb(244, 244, 248)"
                )
            )

        fig = Figure(data=data, layout=layout) 
        graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +  '/recons_initialtopo/'+fname+'_.html', validate=False)
        np.savetxt(self.folder +  '/recons_initialtopo/'+fname+'_.txt', zData,  fmt='%1.2f' )

    def process_inittopo(self, inittopo_vec):
 

        length = self.real_elev.shape[0]
        width = self.real_elev.shape[1]
        len_grid = self.len_grid
        wid_grid = self.wid_grid
        sub_gridlen = 20 #int(length/len_grid)  # 25
        sub_gridwidth = 20 # int(width/wid_grid) # 25
        new_length =len_grid * sub_gridlen 
        new_width =wid_grid *  sub_gridwidth

        reconstructed_topo  = self.real_elev.copy()  # to define the size 

        groundtruth_topo = self.real_elev.copy()
 
        if method == 1: 

            #inittopo_vec = (inittopo_vec/200 ) * self.inittopo_expertknow.flatten()  
            inittopo_vec =  self.inittopo_expertknow.flatten()  +  inittopo_vec  

        elif method ==2:

            inittopo_vec = ((inittopo_vec/200) * self.inittopo_expertknow.flatten()) + self.inittopo_expertknow.flatten()  

 


        scale_factor = np.reshape(inittopo_vec, (sub_gridlen, -1)   )#np.random.rand(len_grid,wid_grid) 


        v_ =   scale_factor  

        #v_ =  np.multiply(self.inittopo_expertknow.copy(), scale_factor.copy())   #+ x_

      
        for l in range(0,sub_gridlen-1):
            for w in range(0,sub_gridwidth-1): 
                for m in range(l * len_grid,(l+1) * len_grid):  
                    for n in range(w *  wid_grid, (w+1) * wid_grid):  
                        reconstructed_topo[m][n]  = reconstructed_topo[m][n] +  v_[l][w] 
 


        width = reconstructed_topo.shape[0]
        length = reconstructed_topo.shape[1]
 
        for l in range(0,sub_gridlen -1 ):  
            w = sub_gridwidth-1
            for m in range(l * len_grid,(l+1) * len_grid):  
                    for n in range(w *  wid_grid,  length):  
                        groundtruth_topo[m][n]   +=  v_[l][w] 

        for w in range(0,sub_gridwidth -1): 

            l = sub_gridlen-1  
            for m in range(l * len_grid,width):  
                    for n in range(w *  wid_grid, (w+1) * wid_grid):  
                        groundtruth_topo[m][n]   +=  v_[l][w]

 

        inside = reconstructed_topo[  0 : sub_gridlen-2 * len_grid,0:   (sub_gridwidth-2 *  wid_grid)  ] 

 
   

        for m in range(0 , inside.shape[0]):  
            for n in range(0 ,   inside.shape[1]):  
                    groundtruth_topo[m][n]   = inside[m][n] 
        #self.plot3d_plotly(reconstructed_topo, 'GTinitrecon_')
 
        groundtruth_topo = gaussian_filter(reconstructed_topo, sigma=1) # change sigma to higher values if needed 

 
        self.plot3d_plotly(reconstructed_topo, 'mean_')



        return groundtruth_topo

  
 
    def view_crosssection_uncertainity(self,  list_xslice, list_yslice):

        ymid = int(self.real_elev.shape[1]/2) 
        xmid = int(self.real_elev.shape[0]/2)

        #x_m = np.arange(20,80, 10)
        #y_m = np.arange(20,80, 5)
  
        
        # list_xslice = list_xslice[20:120,:]
        # list_yslice = list_yslice[20:100,:]
        self.real_elev_ = self.real_elev
        # self.real_elev_ = self.real_elev[20:100, 20:120]
        # self.init_elev = self.init_elev[20:100, 20:120]
        #for i in x_m:
            #xmid = i
            #ymid = i

        x_ymid_real = self.real_elev_[xmid, :] 
        y_xmid_real = self.real_elev_[:, ymid ]
        #x_ymid_init = self.init_elev[xmid, :]
        #y_xmid_init = self.init_elev[:, ymid]
        x_ymid_mean = list_xslice.mean(axis=1)
        y_xmid_mean = list_yslice.mean(axis=1)
    
        x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
        x_ymid_95th= np.percentile(list_xslice, 95, axis=1)

        y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
        y_xmid_95th= np.percentile(list_yslice, 95, axis=1)


        x = np.linspace(0, x_ymid_mean.size * self.resolu_factor, num=x_ymid_mean.size) 
        x_ = np.linspace(0, y_xmid_mean.size * self.resolu_factor, num=y_xmid_mean.size)

        #ax.set_xlim(-width,len(ind)+width)

        self.cross_section(x, x_ymid_mean, x_ymid_real,   x_ymid_5th, x_ymid_95th, 'x_ymid_cross_%s_%s' %(xmid,ymid))
        self.cross_section(x_, y_xmid_mean, y_xmid_real,   y_xmid_5th, y_xmid_95th, 'y_xmid_cross_%s_%s'%(xmid,ymid))

    def cross_section(self, x, pred, real,   lower, higher, fname):

        init =[]

        size = 15

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.plot(x, real, label='Ground Truth') 
        plt.plot(x, pred, label='Badlands Pred.')
        #plt.plot(x, init, label = 'Initial Topo')
        plt.grid(alpha=0.75)

        rmse_init = np.sqrt(np.sum(np.square(pred  -  real))  / real.size)   

        plt.fill_between(x, lower , higher, facecolor='g', alpha=0.2, label = 'Uncertainty')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='best') 
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

        plt.title("Topography  cross section   ", fontsize = size)
        plt.xlabel(' Distance (x 50 km)  ', fontsize = size)
        plt.ylabel(' Height (m)', fontsize = size)
        plt.tight_layout()
          
        plt.savefig(self.folder+'/'+fname+'.pdf')
        plt.clf()

        return rmse_init

    def get_synthetic_initopo(self):

        model = badlandsModel() 
        # Load the XmL input file
        model.load_xml(str(self.run_nb_str), self.xmlinput, muted=True) 
        #Update the initial topography
        #Use the coordinates from the original dem file
        xi=int(np.shape(model.recGrid.rectX)[0]/model.recGrid.nx)
        yi=int(np.shape(model.recGrid.rectY)[0]/model.recGrid.ny)
        #And put the demfile on a grid we can manipulate easily
        elev=np.reshape(model.recGrid.rectZ,(xi,yi))

        return elev

    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):
        path = self.folder +'/posterior/pos_parameters/' 
        x = [] # first get the size of the files

        files = os.listdir(path)
        for name in files: 
            dat = np.loadtxt(path+name)
            x.append(dat.shape[0])
        size_pos = min(x) 
        self.num_chains = len(x)


        self.NumSamples = int((self.num_chains * size_pos)/ self.num_chains)
        burnin =  int((self.NumSamples * self.burn_in)/self.num_chains)
        coverage = self.NumSamples - burnin

        pos_param = np.zeros((self.num_chains, self.NumSamples  , self.num_param))
        list_xslice = np.zeros((self.num_chains, self.NumSamples , self.real_elev.shape[1]))
        list_yslice = np.zeros((self.num_chains, self.NumSamples  , self.real_elev.shape[0]))
        likehood_rep = np.zeros((self.num_chains, self.NumSamples)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        #accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples )) 
        topo  = self.real_elev 

        edp_pts_time = self.real_erodep_pts.shape[1] *self.sim_interval.size

        erodep_pts = np.zeros(( self.num_chains, self.NumSamples  , edp_pts_time )) 
        combined_erodep = np.zeros((self.num_chains, self.NumSamples, self.real_erodep_pts.shape[1] ))
        timespan_erodep = np.zeros(( (self.NumSamples - burnin) * self.num_chains, self.real_erodep_pts.shape[1] ))
        rmse_elev = np.zeros((self.num_chains, self.NumSamples))
        rmse_erodep = np.zeros((self.num_chains, self.NumSamples))
 

        path = self.folder +'/posterior/pos_parameters/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            # print(dat.shape, pos_param.shape,  v, burnin, size_pos, coverage) 
            pos_param[v, :, :] = dat[ :pos_param.shape[1],:] 
            #print (dat)
            # print(v, name, ' is v')
            v = v +1
             


        posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)  
 

        path = self.folder +'/posterior/predicted_topo/x_slice/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            list_xslice[v, :, :] = dat[ : list_xslice.shape[1],: ] 
            v = v +1


        list_xslice = list_xslice[:, burnin:, :]

        xslice = list_xslice.transpose(2,0,1).reshape(self.real_elev.shape[1],-1) 

        path = self.folder +'/posterior/predicted_topo/y_slice/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            list_yslice[v, :, :] = dat[ : list_yslice.shape[1],: ] 
            v = v +1 

        list_yslice = list_yslice[:, burnin:, :] 
        yslice = list_yslice.transpose(2,0,1).reshape(self.real_elev.shape[0],-1) 


        path = self.folder +'/posterior/predicted_topo/sed/' 
        files = os.listdir(path)
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            erodep_pts[v, :, :] = dat[ : erodep_pts.shape[1],: ] 
            v = v +1 

        erodep_pts = erodep_pts[:, burnin:, :] 
        
        erodep_pts = erodep_pts.transpose(2,0,1).reshape(edp_pts_time,-1) 
        # print(erodep_pts.shape, ' ed   ***')
 

        path = self.folder +'/performance/lhood/' 
        files = os.listdir(path)

        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            likehood_rep[v, : ] = dat[ : likehood_rep.shape[1]] 
            v = v +1  

        #likehood_rep = likehood_rep[:, burnin: ] 

        path = self.folder +'/performance/accept/' 
        files = os.listdir(path)

        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            accept_list[v, : ] = dat[ : accept_list.shape[1]] 
            v = v +1 
        #accept_list = accept_list[:, burnin: ] 

        path = self.folder +'/performance/rmse_edep/' 
        files = os.listdir(path) 
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            rmse_erodep[v, : ] = dat[ : rmse_erodep.shape[1]] 
            v = v +1 
        rmse_erodep = rmse_erodep[:, burnin: ] 


        path = self.folder +'/performance/rmse_elev/'
        files = os.listdir(path)
 
        v = 0 
        for name in files: 
            dat = np.loadtxt(path+name) 
            rmse_elev[v, : ] = dat[ : rmse_elev.shape[1]] 
            v = v +1 
        rmse_elev = rmse_elev[:, burnin: ]

        likelihood_vec = likehood_rep 
        accept_list = accept_list 

        rmse_elev = rmse_elev.reshape(self.num_chains*(self.NumSamples -burnin ),1)
        rmse_erodep = rmse_erodep.reshape(self.num_chains*(self.NumSamples -burnin  ),1) 

        #print( ' .... need print file names --------------------------------------------')
        np.savetxt(self.folder + '/pos_param.txt', posterior.T) 
        np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')
        np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')
        #np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')
        
        return posterior, likelihood_vec, accept_list,   xslice, yslice, rmse_elev, rmse_erodep, erodep_pts

    def find_nearest(self, array,value): # just to find nearest value of a percentile (5th or 9th from pos likelihood)
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    def get_uncertainity(self, likehood_rep, pos_param ): 

        likelihood_pos = likehood_rep[:,1]

        a = np.percentile(likelihood_pos, 5)   
        lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
        b = np.percentile(likelihood_pos, 95) 
        lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  
        max_index = np.argmax(likelihood_pos) # find max of pos liklihood to get the max or optimal pos value  

        optimal_para = pos_param[:, max_index] 
        para_5thperc = pos_param[:, index_5th]
        para_95thperc = pos_param[:, index_95th] 

        return optimal_para, para_5thperc, para_95thperc

    def plot_figure(self, list, title): 

        list_points =  list
        fname = self.folder
        size = 15

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)
        plt.hist(list_points,  bins = 20, color='#0504aa',
                            alpha=0.7)   
        plt.title("Posterior distribution ", fontsize = size)
        plt.xlabel(' Parameter value  ', fontsize = size)
        plt.ylabel(' Frequency ', fontsize = size)
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_posterior.pdf')
        plt.clf()

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        listx = np.asarray(np.split(list_points,  self.num_chains ))
        plt.plot(listx.T)   

        plt.title("Parameter trace plot", fontsize = size)
        plt.xlabel(' Number of Samples  ', fontsize = size)
        plt.ylabel(' Parameter value ', fontsize = size)
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_trace.pdf')
        plt.clf()

    def plot_sed(self, list, title): 

        list_points =  list
        fname = self.folder
        size = 15

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)
        plt.hist(list_points,  bins = 20, color='#0504aa',
                            alpha=0.7)   
        plt.title("Sediment distribution ", fontsize = size)
        plt.xlabel(' Elevation (meters)  ', fontsize = size)
        plt.ylabel(' Frequency ', fontsize = size)
        plt.tight_layout()  
        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)
        plt.savefig(fname + '/sed_visual/' + title  + '_sed_distri.pdf')
        plt.clf()

    def heatmap_sed(self, sed_data, title): 
        size = 15
        # plt.imshow(sed_data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Sediment heatmap ", fontsize = size)
        plt.xlabel(' Northings  ', fontsize = size)
        plt.ylabel(' Eastings ', fontsize = size)
        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.savefig(self.folder+ '/sed_visual/' + title  + '_sed_heatmap.pdf')
        plt.clf()
 
 

    def visualize_sediments(self, sediment_timedic):
        # print(" sediments visualize .... ")
        sediment=sediment_timedic[self.simtime]
        # print(sediment, ' sediment grid .')

        length = sediment.shape[0]
        width = sediment.shape[1]
        len_grid = self.len_grid
        wid_grid = self.wid_grid
        sub_gridlen = 30 #int(length/len_grid)  # 25
        sub_gridwidth = 30 # int(width/wid_grid) # 25
        #new_length =len_grid * sub_gridlen 
        #new_width =wid_grid *  sub_gridwidth 
        sed = sediment.copy()
        grid = sediment 
        len_num = 4
        wid_num = 4
 
        len_grid = int(sediment.shape[0]/len_num)  # take care of left over
        wid_grid = int(sediment.shape[1]/wid_num)   # take care of left over
        # print(len_grid, wid_grid, ' len_grid, wid_grid ')
   
        i = 0

        sed_list = grid.flatten()
 
        # print(grid.shape, ' grid ')
        # print(sed_list, ' sed list ')

        # self.plot_sed(sed_list, 'region_x')
        # self.heatmap_sed(grid, 'map_x')


        #---------------------------------------
    def vis_badlands(self, folder):
        # Load the last time step
        file = folder+"/AUSP1306_output/h5"
        stepCounter = len(glob.glob1(folder+"/AUSP1306_output/xmf/","tin.time*"))-1
        print(stepCounter)
        # stepCounter = 50

        # Get the elevation, cumulative elevation change, flow discharge, and sea level 
        # tin,flow,sea = visu.loadStep(folder+"/AUSP1306_output",stepCounter)
        # visu.view1Step(folder+"/AUSP1306_output", tin, flow, sea, scaleZ=20, maxZ=2500, maxED=200, flowlines=False)

        timestep = 10
        strat = strata.stratalSection(file,1)
        strat.loadStratigraphy(stepCounter)
        strat.loadTIN(stepCounter)

        cs=np.zeros((2,2))
        cs[0,:] = [12,100]  # point 1
        cs[1,:] = [24,10]  # point 2
        # Interpolation parameters
        nbpts = 500  
        gfilt = 2  

        strat.plotSectionMap(title='Topography map', xlegend='Distance (m)', ylegend='Distance (m)', 
                     color=cmo.cm.delta, crange=[-2000,2000], cs=None, size=(6,6))
        strat.buildSection(xo = cs[0,0], yo = cs[0,1], xm = cs[1,0], ym = cs[1,1], pts = nbpts, gfilter = gfilt)
        strata.viewSection(width = 800, height = 500, cs = strat, 
            dnlay = 2, rangeX=[2000, 10000], rangeY=[-400,200],
            linesize = 0.5, title='Stratal stacking pattern coloured by time')
                # Specify the range of water depth for the depositional environments, see the table above
        depthID = [0, -25, -100, -200, -500]

        # Define colors for depositional environments, with number of colors equals to len(depthID) + 2
        colorDepoenvi = ['white','limegreen','darkkhaki','sandybrown','khaki','c','teal'] 
        # 'White' colors where either no deposition or deposited sediemnt thickness < 0.01 m.

        # Build an array of depositional environment ID (enviID)
        enviID = np.zeros((strat.nz, len(strat.dist)))
        enviID = strata.buildEnviID(cs = strat, depthID = depthID)
        strata.viewDepoenvi(folder+ "/AUSP1306_output",width = 8, height = 5, cs = strat, enviID = enviID, dnlay = 2, color = colorDepoenvi, 
                    rangeX=[2000, 12000], rangeY=[-500,100], savefig = 'Yes', figname = 'delta_strata_depoenvi')
        
        start_time = 0.  # the start time of the model run [a]
        disptime = 50000.  # the layer interval of the strata module [a]
        end_time = start_time + disptime * timestep  # the time of the loaded output [a]
        layertime = np.linspace(start_time,end_time,strat.nz)  # time of the layers

        # Plot Wheeler diagram
        # strata.viewWheeler(width = 7, height = 4, cs = strat, enviID = enviID, time = layertime, dnlay = 3, color = colorDepoenvi, 
                           # rangeX=[2000, 12000], rangeY = None, savefig = 'Yes', figname = 'delta_Wheeler_diagram')
        
        # Location of the core on the cross-section (m)
        posit = 7000

        # Plot the core
        strata.viewCore(width = 2, height = 5, cs = strat, enviID = enviID, posit = posit, time = layertime, 
                        color = colorDepoenvi, rangeX = None, rangeY = None, savefig = 'Yes', figname = 'delta_core')
        #---------------------------------------

    def run_badlands(self, input_vector, muted = False):
        #Runs a badlands model with the specified inputs
        rain_regiontime = self.rain_region * self.rain_time
        #Create a badlands model instance
        model = badlandsModel()

        #----------------------------------------------------------------
        # Load the XmL input file
        model.load_xml(str(self.run_nb_str), self.input, muted=muted)

        # prob = [0,1,2]
        # if  problem in prob: #problem==1 or problem==2 : # when you have initial topo (problem is global variable)
        #if problem == 0 or problem ==1 or problem ==2 or problem==3:
         #   init = False
        #else:
        init = True # when you need to estimate initial topo


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
            filename=problem_folder +'/inittopo_mean_demfile.csv' 

            elev_framex = np.vstack((model.recGrid.rectX,model.recGrid.rectY,inittopo_estimate.flatten()))
            np.savetxt(filename, elev_framex.T, fmt='%1.2f' ) 
            model.input.demfile=filename 
            model.build_mesh(model.input.demfile, verbose=False)

        # Adjust precipitation values based on given parameter
        #print(input_vector[0:rain_regiontime] )
        model.force.rainVal  = input_vector[0:rain_regiontime] 

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

        if problem != 1:
            model.slp_cr = input_vector[rain_regiontime+5]
            model.perc_dep = input_vector[rain_regiontime+6]
            model.input.criver = input_vector[rain_regiontime+7]
            model.input.elasticH = input_vector[rain_regiontime+8]
            model.input.diffnb = input_vector[rain_regiontime+9]
            model.input.diffprop = input_vector[rain_regiontime+10]

        #Check if it is the mountain problem
        if problem==10: # needs to be updated
            #Round the input vector 
            k=round(input_vector[rain_regiontime+5],1) #to closest 0.1  @Nathan we need to fix this

            #Load the current tectonic uplift parameters
            tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values
        
            #Adjust the parameters by our value k, and save them out
            newFile = "Examples/mountain/tect/uplift"+str(self.temperature)+"_"+str(k)+".csv"
            newtect = pandas.DataFrame(tectonicValues*k)
            newtect.to_csv(newFile,index=False,header=False)

            #Update the model uplift tectonic values
            model.input.tectFile[0]=newFile

        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()
        elev_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted=muted)

            elev, erodep = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))
            elev_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]
                elev_pts[count] = elev[val[0], val[1]]

            # print('Sim time: ', self.simtime  , "   Temperature: ", self.temperature)
            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts
            elev_pts_vec[self.simtime] = elev_pts
 
        return elev_vec, erodep_vec, erodep_pts_vec, elev_pts_vec

    def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

        if zmin == None:
            zmin =  zData.min()

        if zmax == None:
            zmax =  zData.max()

        tickvals= [0,50,75,-50]

        xx = (np.linspace(0, zData.shape[0]* self.resolu_factor, num=zData.shape[0]/10 )) 
        yy = (np.linspace(0, zData.shape[1] * self.resolu_factor, num=zData.shape[1]/10 )) 

        xx = np.around(xx, decimals=0)
        yy = np.around(yy, decimals=0)
        
        axislabelsize = 20

        data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        layout = Layout(  autosize=True, width=width, height=height,scene=Scene(
                    zaxis=ZAxis(title = 'Elev.   ', range=[zmin,zmax], autorange=False, nticks=5, gridcolor='rgb(255, 255, 255)',
                                gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    xaxis=XAxis(title = 'x-axis  ',  tickvals= xx,      gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    yaxis=YAxis(title = 'y-axis  ', tickvals= yy,    gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    bgcolor="rgb(244, 244, 248)"
                )
            )

        fig = Figure(data=data, layout=layout) 
        graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.html', validate=False)

        fname = self.folder + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.pdf' 
        elev_data = np.reshape(zData, zData.shape[0] * zData.shape[1] )   
        hist, bin_edges = np.histogram(elev_data, density=True)

        size = 15 
        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.hist(elev_data, bins='auto')  

        #plt.title("Topography")  
        plt.xlabel('Elevation (m)', fontsize = size)
        plt.ylabel('Frequency', fontsize = size)
        plt.grid(alpha=0.75)


        plt.tight_layout()  
        plt.savefig(fname )
        plt.clf()

    def plot_erodeposition(self, erodep_mean, erodep_std, groundtruth_erodep_pts, sim_interval, fname):
        ticksize = 13

        fig = plt.figure()
        ax = fig.add_subplot(111)
        index = np.arange(groundtruth_erodep_pts.size) 
        ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
        opacity = 0.8
        width = 0.35       # the width of the bars

        rects1 = ax.bar(index, erodep_mean, width,
                    color='blue',
                    yerr=erodep_std,
                    error_kw=dict(elinewidth=2,ecolor='red'))

        rects2 = ax.bar(index+width, groundtruth_erodep_pts, width, color='green', 
                    yerr=ground_erodepstd,
                    error_kw=dict(elinewidth=2,ecolor='red') )
     

        ax.set_ylabel('Height in meters', fontsize=ticksize)
        ax.set_xlabel('Location ID ', fontsize=ticksize)
        ax.set_title('Erosion/Deposition', fontsize=ticksize)
        
        ax.grid(alpha=0.75)
        ax.tick_params(labelsize=ticksize)

        plotlegend = ax.legend( (rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') )
        plt.tight_layout()
        plt.savefig( self.folder+'/sediment_plots/pos_erodep_'+str( sim_interval) +fname+'_.pdf')
        plt.clf()    




# class  above this line -------------------------------------------------------------------------------------------------------


'''def mean_sqerror(  pred_erodep, pred_elev,  real_elev,  real_erodep_pts):
        
        elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
        sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

        return elev + sed, sed'''

def mean_sqerror(  pred_erodep,   real_erodep_pts):
        
        #elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
        sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 
        return   sed

def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)




def main():

    random.seed(time.time()) 

    (problemfolder, xmlinput, simtime, resolu_factor, init_elev, groundtruth_elev, groundtruth_erodep,
    groundtruth_erodep_pts, groundtruth_elev_pts, res_summaryfile, inittopo_expertknow, len_grid, wid_grid, simtime, 
    resolu_factor, likelihood_sediment, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erodep_coords, inittopo_estimated, vec_parameters, minlimits_vec, maxlimits_vec) = problem_setup(problem)

    rain_timescale = rain_intervals  # to show climate change 
    '''rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale)
    rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale)
    minlimits_vec = np.append(rain_minlimits,minlimits_others)
    maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)
    vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec)''' #  draw intial values for each of the free parameters
    true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
    stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
    num_param = vec_parameters.size


    with open ("foldername.txt", "r") as f:
        fname = f.read().splitlines() 

    fname = fname[0].rstrip()
    run_nb_str = fname
    timer_start = time.time()
    sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    if simtime <= 0:
        sim_interval = sim_interval[::-1]
    print("Simulation time interval", sim_interval)

    res = results_visualisation(  vec_parameters, inittopo_expertknow, inittopo_estimated, rain_regiongrid, rain_timescale, len_grid,  wid_grid, num_chains, maxtemp, samples,swap_interval,fname, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, simtime, sim_interval, resolu_factor,  xmlinput,  run_nb_str, init_elev)
    pos_param, likehood_rep, accept_list, xslice, yslice, rmse_elev, rmse_erodep, erodep_pts, rmse_slice_init  = res.results_current()

    print('sucessfully sampled') 
    timer_end = time.time() 
    likelihood = likehood_rep # just plot proposed likelihood  

    plt.plot(likelihood.T)
    plt.savefig( fname+'/likelihood.pdf')
    plt.clf()

    size = 15 

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.plot(accept_list.T)
    plt.title("Replica Acceptance ", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Number Accepted ', fontsize = size)
    plt.tight_layout()
    plt.savefig( fname+'/accept_list.pdf' )
    plt.clf()

    #print(erodep_pts.shape, ' erodep_pts.shape')

    pred_erodep = np.zeros(( groundtruth_erodep_pts.shape[0], groundtruth_erodep_pts.shape[1] )) # just to get the right size

    for i in range(sim_interval.size): 

        begin = i * groundtruth_erodep_pts.shape[1] # number of points 
        end = begin + groundtruth_erodep_pts.shape[1] 

        pos_ed = erodep_pts[begin:end, :] 
        pos_ed = pos_ed.T 
        erodep_mean = pos_ed.mean(axis=0)  
        erodep_std = pos_ed.std(axis=0)  
        pred_erodep[i,:] = pos_ed.mean(axis=0)  


        # print(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], fname) 
        res.plot_erodeposition(erodep_mean[0:200:20], erodep_std[0:200:20], groundtruth_erodep_pts[i,0:200:20], sim_interval[i], 'first') 
        res.plot_erodeposition(erodep_mean[200:400:20], erodep_std[200:400:20], groundtruth_erodep_pts[i,200:400:20], sim_interval[i], 'second') 
        res.plot_erodeposition(erodep_mean[400:600:20], erodep_std[400:600:20], groundtruth_erodep_pts[i,400:600:20], sim_interval[i], 'third') 
         
    pred_elev = np.array([])
    rmse_sed= mean_sqerror(  pred_erodep,  groundtruth_erodep_pts)
    rmse = 0
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    size = 15

    ax.tick_params(labelsize=size)

    plt.legend(loc='upper right') 

    ax.boxplot(pos_param.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Posterior', fontsize=size) 
    plt.title("Boxplot of Posterior", fontsize=size) 
    plt.savefig(fname+'/badlands_pos.pdf')
    plt.clf()
    
    timer_end = time.time() 

    rmse_el = np.mean(rmse_elev[:])
    rmse_el_std = np.std(rmse_elev[:])
    rmse_el_min = np.amin(rmse_elev[:])
    rmse_er = np.mean(rmse_erodep[:])
    rmse_er_std = np.std(rmse_erodep[:])
    rmse_er_min = np.amin(rmse_erodep[:])

    time_total = (timer_end-timer_start)/60

    # print (rmse_el_min, ' minimum value ', np.where(rmse_elev == rmse_el_min), '@ this index')

    # print('pos_param',pos_param.size, pos_param.shape)

############################################################################################
    error_dict = {}
    for i,j in enumerate(rmse_elev):
        if j[0] != 0.0:
            # print ('\ni : ', i, '  j : ', j[0], '\n')
            error_dict[j[0]] = pos_param.T[i,:] 
        else:
            pass
            # print ('the error was 0.0')

    print('min error in dict',min(error_dict))    
    # print(' The parameters with min error are : ', error_dict[min(error_dict)], error_dict[min(error_dict)].shape )
    variables = error_dict[min(error_dict)]
    #print ('variables', variables, variables.shape)

    variables[:15] = [1.0, 1.0, 1.0, 1.0, 1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01]
    #print('variables', variables)
    pred_elev_opt, pred_erodep_opt, pred_erodep_pts_opt, pred_elev_pts_opt = res.run_badlands(error_dict[min(error_dict)], muted = True)

    
    for i in range(sim_interval.size):
        print(pred_erodep_opt[sim_interval[i]], ' pred_erodep_opt[i]')
        np.savetxt(fname+'/sediment_plots/erodep_' +str(i)+'_.txt', pred_erodep_opt[sim_interval[i]],  fmt='%1.2f' )
        np.savetxt(fname+'/sediment_plots/elev_' +str(i)+'_.txt', pred_elev_opt[sim_interval[i]],  fmt='%1.2f' )

        fnameplot = fname +  '/sediment_plots/sediment_map'+str(i) +'_.png' 
        # im = plt.imshow(pred_erodep_opt[sim_interval[i]], cmap='hot', interpolation='nearest')
        plt.colorbar(im) 
        plt.savefig(fnameplot)
        plt.clf()

        fnameplot = fname +  '/sediment_plots/elev_map'+str(i) +'_.png' 
        # im = plt.imshow(pred_elev_opt[sim_interval[i]], cmap='hot', interpolation='nearest')
        plt.colorbar(im) 
        plt.savefig(fnameplot)
        plt.clf()

        res.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev_opt[res.sim_interval[i]], title='Predicted Topography ', time_frame=res.sim_interval[i],  filename= 'optimal')



    res.full_crosssection(pred_elev_opt[0], groundtruth_elev) 



    # for i in range(res.sim_interval.size):
    #     res.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev_opt[res.sim_interval[i]], title='Predicted Topography ', time_frame=res.sim_interval[i],  filename= 'optimal')

############################################################################################
    resultingfile_db = open(problemfolder+res_summaryfile,'a+')  
    swap_perc = 0 # get value later -- to do
    accept_per = 0 

    allres =  np.asarray([ problem, num_chains, maxtemp, samples,swap_interval,  rmse_el, 
                        rmse_er, rmse_el_std, rmse_er_std, rmse_el_min, 
                        rmse_er_min, rmse, rmse_sed, swap_perc, accept_per,  time_total, rmse_slice_init , epsilon]) 
    print(allres, '  result')
        
    np.savetxt(resultingfile_db, allres , fmt='%1.4f',  newline=' ' )  
    np.savetxt(resultingfile_db, [fname] ,  fmt="%s", newline=' \n' ) 


    print("NumChains, problem, folder, time, RMSE_sed, RMSE,samples,swap,maxtemp,burn")
    print (num_chains, problemfolder, run_nb_str, (timer_end-timer_start)/60, rmse_sed, rmse,samples, swap_ratio,maxtemp,burn_in)

    dir_name = fname + '/posterior'
    fname_remove = fname +'/pos_param.txt'
    print(dir_name)

    res.vis_badlands(fname)
    res.visualize_sediments(pred_erodep_opt)    

    '''if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    if os.path.exists(fname_remove):  # comment if you wish to keep pos file
        os.remove(fname_remove) '''

    #stop()
if __name__ == "__main__": main()
