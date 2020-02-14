##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set usefull functions used to analyse the erosion and sedimentation evolution from Badlands outputs.
"""

import os
import math
import h5py
import errno
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
from scipy.interpolate import RectBivariateSpline

from pylab import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def viewData(x0 = None, y0 = None, width = 800, height = 400, linesize = 3, color = '#6666FF',
             xlegend = 'xaxis', ylegend = 'yaxis', title = 'view data'):
    """
    Plot multiple data on a graph.
    Parameters
    ----------
    variable: x0, y0
        Data for plot
    variable: width, height
        Figure width and height.
    variable: linesize
        Requested size for the line.
    variable: color

    variable: xlegend
        Legend of the x axis.
    variable: ylegend
        Legend of the y axis.
    variable: title
        Title of the graph.
    """
    trace = Scatter(
        x=x0,
        y=y0,
        mode='lines',
        line=dict(
            shape='line',
            color = color,
            width = linesize
        ),
        fill=None
    )

    layout = dict(
            title=title,
            font=dict(size=10),
            width=width,
            height=height,
            showlegend = False,
            xaxis=dict(title=xlegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks'),
            yaxis=dict(title=ylegend,
                       ticks='outside',
                       zeroline=False,
                       showline=True,
                       mirror='ticks')
            )

    fig = Figure(data=[trace], layout=layout)
    plotly.offline.iplot(fig)

    return

def getDepositedVolume(data=None, time=None, bbox = None):
    """
    Get the volume deposited in a specific region

    Parameters
    ----------
    variable: data
        Data to calculate
    variable: time
        Model duration for the given time step
    variable: bbox
        Extent of the box where computation is performed
    """

    cumDepositedVolume = np.zeros(len(data))
    rateDepositedVolume = np.zeros(len(data))

    timeInterval = [time[0]+0.1]
    for k in range(1,len(time)):
        timeInterval.append(time[k]-time[k-1])

    for i in range(len(data)):
        posz = np.copy(data[i].cumchange)
        if bbox is None:
            posz[posz<0.] = 0.
            volsed = np.sum(posz)*(data[i].dx*data[i].dx)
        else:
            r1,c1 = np.where(np.logical_and(data[i].x>=bbox[0],data[i].x<=bbox[2]))
            r2,c2 = np.where(np.logical_and(data[i].y>=bbox[1],data[i].y<=bbox[3]))
            lposz = posz[r2.min():r2.max(),c1.min():c1.max()]
            lposz[lposz<0.] = 0.
            volsed = np.sum(lposz)*(data[i].dx*data[i].dx)

        cumDepositedVolume[i] = volsed
        rateDepositedVolume[i] = (volsed/timeInterval[i])

    return cumDepositedVolume, rateDepositedVolume
    
def getErodedVolume(data=None, time=None, bbox = None):
    """
    Get the volume eroded in a specific region

    Parameters
    ----------
    variable: data
        Data to calculate
    variable: time
        Time of the multiplt outputs
    variable: bbox
        Extent of the box where computation is performed
    """

    cumErodedVolume = np.zeros(len(data))
    rateErodedVolume = np.zeros(len(data))

    timeInterval = [time[0]+0.1]
    for k in range(1,len(time)):
        timeInterval.append(time[k]-time[k-1])

    for i in range(len(data)):
        posz = np.copy(data[i].cumchange)
        if bbox is None:
            posz[posz>0.] = 0.
            volsed = np.sum(posz)*(data[i].dx*data[i].dx)
        else:
            r1,c1 = np.where(np.logical_and(data[i].x>=bbox[0],data[i].x<=bbox[2]))
            r2,c2 = np.where(np.logical_and(data[i].y>=bbox[1],data[i].y<=bbox[3]))
            lposz = posz[r2.min():r2.max(),c1.min():c1.max()]
            lposz[lposz>0.] = 0.
            volsed = np.sum(lposz)*(data[i].dx*data[i].dx)

        cumErodedVolume[i] = volsed
        rateErodedVolume[i] = (volsed/timeInterval[i])

    return cumErodedVolume, rateErodedVolume

class SedEro:
    """
    Class for analysing morphometrics from Badlands outputs.
    """

    def __init__(self, folder=None, ncpus=1, bbox=None):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation. 

        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        variable: ncpus
            Number of CPUs used to run the simulation.
        variable: bbox
            Bounding box extent SW corner and NE corner.

        """

        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')

        self.ncpus = ncpus
        self.x = None
        self.y = None
        self.z = None
        self.discharge = None
        self.logdischarge = None
        self.cumchange = None
        self.dx = None
        self.nx = None
        self.ny = None
        self.dx = None
        self.bbox = bbox

        return

    def loadHDF5(self, timestep=0):
        """
        Read the HDF5 file for a given time step.

        Parameters
        ----------
        variable : timestep
            Time step to load.

        """

        for i in range(0, self.ncpus):
            df = h5py.File('%s/tin.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
            coords = np.array((df['/coords']))
            cumdiff = np.array((df['/cumdiff']))
            discharge = np.array((df['/discharge']))
            if i == 0:
                x, y, z = np.hsplit(coords, 3)
                c = cumdiff
                d = discharge
            else:
                c = np.append(c, cumdiff)
                d = np.append(d, discharge)
                x = np.append(x, coords[:,0])
                y = np.append(y, coords[:,1])
                z = np.append(z, coords[:,2])

        self.dx = x[1]-x[0]
        if self.bbox == None:
            self.nx = int((x.max() - x.min())/self.dx+1)
            self.ny = int((y.max() - y.min())/self.dx+1)
            self.x = np.linspace(x.min(), x.max(), self.nx)
            self.y = np.linspace(y.min(), y.max(), self.ny)
            self.bbox = np.zeros(4,dtype=float)
            self.bbox[0] = x.min()
            self.bbox[1] = y.min()
            self.bbox[2] = x.max()
            self.bbox[3] = y.max()
        else:
            if self.bbox[0] < x.min():
                self.bbox[0] = x.min()
            if self.bbox[2] > x.max():
                self.bbox[2] = x.max()
            if self.bbox[1] < y.min():
                self.bbox[1] = y.min()
            if self.bbox[3] > y.max():
                self.bbox[3] = y.max()
            self.nx = int((self.bbox[2] - self.bbox[0])/self.dx+1)
            self.ny = int((self.bbox[3] - self.bbox[1])/self.dx+1)
            self.x = np.linspace(self.bbox[0], self.bbox[2], self.nx)
            self.y = np.linspace(self.bbox[1], self.bbox[3], self.ny)

        self.x, self.y = np.meshgrid(self.x, self.y)
        xyi = np.dstack([self.x.flatten(), self.y.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        z_vals = z[indices][:,:,0]
        d_vals = d[indices][:,:,0]
        c_vals = c[indices][:,:,0]
        
        zi = np.zeros(len(xyi))
        di = np.zeros(len(xyi))
        ci = np.zeros(len(xyi))
        onIDs = np.where(distances[:,0] > 0)[0]
        zi[onIDs] = np.average(z_vals[onIDs,:],weights=(1./distances[onIDs,:]), axis=1)
        di[onIDs] = np.average(d_vals[onIDs,:],weights=(1./distances[onIDs,:]), axis=1)
        ci[onIDs] = np.average(c_vals[onIDs,:],weights=(1./distances[onIDs,:]), axis=1)

        onIDs = np.where(distances[:,0] == 0)[0]
        
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0],0]
            di[onIDs] = d[indices[onIDs,0],0]
            ci[onIDs] = c[indices[onIDs,0],0]

        self.z = np.reshape(zi,(self.ny,self.nx))
        self.discharge = np.reshape(di,(self.ny,self.nx))
        self.cumchange = np.reshape(ci,(self.ny,self.nx))

        logdis = self.discharge
        IDs = np.where(logdis<1.)
        logdis[IDs] = 1.
        self.logdischarge = logdis

        return
    
    def plotdataSet(self, data=None, paramter=None, title='Topography map', xlegend=None, ylegend=None, color=None,
                    colorcs=None, crange=None, cs=None, ctr='k',size=(8,8)):
        """
        Plot the elevation and the cumulative elevation change

        Parameters
        ----------
        variable: data
            Dataset to plot 
        variable: paramter
            Dataset to plot, either 'elevation' or 'cumdiff'
        variable: title
            Title of the plot
        variable: color
            Colormap of the dataset            
        variable: colorcs
            Color of the cross-section
        variable: crange
            Range of values for the dataset
        variable: cs
            Defined cross-section
        variable: size
            Figure size
        """

        rcParams['figure.figsize'] = size
        ax=plt.gca()
        
        if paramter=='elevation':
            im = ax.imshow(np.flipud(data.z),interpolation='nearest',cmap=color,
                               vmin=crange[0], vmax=crange[1], extent=[self.bbox[0], self.bbox[2], self.bbox[1],self.bbox[3]])

            plt.contour(data.x, data.y, data.z, (0,), colors=ctr, linewidths=2)
            plt.xlabel(xlegend)
            plt.ylabel(ylegend)
        
        if paramter=='cumdiff':
            im = ax.imshow(np.flipud(data.cumchange),interpolation='nearest',cmap=color,
                               vmin=crange[0], vmax=crange[1], extent=[self.bbox[0], self.bbox[2], self.bbox[1],self.bbox[3]])

            plt.contour(data.x, data.y, data.cumchange, (0,), colors=ctr, linewidths=2)
            plt.xlabel(xlegend)
            plt.ylabel(ylegend)

        if cs is not None:
            plt.plot(cs[:,0],cs[:,1], '-x', color=colorcs, markersize=4)
        plt.title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        plt.colorbar(im,cax=cax)
        plt.show()
        plt.close()

        return
    
    def sederoRate(self, width = 6, height = 6, data = None, Time = None, bbox = None, color=None, crange=None, folder=None):
    
        """
        Plot the map of erosion and sedimentation rate through time

        Parameters
        ----------
        variable: data
            Dataset to plot 
        variable: Time
            Time of the dataset
        variable: color
            Colormap of the dataset            
        variable: crange
            Range of values for the dataset
        """
        
        if len(data)<2:
            print('This function works when multiple outputs are loaded!')
        else:
            for i in range(len(data)-1):
                fig = plt.figure(figsize = (width,height))
                plt.rc("font", size=10)
                # 
                ax = plt.gca()
                if bbox is None:
                    im = ax.imshow( np.flipud((data[i+1].cumchange-data[i].cumchange)/(Time[i+1]-Time[i])),
                                    interpolation='nearest',cmap=color,vmin=crange[0], vmax=crange[1], 
                                    extent=[self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]])
                else:
                    im = ax.imshow( np.flipud((data[i+1].cumchange-data[i].cumchange)/(Time[i+1]-Time[i])),
                                    interpolation='nearest',cmap=color,vmin=crange[0], vmax=crange[1], 
                                    extent=[bbox[0], bbox[2], bbox[1], bbox[3]])
                plt.title('From %.2f yr to %.2f yr [m/yr]'%(Time[i],Time[i+1]))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.xlabel('Distance (m)')
                plt.ylabel('Distance (m)')

                fig.savefig("%s/sederoMap%s.pdf"%(folder,str(i).zfill(2)), dpi=400)
                plt.close()

        return

    def _assignBCs(self):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.

        """

        self.Zbc = np.zeros((self.ny + 2, self.nx + 2))
        self.Zbc[1:-1,1:-1] = self.z

        # Assign boundary conditions - sides
        self.Zbc[0, 1:-1] = self.z[0, :]
        self.Zbc[-1, 1:-1] = self.z[-1, :]
        self.Zbc[1:-1, 0] = self.z[:, 0]
        self.Zbc[1:-1, -1] = self.z[:,-1]

        # Assign boundary conditions - corners
        self.Zbc[0, 0] = self.z[0, 0]
        self.Zbc[0, -1] = self.z[0, -1]
        self.Zbc[-1, 0] = self.z[-1, 0]
        self.Zbc[-1, -1] = self.z[-1, 0]

        return

    def _cross_section(self, xo, yo, xm, ym, pts):
        """
        Compute cross section coordinates.
        """

        if xm == xo:
            ysec = np.linspace(yo, ym, pts)
            xsec = np.zeros(pts)
            xsec.fill(xo)
        elif ym == yo:
            xsec = np.linspace(xo, xm, pts)
            ysec = np.zeros(pts)
            ysec.fill(yo)
        else:
            a = (ym-yo)/(xm-xo)
            b = yo - a * xo
            xsec = np.linspace(xo, xm, pts)
            ysec = a * xsec + b

        return xsec,ysec


    def viewSection(self, xo = None, yo = None, xm = None, ym = None, pts = 100, vData = None,
                    width = 800, height = 400, color = 'green', linesize = 3,
                    markersize = 5, title = 'Cross section'):
        """
        Extract a slice from the 3D data set and plot required data on a graph.

        Parameters
        ----------

        variable: xo, yo
            Lower X,Y coordinates of the cross-section

        variable: xm, ym
            Upper X,Y coordinates of the cross-section

        variable: pts
            Number of points to discretise the cross-section

        variable: vData
            Dataset to plot.

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: color
            Color scale.

        variable: linesize, markersize
            Requested size for the line and markers.

        variable: title
            Title of the graph.
        """

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        rect_B_spline = RectBivariateSpline(self.y[:,0], self.x[0,:], vData)
        datasec = rect_B_spline.ev(ysec, xsec)
        dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)

        data = Data([
           Scatter(
                x=dist,
                y=datasec,
                mode='lines+markers',
                name="'spline'",
                line=dict(
                    shape='spline',
                    color = color,
                    width = linesize

                ),
                marker = dict(
                    symbol='circle',
                    size = markersize,
                    color = 'white',
                    line = dict(
                        width = 1,
                        color = 'black'
                    )
                )
            )
            ])
        layout = dict(
            title=title,
            width=width,
            height=height
            )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return

    def extractSection(self, xo = None, yo = None, xm = None, ym = None, pts = 100, vData = None,
                    view = True, width = 800, height = 400, color = 'green', linesize = 3,
                    markersize = 5, title = 'Cross section'):
        """
        Extract a slice from the 3D data set and plot required data on a graph.

        Parameters
        ----------

        variable: xo, yo
            Lower X,Y coordinates of the cross-section

        variable: xm, ym
            Upper X,Y coordinates of the cross-section

        variable: pts
            Number of points to discretise the cross-section

        variable: vData
            Dataset to plot.

        variable: view
            Show the section plot.

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: color
            Color scale.

        variable: linesize, markersize
            Requested size for the line and markers.

        variable: title
            Title of the graph.

        Return:

        variable: dist, datasec
            X, Y values for the profile

        """

        if xm > self.x.max():
            xm = self.x.max()

        if ym > self.y.max():
            ym = self.y.max()

        if xo < self.x.min():
            xo = self.x.min()

        if yo < self.y.min():
            yo = self.y.min()

        xsec, ysec = self._cross_section(xo, yo, xm, ym, pts)
        rect_B_spline = RectBivariateSpline(self.y[:,0], self.x[0,:], vData)
        datasec = rect_B_spline.ev(ysec, xsec)
        dist = np.sqrt(( xsec - xo )**2 + ( ysec - yo )**2)

        if view:
            data = Data([
               Scatter(
                    x=dist,
                    y=datasec,
                    mode='lines+markers',
                    name="'spline'",
                    line=dict(
                        shape='spline',
                        color = color,
                        width = linesize

                    ),
                    marker = dict(
                        symbol='circle',
                        size = markersize,
                        color = 'white',
                        line = dict(
                            width = 1,
                            color = 'black'
                        )
                    )
                )
                ])
            layout = dict(
                title=title,
                width=width,
                height=height
                )

            fig = Figure(data=data, layout=layout)
            plotly.offline.iplot(fig)

        return dist, datasec

    def profile_mean(self,a):
        return sum(a) / len(a)

    def profile_min(self,a):
        return min(a)

    def profile_max(self,a):
        return max(a)

    def statProfiles(self, pData = None, pDist = None, width = 800, height = 400, color = 'green', linesize = 2,
                    title = 'Section Min, Mean, Max '):
        """
        Plot profile mean, max and min.

        Parameters
        ----------

        variable: pData
            Dataset to plot along Y axis.

        variable: pDist
            Dataset to plot along X axis.

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: color
            Color scale.

        variable: linesize, markersize
            Requested size for the line and markers.

        variable: title
            Title of the graph.

        Return:

        variable: minZ, meanZ, maxZ
            Y values for the profile (minZ, meanZ, maxZ)
        """

        meanZ = map(self.profile_mean, zip(*pData))
        minZ = map(self.profile_min, zip(*pData))
        maxZ = map(self.profile_max, zip(*pData))

        trace0 = Scatter(
            x=pDist,
            y=maxZ,
            mode='lines',
            line=dict(
                shape='spline',
                width = 0.5,
                color = 'rgb(0, 0, 0)'
            ),
            name='max'
        )

        trace1 = Scatter(
            x=pDist,
            y=minZ,
            mode='lines',
            line=dict(
                shape='spline',
                width = 0.5,
                color = 'rgb(0, 0, 0)'
            ),
            opacity=0.5,
            fill='tonexty',
            fillcolor=color,
            name='min'
        )

        trace2 = Scatter(
            x=pDist,
            y=meanZ,
            mode='lines',
            line=dict(
                shape='spline',
                width = linesize,
                color = 'rgb(0, 0, 0)'
            ),
            name='mean'
        )
        data = [trace0,trace1,trace2]

        layout = dict(
            title=title,
            width=width,
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return minZ, meanZ, maxZ

    def timeProfiles(self, pData = None, pDist = None, width = 800, height = 400, linesize = 2,
                    title = 'Profile evolution with time'):
        """
        Plot profile mean, max and min.

        Parameters
        ----------

        variable: pData
            Dataset to plot along Y axis.

        variable: pDist
            Dataset to plot along X axis.

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: color
            Color scale.

        variable: linesize, markersize
            Requested size for the line and markers.

        variable: title
            Title of the graph.

        Return:

        variable: minZ, meanZ, maxZ
            Y values for the profile (minZ, meanZ, maxZ)
        """

        trace = {}
        data = []

        for i in range(0,len(pData)):
            trace[i] = Scatter(
                x=pDist,
                y=pData[i,:],
                mode='lines',
                line=dict(
                    shape='spline',
                    width = linesize,
                    #color = color
                ),
            )
            data.append(trace[i])

        layout = dict(
            title=title,
            width=width,
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def viewGrid(self, width = 800, height = 800,
                 Dmin = None, Dmax = None, color = None, reverse=False,
                 Data = None, title='Grid'):
        """
        Use Plotly library to visualise a dataset in 2D.

        Parameters
        ----------

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: Dmin
            Colorbar minimal value.

        variable: Dmax
            Colorbar maximal value.

        variable: color
            Color scale.

        variable: reverse
            Reverse color scale.

        variable: Data
            Dataset to plot.

        variable: title
            Title of the graph.
        """

        if color == None:
            color = 'Picnic'

        data = [
            Heatmap(
                z = Data, colorscale = color,\
                    zmin = Dmin, zmax = Dmax,
                    reversescale=reverse
                )
            ]
        dy = self.bbox[3]-self.bbox[1]
        dx = self.bbox[2]-self.bbox[0]
        if dx>=dy:
            dr = 0.5 * (dx-dy)
            rangeX = [self.bbox[0],self.bbox[2]]
            rangeY = [self.bbox[1]-dr,self.bbox[3]+dr]
        else:
            dr = 0.5 * (dy-dx)
            rangeX = [self.bbox[0]-dr,self.bbox[2]+dr]
            rangeY = [self.bbox[1],self.bbox[3]]

        layout = Layout(
            title=title,
            autosize=True,
            width=width,
            height=height,
            scene=Scene(
                xaxis=XAxis(autorange=False, range=rangeX, nticks=10, \
                            gridcolor='rgb(255, 255, 255)', \
                            gridwidth=2,zerolinecolor='rgb(255, 255, 255)', \
                            zerolinewidth=2),
                yaxis=YAxis(autorange=False, range=rangeY, nticks=10, \
                            gridcolor='rgb(255, 255, 255)', \
                            gridwidth=2,zerolinecolor='rgb(255, 255, 255)', \
                            zerolinewidth=2),
                bgcolor="rgb(244, 244, 248)"
            )
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return

    def viewScatter3D(self, width = 800, height = 800, colors='Viridis',
                 dataX = None, dataY = None, dataZ = None, title='Scatter plot'):
        """
        Use Plotly library to visualise a dataset in 3D.

        Parameters
        ----------

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: colors
            Color scale.

        variable: dataX
            Data for X-axis.

        variable: dataY
            Data for Y-axis.

        variable: dataZ
            Data for Z-axis.

        variable: title
            Title of the graph.
        """

        #trace = {}
        data = []
        #A = np.asarray(dataX) / np.asarray(dataY)
        #A[np.isnan(A)] = 0
        #A[np.isinf(A)] = max(A[A<1000])+1
        trace = Scatter3d(
           x=dataX,
           y=dataY,
           z=dataZ,
           mode='markers',
           marker=dict(
                    size=8,
                    #color=A,
                    #colorscale=colors,
                    opacity=0.8
                )
           )
        data.append(trace)

        layout = dict(
            title=title,
            width=width,
            height=height,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            scene=Scene(
                xaxis=XAxis(title='dip'),
                yaxis=YAxis(title='slip'),
                zaxis=ZAxis(title='sed')
            )
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return

    def viewScatter(self, width = 800, height = 800,
                 dataX = None, dataY = None, title='Scatter plot'):
        """
        Use Plotly library to visualise a dataset in 2D.

        Parameters
        ----------

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: dataX
            Data for X-axis.

        variable: dataY
            Data for Y-axis.

        variable: title
            Title of the graph.
        """

        #trace = {}
        data = []

        trace = Scatter(
           x=dataX,
           y=dataY,
           mode='markers',
           )
        data.append(trace)

        layout = dict(
            title=title,
            width=width,
            height=height
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return

    def viewSurf(self, width = 800, height = 800,
                 zmin = None, zmax = None, color = None, reverse=False,
                 vData = None, subsample = 1, title='Surface'):
        """
        Use Plotly library to visualise a dataset over a surface in 3D.

        Parameters
        ----------

        variable: width
            Figure width.

        variable: height
            Figure height.

        variable: zmin
            Minimal Z-axis value.

        variable: zmax
            Maximal Z-axis value.

        variable: color
            Color scale.

        variable: reverse
            Reverse color scale.

        variable: vData
            Dataset to plot.

        variable: subsample
            Subsampling data everythin nth value.

        variable: title
            Title of the graph.
        """

        if color == None:
            color = 'YIGnBu'

        if zmin == None:
            zmin = vData.min()

        if zmax == None:
            zmax = vData.max()

        data = Data([
                Surface(
                    x = self.x[::subsample,::subsample],
                    y = self.y[::subsample,::subsample],
                    z = vData[::subsample,::subsample],
                    colorscale = color,
                    reversescale=reverse
                )
            ])

        dy = self.bbox[3]-self.bbox[1]
        dx = self.bbox[2]-self.bbox[0]
        if dx>=dy:
            dr = 0.5 * (dx-dy)
            rangeX = [self.bbox[0],self.bbox[2]]
            rangeY = [self.bbox[1]-dr,self.bbox[3]+dr]
        else:
            dr = 0.5 * (dy-dx)
            rangeX = [self.bbox[0]-dr,self.bbox[2]+dr]
            rangeY = [self.bbox[1],self.bbox[3]]

        layout = Layout(
            title=title,
            autosize=True,
            width=width,
            height=height,
            scene=Scene(
                zaxis=ZAxis(range=[zmin, zmax], \
                            autorange=False,nticks=10, \
                            gridcolor='rgb(255, 255, 255)', \
                            gridwidth=2,zerolinecolor='rgb(255, 255, 255)', \
                            zerolinewidth=2),

                xaxis=XAxis(autorange=False, range=rangeX, \
                            nticks=10, gridcolor='rgb(255, 255, 255)', \
                            gridwidth=2,zerolinecolor='rgb(255, 255, 255)', \
                            zerolinewidth=2),

                yaxis=YAxis(autorange=False, range=rangeY, nticks=10, \
                            gridcolor='rgb(255, 255, 255)', \
                            gridwidth=2,zerolinecolor='rgb(255, 255, 255)', \
                            zerolinewidth=2),

                bgcolor="rgb(244, 244, 248)"
            )
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        return
    
    
