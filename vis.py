import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html

def main():
	a = np.loadtxt('Testing/final_elev.txt')
	b = np.loadtxt('Testing/initial_elev.txt')
	viewGrid(zData = a, filename = 'finallllll')
	viewGrid(zData = b, filename = 'initiallllll')
	print(a.shape)
	print(b.shape)


def viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

        if zmin == None:
            zmin =  zData.min()

        if zmax == None:
            zmax =  zData.max()

        tickvals= [0,50,75,-50]

        axislabelsize = 20

        data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        layout = Layout(  autosize=True, width=width, height=height,scene=Scene(
                    zaxis=ZAxis(title = 'Elev.   ', range=[zmin,zmax], autorange=False, nticks=5, gridcolor='rgb(255, 255, 255)',
                                gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    xaxis=XAxis(title = 'x-axis  ',   gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    yaxis=YAxis(title = 'y-axis  ',   gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    bgcolor="rgb(244, 244, 248)"
                )
            )

        fig = Figure(data=data, layout=layout) 
        graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= filename+ '_.html', validate=False)

        fname = filename+ '_.pdf' 
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


if __name__ == "__main__": main()