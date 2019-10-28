
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time




x =  np.loadtxt('Examples/australia/results_3/recons_initialtopo/inittopo_smooth_10.txt')



y =  np.loadtxt('Examples/australia/results_3/recons_initialtopo/inittopo_smooth_17.txt')


z = x-y

xxx'

print(z)

z = np.random.rand(3,2)


print(z)

im = plt.imshow(z, cmap='hot', interpolation='nearest')
plt.show()
plt.savefig('fnameplot.png')
plt.clf()
