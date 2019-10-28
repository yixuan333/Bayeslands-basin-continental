
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time




x =  np.loadtxt('Examples/australia/results_8/recons_initialtopo/inittopo_smooth_12.txt')



y =  np.loadtxt('Examples/australia/results_8/recons_initialtopo/inittopo_smooth_18.txt')


z = x-y
 

print(z)

#z = np.random.rand(3,2)


print(z)

im = plt.imshow(x, cmap='hot', interpolation='nearest')
plt.colorbar(im)
plt.savefig('fnameplotx.png')
plt.clf()


im = plt.imshow(y, cmap='hot', interpolation='nearest')
plt.colorbar(im)
plt.savefig('fnameploty.png')
plt.clf()


im = plt.imshow(z, cmap='hot', interpolation='nearest')
plt.colorbar(im)
plt.savefig('fnameplotz.png')
plt.clf()


