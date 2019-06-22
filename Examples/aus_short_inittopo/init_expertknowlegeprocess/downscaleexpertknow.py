
 
import scipy.ndimage as ndimage  
from scipy import stats  

import numpy as np
import seaborn as sns; sns.set()

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.style.use('ggplot')


expert_know = np.loadtxt('init_expert.txt')

print(expert_know)

#uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(expert_know)

fig = ax.get_figure()
fig.savefig("output.png")