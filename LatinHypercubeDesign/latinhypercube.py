

from pyDOE import *
from scipy.stats.distributions import norm

#lhs(n, [samples, criterion, iterations])

lhd = lhs(2, samples=5)
lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here

lhs(4, samples=10, criterion='center')