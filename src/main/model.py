from routing import shipment, shipment_struct
from utility import rte2loc, rte2idx
import numpy as np
if __name__ == '__main__':
    b = np.asarray([i for i in range(10)])
    e = np.asarray([10+i for i in range(10)])
    temin = b
    temax = e
    tbmin = b
    tbmax = e
    sh = shipment_struct(b,e,tbmin,tbmax,temin,temax)
    print(sh)
    r = [np.asarray([0,1,2,3,2,1,3,0]),np.asarray([4,5,4,6,6,5])]
    print(rte2loc(r,sh))
    r_numpy = np.asarray([0,1,2,3,2,1,3,0])
    print(type(rte2loc(r_numpy,sh)))