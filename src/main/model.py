from routing import shipment, shipment_struct
from utility import rte2loc, rte2idx, pairwisesavings, savings
from cost import rte_tc
import numpy as np
if __name__ == '__main__':
    data = pd.read_csv()
    b = np.array([1,2,3])
    e = np.array([0,0,0])
    # e = np.asarray([10+i for i in range(10)])
    # temin = -np.Inf*np.ones(shape=b.shape)
    # temax = np.Inf*np.ones(shape=b.shape)
    # tbmin = -np.Inf*np.ones(shape=b.shape)
    # tbmax = np.Inf*np.ones(shape=b.shape)
    tbmin = np.array([8,12, 15])
    tbmax = np.array([11 ,14, 18])
    temax = np.Inf*np.ones(tbmin.shape)
    temin = -np.Inf*np.ones(tbmin.shape)

    sh = shipment_struct(b,e,tbmin,tbmax,temin,temax)
    C = np.asarray(np.meshgrid(np.arange(0,20,1),np.arange(0,20,1)))
    C = C[0]
    C = C + C.transpose()
    np.fill_diagonal(C,0)
    print(sh)
    C = np.array([[0 ,1 ,0, 0], [0, 0, 2 ,0], [0, 0, 0, 1],[ 1, 0, 0, 0]])
    rte = np.array([0,1,2,0,1,2])
    r = [np.asarray([0,1,2,3,2,1,3,0]),np.asarray([4,5,4,6,6,5])]
    # print(rte2loc(r,sh))
    # print(rte2idx(r))
    # r_numpy = np.asarray([0,1,0,1])
    # print(rte2loc(r_numpy,sh))
    # print(rte2idx(r_numpy))
    print(rte_tc(sh,rte,C))
    
    # rte_h = lambda x: rte_tc(sh,x,C)
    # IJS,S,TCij,Rij = pairwisesavings(rte_h,sh)
    # print(savings(rte_h,sh,IJS,dodisp=True))

