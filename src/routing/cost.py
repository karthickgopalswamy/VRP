from routing import shipment_struct, driver
from utility import rte2idx, rte2loc, isorigin
from typing import List,Union
import pandas as pd
import numpy as np

def scanTW(t,tLU,b,e):
    #SCANTW Single location sequence time window scan.

    tol = 1e-8
    n = b.shape[0]

    s = b[1] + tLU[1]
    for i in range(1,n): #Forward scan to determine earliest finish time
        bi = b[i] + tLU[i]
        s = s + t[i] + tLU[i]
        if s < bi - tol:
            s = bi
        elif s > e[i] + tol:
            TC = np.Inf 
            s = np.NaN
            w = np.NaN 
            return (TC,s,w)
    f = s

    s = f - tLU[n-1]
    for i in range(n-2,-1,-1): #Reverse scan to determine latest start time for the
        # earliest finish
        s = s - t[i+1] - tLU[i]
        ei = e[i] - tLU[i]
        if (s > ei + tol):
            s = ei

    TC = f - s
    if np.isnan(TC):
        TC = sum(t) + sum(tLU)  


    s = np.hstack([s ,np.zeros(shape=n-1)])
    w = np.zeros(shape=n)
    for i in range(1,n):  # Second forward scan to delay waits as much as possible
        # to the end of the loc seq in case unexpected events occur
        s[i] = s[i-1] + tLU[i-1] + t[i]
        bi = b[i]
        if not(s[i] + tol >= bi and s[i] + tLU[i] - tol <= e[i]):
            w[i] = s[i]
            s[i] = bi
            w[i] = s[i] - w[i]
    return (TC,s,w)


def rte_tc(sh:shipment_struct, rte:List[np.array],C:np.ndarray,tr:Union[None,driver]=None):
    """ Route Total Cost 
    Params:
        sh: Shipment Structure with shipments inside
        rte: m-element route List
        C: n x n array of distance/time between all the points (pickups + drops)
        tr: Optional, driver class with details about driver and constraints
    Returns:
        TC[i] = total cost of route i = sum of C and, if specified, tL and tU
                = Inf if route i is infeasible
        XFlg[i] = exitflag for route i
                =  1, if route is feasible
                = -1, if degenerate location vector for route (see RTE2LOC)
                = -2, if infeasible due to excess weight
                = -3, if infeasible due to excess cube
                = -4, if infeasible due to exceeding max TC (TC(i) > trmaxTC)
                = -5, if infeasible due to time window violation
        out = m-element pandas Datframe  with columns
        outiloc[i] = output series with elments:
                Rte     = route, with 0 at begin/end if truck locations provided
                Loc     = location sequence
                Cost    = cost (drive timespan) from location j-1 to j,
                        Cost(j) = C(j-1,j) and Cost(1) = 0
                Arrive  = time of arrival
                Wait    = wait timespan if arrival prior to beginning of window
                TWmin   = beginning of time window (earliest time)
                Start   = time loading/unloading started (starting time for
                        route is Start(1))
                LU      = loading/unloading timespan
                Depart  = time of departure (finishing time is Depart(end))
                TWmax   = end of time window (latest time)
                Total   = total timespan from departing loc j-1 to depart loc j
                        (= drive + wait + loading/unloading timespan)
    """
    doTW = True if isinstance(sh.tbmin,np.ndarray) else False
    domaxTC = True if tr and tr.maxTC else False
    doLU = False
    if doTW and np.any((sh.tbmin > sh.tbmax) | (sh.temin > sh.temax)):
        raise Exception('Min shipment time window exceeds max')

    rte = [np.array(rte)] if not isinstance(rte,List) or isinstance(rte[0],int) else rte

    TC = np.zeros(shape=(len(rte),1),dtype=np.float64)
    Xflg = np.ones(shape=(len(rte),1),dtype=np.int64)

    if doTW:
        cols = ['Rte','Loc','Cost','Arrive','Wait','TWmin','Start','LU','Depart','TWmax','Total']
    else:
        cols = ['Rte','Loc','Cost','Total']

    out = pd.DataFrame(columns=cols,index=range(len(rte)))

    for i in range(len(rte)):
        loc = rte2loc(rte[i],sh)
        if any(map(lambda x: np.isnan(x) , loc)):
            Xflg[i] = -1

        if Xflg[i] < 0:
            TC[i] = np.Inf
            continue
      
        idx = np.nonzero(np.hstack((1,np.diff(loc))))[0]
        c = loccost(loc[idx],C)
        TC[i] = sum(c)
        t = np.zeros(loc.shape)
        t[idx[1:]] = c

        if domaxTC and TC[i] > tr.maxTC:  # Max TC Feasibility
            Xflg[i] = -4 
            TC[i] = np.Inf

        if Xflg[i] > 0 and doTW:   # Time Window Feasibility
            if not doLU: tLU = np.zeros(shape=loc.shape)
            isL = isorigin(rte[i])
            tmin = np.zeros(isL.shape[0])
            tmax = tmin.copy()
            tmin[isL] = sh.tbmin[rte[i][isL]]
            tmin[np.invert(isL)] = sh.temin[rte[i][np.invert(isL)]]
            tmax[isL] = sh.tbmax[rte[i][isL]]
            tmax[np.invert(isL)] = sh.temax[rte[i][np.invert(isL)]]
            TC[i],s,w= scanTW(t,tLU,tmin,tmax)
            if np.isinf(TC[i]): Xflg[i] = -5 

        if Xflg[i] > 0:  # Output Structure
            out.iloc[i].Rte = rte[i]
            out.iloc[i].Loc = loc
            out.iloc[i].Cost = t
            out.iloc[i].Total = t
            if doLU or doTW:
                out.iloc[i].Total = t + tLU
                out.iloc[i].LU = tLU
            if doTW:
                out.iloc[i].Arrive = np.hstack([0, s[1:]-w[1:]])
                out.iloc[i].Wait = w
                out.iloc[i].TWmin = tmin
                out.iloc[i].Start = s
                out.iloc[i].Depart = s + tLU
                out.iloc[i].TWmax = tmax
                out.iloc[i].Total = t + w + tLU
    return (TC.item() if TC.size==1 else TC,Xflg,out)

def loccost(loc, C):
    # LOCCOST Calculate location sequence cost.
    # c = loccost(loc,C)
    #    loc = location vector
    #      C = n x n matrix of costs between n locations
    # 
    #  Example:
    #  loc = [1   2   4   3];
    #    C = triu(magic(4),1); C = C + C'
    #                                       C =  0   2   3  13
    #                                            2   0  10   8
    #                                            3  10   0  12
    #                                           13   8  12   0
    #  c = loccost(loc,C)
    #                                       c =  2
    #                                             8
    #                                            12

    if max(loc) > len(C):
        raise ValueError('Location exceeds size of cost matrix.')

    c = C[loc[:-1], loc[1:]]
    return c