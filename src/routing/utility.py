import numpy as np
from sklearn.metrics.pairwise import haversine_distances as dist
from itertools import compress
from typing import Iterable
def isorigin(rte):
#   ISORIGIN Identify origin of each shipment in route.
#   is = isorigin(rte)
#   rte = route vector
#   is = logical vector, such that is(i) = true if rte(i) is origin

  rte = np.argsort(rte)
  i = np.zeros(len(rte),dtype=np.bool)
  i[rte[0::2]] = True
  return i

def rte2idx(rtein):
# """
# RTE2IDX Convert route to shipment index vector.
#     Params:
#         rte = route vector
#             = m-element list of m route vectors
#     Returns:
#         idx = shipment index vector, such that idx = rte(isorigin(rte))
#             = m-element list of m shipment index vectors
       
#     Example:
#     rte = [23   15   6   23   27   17   24   27   15   17   6   24];
#     idx = rte2idx(rte)     idx = 23   15   6   27   17   24
    
    if not isinstance(rtein, list): 
        rte = [rtein]
    else:
        rte = rtein

    idx = [None]*(len(rte))
    for i in range(len(rte)):
        idx[i] = rte[i][isorigin(rte[i])]

    if not any([isinstance(rtein, list)]): 
        idx = np.asarray(*idx)

    return idx


def sh2rte(idx,rtein,rteTC_h):
    idx = np.array(range(len(idx)))
    if not isinstance(rtein,list) and len(rtein) > 0:
        rte = [rtein]
    else:
        rte = rtein
    if len(rte) > 0:
        ri = rte2idx(np.concatenate(rte).ravel())
    else:
        ri = None
    idx1 = np.setdiff1d(idx,ri)
    n = len(rte)
    for i in range(len(idx1)):
        rte.append(np.array([idx1[i],idx1[i]]))
    if rteTC_h:
        if len(idx1) > 0:
            print('ADD SINGLE-SHIPMENT ROUTES:\n: Added shipments')
            print('{}\n\n'.format(idx1))
    return (rte,idx1)

def rte2loc(rtein,sh):

# """
#     RTE2LOC Convert route to location vector.
#     Params:
#         loc = rte2loc(rte,sh)
#             = rte2loc(rte,sh,tr)  Include beginning/ending truck locations
#         rte = route vector
#             = m-element cell array of m route vectors
#         sh = shiment struct with fields:
#             .b = beginning location of shipment
#             .e = ending location of shipment
#     Returns:
#         loc = location vector
#             = m-element List of m location vectors"""



    if not isinstance(rtein, list): 
        rte = [rtein]
    else:
        rte = rtein

    loc = [np.empty_like(ele) for ele in rte]
    for i in range(len(rte)):
        if np.isnan(sum(rte[i])):
            loc[i] = np.nan
            continue

        loc[i] = np.zeros(np.size(rte[i]),dtype=int)
        iss = isorigin(rte[i])
        loc[i][iss] = sh.b[rte[i][iss]]
        loc[i][np.invert(iss)] = sh.e[rte[i][np.invert(iss)]]
        isrow = loc[i].shape[0] == 1

        if isrow:
            loc[i] = loc[i][:].transpose()

    if not any([isinstance(rtein, list)]): 
        loc = loc[0]

    return loc



def pairwisesavings(rteTC_h,sh,TC1=None,doZero=True):

# """
#   PAIRWISESAVINGS Calculate pairwise savings.
#  [IJS,S,TCij,Rij] = pairwisesavings(rteTC_h,sh,TC1,doNegSav)
#     Params:
#         rteTC_h = lambda handle to route total cost function, rteTC_h(rte)
#             sh = structure array with fields:
#                 .b = beginning location of shipment
#                 .e = ending location of shipment
#             TC1 = (optional) user-supplied independent shipment total cost
#                 = rteTC_h([i i]) for shipment i, default
#         doZero = set negative savings values to zero
#             = true, default
#     Returns:
#         IJS = savings list in nonincreasing order, where for row IJS(i,j,s)
#             s is savings associated with adding shipments i and j to route
#             = [], default, savings
#         S = savings matrix
#         TCij = pairwise total cost matrix
#         Rij = pairwise route list"""

    n = len(sh)
    S = np.zeros(shape=(n,n))
    TCij = np.zeros(shape=(n,n))
    Rij = [[None]*n for i in range(n)]
    if not TC1:
        TC1 = np.zeros(n)
        for i in range(n):
            TC1[i],_,_ = rteTC_h(np.array([i,i]))

    for i in range(n-1):
        for j in range(i+1,n):
            rij,tcij,_ = mincostinsert(np.asarray([i,i]), np.array([j,j]), rteTC_h, sh)
            s = TC1[i] + TC1[j] - tcij
            if not(doZero) or s > 0:
                S[i][j] = s
                TCij[i][j] = tcij
                Rij[i][j] = rij
                Rij[j][i] = rij

    i,j = S.nonzero()
    s = S[i,j]
    IJS = np.vstack([i,j,s]).transpose()
    IJS = IJS[np.argsort(-s),:]
    S = S + S.transpose()
    TCij = TCij + TCij.transpose()
    return [IJS,S,TCij,Rij]


def savings(rteTC_h,sh,IJS,dodisp):
    # SAVINGS Savings procedure for route construction.
    # [rte,TC] = savings(rteTC_h,sh,IJS,dodisp)
    #          = savings(rteTC_h,sh,IJS,prte_h)
    #  rteTC_h = handle to route total cost function, rteTC_h(rte)
    #      sh  = structure array with fields:
    #           .b = beginning location of shipment
    #           .e = ending location of shipment
    #      IJS = 3-column savings list
    #          = pairwisesavings(rteTC_h)
    #   dodisp = display intermediate results = false, default
    #   prte_h = handle to route plotting function, prte_h(rte)
    #            (dodisp = true when handle input)
    #      rte = route vector
    #          = m-element cell array of m route vectors
    #    TC(i) = total cost of route i

    TCr = []
    n = len(sh)
    if not IJS.size:
        rte = np.empty(shape=0)
        return [rte,TCr]
    if n < 2:
        rte = [np.zeros(2)]
        TCr,_,_ = rteTC_h(rte[1])
        return [rte,TCr]

    i = IJS[:,0].astype(np.int64)
    j = IJS[:,1].astype(np.int64)
    TC1 = np.zeros(n)
    for k in range(n):
        TC1[k],_,_ = rteTC_h(np.array([k,k]))

    if dodisp:
        print('Savings \n')

    inr = -1*np.ones(len(sh),dtype=np.int64)
    rte = []
    did = np.empty(shape=(0,len(sh)))
    Did = np.empty(shape=(0,0))
    n = -1
    done = False

    while not done:
        ischg = False
        for k in range(len(i)):
            ik = i[k]
            jk = j[k]
            if inr[int(ik)] == -1 and inr[int(jk)] == -1:
                rij,TCij,_= mincostinsert(np.array([ik,ik]),np.array([jk,jk]),rteTC_h,sh,True)
                sij = TC1[ik] + TC1[jk] - TCij
                if sij > 0:
                    n = n + 1
                    rte.append(rij)
                    TCr.append(TCij)
                    inr[ik] = n
                    inr[jk] = n
                    ischg = True

                    if n == 0:
                        did = np.zeros((1,len(sh)))
                        Did = np.array([[0]])
                    else: 
                        did = np.block([[did],[np.zeros((1,len(sh)))]])
                        Did = np.block([[Did, np.zeros((n,1))],
                                        [np.zeros((1,n+1))]
                                        ])
                    if dodisp:
                        tc,_,_ = rteTC_h(list(compress(rte,np.invert(np.isnan(TCr)))))
                        c = sum(tc) if isinstance(tc,Iterable) else tc
                        print('{0}: Make Rte {1} using {2} and {3}\n'.format(c,n,ik,jk))

            elif np.logical_xor(inr[ik]+1,inr[jk]+1):
                if inr[ik] == -1:
                    ik,jk = jk,ik
                if not did[inr[ik],jk]:
                    rij,TCij,_ = mincostinsert(np.array([jk,jk]),rte[inr[ik]],rteTC_h,sh,True)
                    did[inr[ik],jk] = True
                    sij = TC1[jk] + TCr[inr[ik]] - TCij
                    if sij > 0:
                        rte[inr[ik]] = rij
                        TCr[inr[ik]] = TCij
                        inr[jk] = inr[ik]
                        did[inr[ik],:] = False
                        Did[inr[ik],:] = False
                        Did[:,inr[ik]] = False
                        if dodisp:
                            tc,_,_ = rteTC_h(list(compress(rte,np.invert(np.isnan(TCr)))))
                            c = sum(tc) if isinstance(tc,Iterable) else tc
                            print('{0}: Add {1} to Rte {2}\n'.format(c,jk,inr[ik]))

            elif (inr[ik] != inr[jk]) and not (Did[inr[ik],inr[jk]] or Did[inr[jk],inr[ik]]):
                rij,TCij,_ = mincostinsert(rte[inr[ik]],rte[inr[jk]],rteTC_h,sh,True)
                Did[inr[ik],inr[jk]] = True
                Did[inr[jk],inr[ik]] = True
                if not np.isnan(rij).all():
                    sij = TCr[inr[ik]] + TCr[inr[jk]] - TCij
                    if sij > 0:
                        inrjk = inr[jk]
                        inr[rte[inrjk][isorigin(rte[inrjk])]] = inr[ik]
                        rte[inrjk] = np.nan
                        TCr[inrjk] = np.nan
                        rte[inr[ik]] = rij
                        TCr[inr[ik]] = TCij
                        ischg = True
                        if dodisp:
                            tc,_,_ = rteTC_h(list(compress(rte,np.invert(np.isnan(TCr)))))
                            c = sum(tc) if isinstance(tc,Iterable) else tc
                            print('{0}: Combine Rte {1} to Rte {2}\n'.format(c,inrjk,inr[ik]))

        if not ischg or np.all(inr >=0):
            done = True

    if rte:
        rte = list(compress(rte,np.invert(np.isnan(TCr))))
        # rte[np.isnan(TCr)] = np.empty(shape=0)
    if rte:
        TCr = list(compress(TCr,np.invert(np.isnan(TCr))))
        # TCr[np.isnan[TCr]] = np.empty(shape=0)
    return [rte,TCr]


def mincostinsert(rtei,rte,rteTC_h,sh,doNaN=False):
# MINCOSTINSERT Min cost insertion of route i into route j.
# [rte,TC,TCij] = mincostinsert(rtei,rte,rteTC_h,sh,doNaN)
#     rtei = route i vector
#     rtej = route j vector
#  rteTC_h = handle to route total cost function, rteTC_h(rte)
#       sh = structure array with fields:
#           .b = beginning location of shipment
#           .e = ending location of shipment
#    doNaN = return NaN for rte if cost increase
#          = false, default
#      rte = combined route vector
#          = NaN if cost increase
#       TC = total cost of combined route
#     TCij = original total cost of routes i and j

    rteiTC,_,_ = rteTC_h(rtei) # import this function
    rtejTC,_,_ = rteTC_h(rte)

    if (len(rte) < len(rtei)) or ((len(rte) == len(rtei)) and (rtejTC < rteiTC)):
        rte, rtei = rtei,rte

    si = rte2idx(rtei) # import this function
    for i in range(len(si)):
        loc = rte2loc(rte,sh) # import this function
        bloci = sh.b[si[i]]
        isduploc = np.hstack([False,loc[:-1] == loc[1:]])
        [rte,minTC] = mincostshmtinsert(si[i],rte,rteTC_h,isduploc,loc,bloci)
        if np.isinf(minTC):
            return [rte,minTC,np.Inf]

    if doNaN and (minTC >= rteiTC + rtejTC):
        rte = np.nan

    TCij = rteiTC + rtejTC
    return [rte,minTC,TCij]



def mincostshmtinsert(idx,rte,rteTC_h,isduploc,loc,bloci):

    if sum(idx == rte2idx(rte)):
        rte = np.nan
        minTC = np.Inf
        return [rte,minTC]

    minTC = np.Inf
    for i in range(len(rte)+1):
        for j in range(i,len(rte)+1):
            if (j>0) and (isduploc[j-1]):
                continue

            if (i<len(rte)) and (bloci == loc[i]) and rte[i] < 0:
                break

            rij = np.hstack([rte[:i],idx,rte[i:j],idx,rte[j:]])
            cij,_,_ = rteTC_h(rij)

            if cij < minTC:
                minTC = cij
                mini = i
                minj = j

    if not np.isinf(minTC):
        rte = np.hstack([rte[:mini],idx,rte[mini:minj],idx,rte[minj:]])
    else:
        rte = np.nan

    return [rte,minTC]

 
def get_distance(points):
    """points assumed to be latitue and longitude in  m"""
    return dist(np.radians(points))*6371*1000
    