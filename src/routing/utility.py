import numpy as np

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
# %RTE2IDX Convert route to shipment index vector.
# % idx = rte2idx(rte)
# %   rte = route vector
# %       = m-element cell array of m route vectors
# %   idx = shipment index vector, such that idx = rte(isorigin(rte))
# %       = m-element cell array of m shipment index vectors
# %       
# % Example:
# % rte = [23   15   6   23   27   17   24   27   15   17   6   24];
# % idx = rte2idx(rte)    % idx = 23   15   6   27   17   24

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

def rte2loc(rtein,sh):
# %RTE2LOC Convert route to location vector.
# % loc = rte2loc(rte,sh)
# %     = rte2loc(rte,sh,tr) % Include beginning/ending truck locations
# %   rte = route vector
# %       = m-element cell array of m route vectors
# %    sh = structure array with fields:
# %        .b = beginning location of shipment
# %        .e = ending location of shipment
# %    tr = (optional) structure with fields:
# %        .b = beginning location of truck
# %           = sh(rte(1)).b, default
# %        .e = ending location of truck
# %           = sh(rte(end)).e, default
# %   loc = location vector
# %       = m-element cell array of m location vectors
# %       = NaN, degenerate location vector, which occurs if bloc = eloc and
# %         truck returns to eloc before end of route (=> > one route)
# %
# % Example:
# % rte = [1   2  -2  -1];
# %  sh = vect2struct('b',[1 2],'e',[3 4]);
# % loc = rte2loc(rte,sh)               % loc = 1   2   4   3

    if not isinstance(rtein, list): 
        rte = [rtein]
    else:
        rte = rtein

    loc = [None]*(len(rte))
    for i in range(len(rte)):
        if sum(rte[i]) is None:
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
        loc = np.asarray(*loc)

    return loc


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

  c = np.diag(C[loc[:-1], loc[1:]])
  return c