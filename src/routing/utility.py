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



def pairwisesavings(rteTC_h,sh,TC1,doZero):
#   %PAIRWISESAVINGS Calculate pairwise savings.
# % [IJS,S,TCij,Rij] = pairwisesavings(rteTC_h,sh,TC1,doNegSav)
# % rteTC_h = handle to route total cost function, rteTC_h(rte)
# %      sh = structure array with fields:
# %          .b = beginning location of shipment
# %          .e = ending location of shipment
# %     TC1 = (optional) user-supplied independent shipment total cost
# %         = rteTC_h([i i]) for shipment i, default
# %  doZero = set negative savings values to zero
# %         = true, default
# %     IJS = savings list in nonincreasing order, where for row IJS(i,j,s)
# %           s is savings associated with adding shipments i and j to route
# %         = [], default, savings
# %       S = savings matrix
# %    TCij = pairwise total cost matrix
# %     Rij = pairwise route cell array

    n = len(sh)
    S = np.zeros(n)
    TCij = np.zeros(n)
    Rij = []
    if not TC1:
        for i in range(n):
            TC1[i] = rteTC_h([i,i])

    for i in range(n-1):
        for j in range(i+1,n,1):
            [rij,tcij] = mincostinsert([i,i], [j,j], rteTC_h, sh)
            s = TC1[i] + TC1[j] - TCij
            if not(doZero) or s > 0:
                S[i][j] = s
                TCij[i][j] = tcij
                Rij[i][j] = rij
                Rij[j][i] = rij

    s = S[i,j]
    IJS = IJS[np.argsort(-s),:]
    S = S + S.transpose()
    TCij = TCij + TCij.transpose()
    return [IJS,S,TCij,Rij]


def savings(rteTC_h,sh,IJS,dodisp):
# %SAVINGS Savings procedure for route construction.
# %[rte,TC] = savings(rteTC_h,sh,IJS,dodisp)
# %         = savings(rteTC_h,sh,IJS,prte_h)
# % rteTC_h = handle to route total cost function, rteTC_h(rte)
# %     sh  = structure array with fields:
# %          .b = beginning location of shipment
# %          .e = ending location of shipment
# %     IJS = 3-column savings list
# %         = pairwisesavings(rteTC_h)
# %  dodisp = display intermediate results = false, default
# %  prte_h = handle to route plotting function, prte_h(rte)
# %           (dodisp = true when handle input)
# %     rte = route vector
# %         = m-element cell array of m route vectors
# %   TC(i) = total cost of route i

    TCr = []
    if not IJS:
        rte = []
        return [rte,TCr]
    if len(sh) < 2:
        rte = [np.ones(2)]
        TCr = rteTC_h(rte[1])
        return [rte,TCr]

    i = IJS[:,0]
    j = IJS[:,1]

    for k in range(len(sh)):
        TC1[k] = rteTC_h([k,k])

    if dodisp:
        print('Savings \n')

    inr = np.zeros((len(sh),1))
    rte = []
    did = []
    Did = []
    n = 0
    done = False

    while not done:
        ischg = False
        for k in range(len(i)):
            ik = i[k]
            jk = j[k]
            if inr[ik] == 0 and inr[jk] == 0:
                [rij,TCij] = mincostinsert([ik,ik],[jk,jk],rteTC_h,sh,True)
                sij = TC1[ik] + TC1[jk] - TCij
                if sij > 0:
                    n = n + 1
                    rte[n] = rij
                    TCr[n] = TCij
                    inr[ik] = n
                    inr[jk] = n
                    ischg = True
                    did = [did; np.zeros((1,len(sh)))]
                    Did = [Did, np.zeros((n-1,1)); np.zeros((1,n))]
                    if dodisp:
                        c = sum(rteTC_h(rte[not np.isnan(TCr)]))
                        print('Make Rte %d using %d and %d\n',c,n,ik,jk)

                elif inr[ik] != inr[jk]:
                    if inr[ik] == 0:
                        temp = jk
                        jk = ik
                        ik = temp
                    if not did[inr[ik],jk]:
                        [rij,TCij] = mincostinsert([jk,jk],rte[inr[ik]],rteTC_h,sh,True)
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
                                c = sum(rteTC_h(rte[not np.isnan(TCr)]))
                                print('Add %d to Rte %d\n',c,jk,inr[ik])

                elif (inr[ik] != inr[jk]) and (not Did[inr[ik],inr[jk]]) or Did[inr[jk],inr[ik]]:
                    [rij,TCij] = mincostinsert(rte[inr[ik]],rte[inr[jk]],rteTC_h,sh,True)
                    Did[inr[ik],inr[jk]] = True
                    Did[inr[jk],inr[ik]] = True
                    if not np.isnan(rij):
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
                                c = sum(rteTC_h(rte[not np.isnan(TCr)]));
                                print('Combine Rte %d to Rte %d\n',c,inrjk,inr(ik))

        if not ischg or (not sum((not inr))):
            done = True

    if rte:
        rte[np.isnan(TCr)] = []
    if rte:
        TCr[np.isnan[TCr]] = []
    return [rte,TCr]


def mincostintert(rtei,rtej,rteTC_h,sh,doNaN):
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

    rteiTC = rteTC_h(rtei) # import this function
    rtejTC = rteTC_h(rte)

    if (len(rte) < len(rtei)) or ((len(rte) == len(rtei)) and (rtejTC < rteiTC)):
        temp = rte 
        rte = rtei
        rtei = temp

    si = rte2idx(rtei) # import this function
    for i in range(len(si)):
        loc = rte2loc(rte,sh) # import this function
        bloci = sh.b[si[i]]
        isduploc = [False,loc[:-1] == loc[1:]]
        [rte,minTC] = mincostshmtinsert(si[i],rte,rteTC_h,isduploc,loc,bloci)
        if np.isinf(minTC):
            return [rte,minTC,TCij]

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
            if (j>1) and (isduploc[j-1]):
                continue

            if (i<len(rte)+1) and (bloci == loc[i]) and rte[i] < 0:
                break

            rij = [rte[:i-1],idx,rte[i:j-1],idx,rte[j:]]
            cij = rteTC_h(rij)

            if cij < minTC:
                minTC = cij
                mini = i
                minj = j

    if not np.isinf(minTC):
        rte = [rte[:mini-1],idx,rte[mini:minj-1],idx,rte[minj:]]
    else:
        rte = np.nan

    return [rte,minTC]