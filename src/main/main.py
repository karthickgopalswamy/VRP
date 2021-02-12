from routing import shipment, shipment_struct
from utility import rte2loc, rte2idx, pairwisesavings, savings, get_distance, sh2rte
from cost import rte_tc
import numpy as np
import pandas as pd
import time

def get_route_region(pdf:pd.DataFrame):

    lat_long = np.concatenate((pdf[['pickup_lat','pickup_long']].values, pdf[['dropoff_lat','dropoff_long']].values))
    T = (get_distance(lat_long)/SPEED).astype(int)
    n = pdf.shape[0]
    b = np.array(range(n))
    e = b + n
    tbmin = pdf.food_ready_time.values
    tbmax = np.Inf*np.ones(tbmin.shape)
    temin = -np.Inf*np.ones(tbmin.shape)
    temax = tbmin + T[b,e]
    temax = np.max((temax, pdf.created_at.values + WINDOWLIMIT),axis = 0)

    sh = shipment_struct(b,e,tbmin,tbmax,temin,temax)
    rte_h = lambda rte:  rte_tc(sh,rte,T)
    IJS,S,TCij,Rij = pairwisesavings(rte_h,sh)
    r,Time = savings(rte_h,sh,IJS,dodisp=True)
    r,_, = sh2rte(sh,r,rte_h)
    TC,Xflg,outtemp = rte_tc(sh,r,T,isout=True)
    print(outtemp)
    out = pd.DataFrame(columns=['Route_ID','Route_Point_Index','Delivery_ID','Route_Point_Type'])
    data = []
    # for i,route in enumerate(r):
    #     n = route.shape[0]
    #     location = rte2loc(route,sh)
    #     pickup = route == location
    #     delivery_id = pdf.reset_index(drop=True).loc[route].delivery_id.values
    #     out.iloc[i].Depart
    #     data.append(np.block([[(np.ones(n)*i).astype(np.int64)],[route],[location],[pickup],[delivery_id]]).transpose())
    for i,route in enumerate(out):
        n = route.shape[0]
        location = rte2loc(route,sh)
        pickup = route == location
        delivery_id = pdf.reset_index(drop=True).loc[route].delivery_id.values
        data.append(np.block([[np.ones(n)*i],[route],[location],[pickup],[delivery_id]]).transpose())
    
    data = np.vstack(data)
    out = pd.DataFrame(data,columns=['Route_ID','Route_Point_Index','Location','isPickup','Delivery_ID'])
    print(out.head())

    return r

if __name__ == '__main__':
    start = time.time()
    SPEED = 4.5
    WINDOWLIMIT = 3600
    df = pd.read_csv('/home/karthick/VRP/resource/data.csv')
    df['created_at'] = pd.to_datetime(df['created_at']).astype(int)//10**9
    df['food_ready_time'] = pd.to_datetime(df['food_ready_time']).astype(int)//10**9

    routes = df.groupby('region_id').apply(get_route_region)

    # df_1 = df.set_index('region_id').loc[9].reset_index()
    # lat_long = np.concatenate((df_1[['pickup_lat','pickup_long']].values, df_1[['dropoff_lat','dropoff_long']].values))
    # T = (get_distance(lat_long)/SPEED).astype(int)
    # n = df_1.shape[0]
    # b = np.array(range(n))
    # e = b + n
    # tbmin = df_1.food_ready_time.values
    # tbmax = np.Inf*np.ones(tbmin.shape)
    # temin = -np.Inf*np.ones(tbmin.shape)
    # temax = tbmin + T[b,e]
    # temax = np.max((temax, df_1.created_at.values + WINDOWLIMIT),axis = 0)

    # sh = shipment_struct(b,e,tbmin,tbmax,temin,temax)
    # rte_h = lambda rte:  rte_tc(sh,rte,T)
    # print(rte_h(np.array([9,4,10,9,4,27,17,10,17,27])))
    # IJS,S,TCij,Rij = pairwisesavings(rte_h,sh)
    # r,Time = savings(rte_h,sh,IJS,dodisp=True)
    print(time.time()-start)
    print('Routes....')
    print(routes)