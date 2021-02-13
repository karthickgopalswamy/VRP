from routing import shipment, shipment_struct
from utility import rte2loc, rte2idx, pairwisesavings, savings, get_distance, sh2rte
from cost import rte_tc, get_metrics
import numpy as np
import pandas as pd
import time
import configparser
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
    data = []
    for i,route in enumerate(r):
        n = route.shape[0]
        location = rte2loc(route,sh)
        pickup = route == location
        delivery_id = pdf.reset_index(drop=True).loc[route].delivery_id.values
        created_time = pdf.reset_index(drop=True).loc[route].created_at.values
        DepartTime = outtemp.iloc[i].Depart
        data.append(np.block([[(np.ones(n)*i).astype(np.int64)],[np.arange(n,dtype=np.int64)],[pickup],[delivery_id],[DepartTime],[created_time]]).transpose())
    
    data = np.vstack(data)
    out = pd.DataFrame(data,columns=['Route_ID','Route_Point_Index','isPickup','Delivery_ID','Route_Point_Time','created_time'])
    out['route_count'] = len(r)
    print(out.head())

    return out

if __name__ == '__main__':
    start = time.time()
    config = configparser.ConfigParser()
    config.read('config.ini')
    SPEED = float(config['PARAMS']['speed'])
    WINDOWLIMIT = int(config['PARAMS']['windowlimit'])
    PATH = config['PATH']['filepath']
    df = pd.read_csv(PATH)
    df['created_at'] = pd.to_datetime(df['created_at']).astype(int)//10**9
    df['food_ready_time'] = pd.to_datetime(df['food_ready_time']).astype(int)//10**9

    routes = df.groupby('region_id').apply(get_route_region)

    region_map = routes.reset_index(level=0).groupby('region_id').route_count.unique()
    region_mp = region_map.shift(1).cumsum().fillna(0) 

    routes = routes.reset_index(level=0)
    routes['Route_ID'] = routes.apply(lambda x: x.Route_ID + (region_mp.loc[x.region_id] if isinstance(region_mp.loc[x.region_id],int) else region_mp.loc[x.region_id].item())  ,axis = 1)

    wait_time, delivery_ph = get_metrics(routes)
    print('Average wait time for the solution is: {0}, \n Deliveries per hour is: {1}'.format(wait_time,delivery_ph))
    print(f'Total solver time is: {time.time()-start} seconds')
    print('Total numner of routes is {}'.format(routes.Route_ID.nunique()))

    routes['Route_Point_Type'] = routes.apply(lambda x: 'Pickup' if x.isPickup else 'Dropoff', axis = 1)
    columns = ['Route_ID','Route_Point_Index','Delivery_ID','Route_Point_Type','Route_Point_Time']
    routes[columns[:3]] = routes[columns[:3]].astype(np.int64)
    routes[columns].to_csv('Solution_Karthick_Gopalswamy.csv',index=False)