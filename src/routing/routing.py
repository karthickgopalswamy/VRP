import numpy as np
from typing import Union
class shipment:
    def __init__(self,
        b:int,
        e:int, 
        tbmin:Union[int,float]=None,
        tbmax: Union[int,float]=None,
        temin: Union[int,float]=None,
        temax:Union[int,float]=None
        ):
        self.__b = b
        self.__e = e
        self.__tbmin = tbmin
        self.__tbmax = tbmax
        self.__temin = temin
        self.__temax = temax

    @property
    def b(self):
        return self.__b    
    
    @property
    def e(self):
        return self.__e

    @property
    def tbmin(self):
        return self.__tbmin

    @property
    def tbmax(self):
        return self.__tbmax

    @property
    def temin(self):
        return self.__temin

    @property
    def temax(self):
        return self.__temax  

    def __str__(self):
        return f"b: {self.b}, e: {self.e}, tbmin: {self.tbmin}, tbmax: {self.tbmin}, temin: {self.temin}, temax: {self.temax}"
    def __repr__(self):
        return f"b: {self.b}, e: {self.e}, tbmin: {self.tbmin}, tbmax: {self.tbmin}, temin: {self.temin}, temax: {self.temax}"

class shipment_struct:
    def __init__(self,b,e,tbmin,tbmax,temin,temax):
        self.__shipments = dict()
        self.__b = b
        self.__e = e
        self.__tbmin = tbmin
        self.__tbmax = tbmax
        self.__temin = temin
        self.__temax = temax

        for i,element in enumerate(zip(b,e,tbmin,tbmax,temin,temax)):
            self.__shipments[i] = shipment(*element)

    @property
    def b(self):
        return self.__b    
    
    @property
    def e(self):
        return self.__e

    @property
    def tbmin(self):
        return self.__tbmin

    @property
    def tbmax(self):
        return self.__tbmax

    @property
    def temin(self):
        return self.__temin

    @property
    def temax(self):
        return self.__temax  
           
    def __len__(self):
        return len(self.__shipments)
    
    def __getitem__(self, key):
        return self.__shipments[key] if isinstance(key,int) else [self.__shipments[k] for k in key]

    def __str__(self):
        return ''.join([f"{sh}\n" for sh in self.__shipments.values()])


class driver:
    def __init__(self,b,e,maxTC=None):
        self.__b = b
        self.__e = e
        self.__maxTC = maxTC

    @property
    def b(self):
        return self.__b    
    
    @property
    def e(self):
        return self.__e

    @property
    def maxTC(self):
        return self.__maxTC

    