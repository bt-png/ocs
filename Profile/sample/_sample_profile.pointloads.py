# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:11:19 2024

@author: TharpBM
"""
import time
import sys
sys.path.insert(0, 'U:\\Tools-Coding\\Python\\Profile')
from library import system as OCS

st = time.time()
## CONDUCTOR PARTICULARS AND LOADING CONDITIONS
cN = OCS.conductor_particulars((1.544, 5000), (1.063, 3300), 0.2)

## LAYOUT DESIGN
filepath_none = '..\\input\\InputData_none.csv'
wr = OCS.wire_run(filepath_none)


## POINT LOADS
pN = (
('term span', 'uninsulated overlap', 'mpa z-anchor', 'mpa z-anchor', 'SI and feeders', 'insulated overlap', 'term span'),
(162442+30, 162442-15, 164755-25, 164755+25, 166379-15, 166980-15, 167160-30),
(15, 5, -15, -15, 30, 5, 15),
(15, 5, 0, 0, 85, 5, 15)
)

Nominal = OCS.CatenaryFlexible(cN, wr)
Nominal.resetloads(pN)

#=========PLOT==============

Nominal.plot()
Nominal.plot_spt()
Nominal.plot_halength()

et = time.time()
elapsed_time = et - st
print('total elapsed time =', elapsed_time)
