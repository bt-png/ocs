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
# LAYOUT DESIGN
## Data loaded from Sound Transit L800, wire run N51
wirerunfilepath = '..\\input\\InputData_none.csv'
wr = OCS.wire_run(wirerunfilepath)

# CONDUCTOR PARTICULARS AND LOADING CONDITIONS
## Nominal
cN = OCS.conductor_particulars((1.544, 5000), (1.063, 3300), 0.2)
## LC1 Static, Maximum - cold, no ice
c1 = OCS.conductor_particulars((1.544, 5129), (1.063, 3385), 0.2)
## LC2 Static, Minimum - hot, no wind
c2 = OCS.conductor_particulars((1.544, 4187), (1.063, 2482), 0.2)

# SOLVE
## Solve flexible hanger catenary geometry utilizing the _systems module
Nominal = OCS.CatenaryFlexible(cN, wr)
# since we arent planning to change any other variables, and the model
# doesn't solve until its called, we need to call it now
Nominal._solve()
LC1 = OCS.AltCondition(c1, Nominal)
LC2 = OCS.AltCondition(c2, Nominal)

# OUTPUT
## Generate sag plots of the wire elevations
pl = OCS.combine_plots(
    ('LC1', 'LC2', 'Nominal'),
    (LC1.dataframe(), LC2.dataframe(), Nominal.dataframe()),
    #row='type'
)
## Generate plots of the Contact Wire elevation deviation
pl = OCS.combine_plots(
    ('LC1', 'LC2'),
    (LC1.dataframe_cwdiff(), LC2.dataframe_cwdiff()),
    yscale=12
    #row='type'
)
### Update axis titles
pl.set_axis_labels(y_var='Rise (in)')
## Generate plots of the steady arm vertical resistive loading
## due to changes in the installed heel setting and radial load
pl_load = OCS.combine_plots_load(
    ('LC1', 'LC2'),
    (LC1.dataframe_sr(), LC2.dataframe_sr()),
    #row='type'
)

et = time.time()
elapsed_time = et - st
print('total elapsed time =', elapsed_time)
