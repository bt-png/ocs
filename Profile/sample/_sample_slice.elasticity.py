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
# ELASTICITY CONDITIONS
## Positive Uplift force is in the upward direction.
### (positive weight is normally downwards, but case has changed due to name)
UpliftForce = 25
## Resolution of elasticity graph
SpanStepSize = 1
## Starting location for analysis from wire run
StartSPT = 3
## Recommend to limit analysis to 1 to 3 spans for simulation time
## Recommend at least 3 span buffer for simulation accuracy
SpanstoAnalyze = 1
StartBuffer = 3
EndBuffer = 3

# LAYOUT DESIGN
## Data loaded from Sound Transit L800, wire run N51
wirerunfilepath = '..\\input\\InputData_none.csv'
wr = OCS.wire_run(wirerunfilepath, StartSPT, StartSPT+StartBuffer+SpanstoAnalyze+EndBuffer)

# CONDUCTOR PARTICULARS AND LOADING CONDITIONS
## Input format is (MW(Weight, Tension), CW(Weight, Tension), HA Weight)
cN = OCS.conductor_particulars((1.544, 5000), (1.063, 3300), 0.2)

# SOLVE
## Solve flexible hanger catenary geometry utilizing the _systems module
Nominal = OCS.CatenaryFlexible(cN, wr)
Nominal.plot()

Elasticity = OCS.Elasticity(cN, Nominal, UpliftForce, SpanStepSize, StartBuffer, StartBuffer+SpanstoAnalyze)
Elasticity.plot()

Elasticity.plot_upliftedEL()

et = time.time()
elapsed_time = et - st
print('total elapsed time =', elapsed_time)
