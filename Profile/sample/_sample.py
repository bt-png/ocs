# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:11:19 2024

@author: TharpBM
"""
import time
import pandas as pd
import _system as OCS

st = time.time()
## CONDUCTOR PARTICULARS AND LOADING CONDITIONS
cN = OCS.conductor_particulars((1.544, 5000), (1.063, 3300), 0.2)

## LAYOUT DESIGN
filepath_none = 'InputData_none.csv'
wr_none = OCS.wire_run(filepath_none)
filepath_neg = 'InputData_neg.csv'
wr_neg = OCS.wire_run(filepath_neg)
filepath_pos = 'InputData_pos.csv'
wr_pos = OCS.wire_run(filepath_pos)

# =============================================================================
# STANDARD DESIGN DATA
# =============================================================================
PreSagNone = OCS.CatenaryFlexible(cN, wr_none)
PreSagNone._solve()
PreSagNeg = OCS.CatenaryFlexible(cN, wr_neg)
PreSagPos = OCS.CatenaryFlexible(cN, wr_pos)

#=========PLOT==============

pl = OCS.combine_plots(
    ('Neg', 'Pos', 'None'),
    (PreSagNeg.dataframe(), PreSagPos.dataframe(), PreSagNone.dataframe()),
    row='type'
)


# =============================================================================
# SLICED DESIGN DATA
# =============================================================================
start = 3
end = 7

SlicedWireRun_None = OCS.wire_run('InputData_none.csv', start, end)
PreSagNone_sliced = OCS.CatenaryFlexible(cN, SlicedWireRun_None)
PreSagNone_sliced._solve()

SlicedWireRun_Neg = OCS.wire_run('InputData_neg.csv', start, end)
PreSagNeg_sliced = OCS.CatenaryFlexible(cN, SlicedWireRun_Neg)

SlicedWireRun_Pos = OCS.wire_run('InputData_pos.csv', start, end)
PreSagPos_sliced = OCS.CatenaryFlexible(cN, SlicedWireRun_Pos)

#=========PLOT==============
pl = OCS.combine_plots(
    ('Neg', 'Pos', 'None'),
    (PreSagNeg_sliced.dataframe(), PreSagPos_sliced.dataframe(), PreSagNone_sliced.dataframe())
)

# =============================================================================
# ALTERNATE DESIGN DATA
# SLICED DESIGN DATA
# =============================================================================
#Utilizing No PreSag, and same slicing as above
## LC1 Static, Maximum - cold, no ice
c1 = OCS.conductor_particulars((1.544, 5129), (1.063, 3385), 0.2)
## LC2 Static, Minimum - hot, no wind
c2 = OCS.conductor_particulars((1.544, 4187), (1.063, 2482), 0.2)
Uplift = OCS.AltCondition(c1, PreSagNone_sliced)
Uplift.resetloadlist((-25,163050))

Uplift.compareplot()


# =============================================================================
# ELASTICITY CHECK
# SLICED DESIGN DATA
# =============================================================================
Elasticity = OCS.Elasticity(c1, PreSagNone_sliced, 25, 1, 1, 2)
Elasticity.plot()

et = time.time()
elapsed_time = et - st
print('total elapsed time =', elapsed_time)
