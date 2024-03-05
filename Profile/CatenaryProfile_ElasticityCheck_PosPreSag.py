# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:36:43 2024

@author: TharpBM
"""
import pandas as pd
import numpy as np
import seaborn as sns
import Template_GeneralFunctions as GenFun
import Template_CatenaryFlexibleFunctions as Flex

# INPUT DATA
## CALCULATION AND DESIGN CONSTANTS

### Calculation Variables
xRound = 1
xStep = 1
xMultiplier = 1
yMultiplier = 1
xStaticMultiplier = 1
yStaticMultiplier = 1

### Design variables (Base Units [ft, lbf])
MaxHASpacing = 30
minCW_P = 0
MinHALength = 3/12
HA_Accuracy = 1/50/12
SteadyArmLength = 3

## CONDUCTOR PARTICULARS AND LOADING CONDITIONS
Conductors = pd.DataFrame({
    'Type': ['Messenger Wire', 'Contact Wire', 'Hangers'],
    'Weight': [1.544, 1.063, 0.2],
    'Tension': [5000, 3300, 0]
     })

## LAYOUT DESIGN
df = pd.read_csv('InputData.csv')

## DISCRETE SPAN LOADS AND DESIGN CONDITIONS

### Discrete span loads. + weight is in the downard direction
STA_DiscreteLoad_CW = np.array([162298, 162612, 0])
P_DiscreteLoad_CW = np.array([5, 10, 0])
STA_DiscreteLoad_MW = np.copy(STA_DiscreteLoad_CW)
P_DiscreteLoad_MW = np.array([0, 0, 0])

### Steady arms with fixed elevations /applied load calculated by span geometry
ExpectedSupport_P = 5
STA_FixedSupport_CW = np.array([0, 0, 0, 0])
FixedSupport_EL_Tolerance = 0.1/12

### Critical Clearance Stationing and Elevation
ClearanceSTA = np.array([0, 0])
ClearanceEL = np.array([0, 0])

### Stationing of Interest
FocusStation = 165250
FocusNeg = 350
FocusPos = 350

# CALCULATIONS
FullWireRun = GenFun.WireRun_df(df,xRound)
FullWireRun['PreSag'] = 0

L_DesignBase = [xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength]
L_SpanLoading = [
    P_DiscreteLoad_CW, STA_DiscreteLoad_CW,
    P_DiscreteLoad_MW, STA_DiscreteLoad_MW
    ]
L_DesignHA = [MaxHASpacing, minCW_P, MinHALength, HA_Accuracy]
L_StaticCWLoading = [-25, 162475]

Nominal = Flex.CatenarySag_Flexible(
    FullWireRun, Conductors['Weight'], Conductors['Tension'], L_SpanLoading,
    L_DesignHA, L_DesignBase)
""" Nominal dictionary keys
Stationing, HA_STA, HA_EL
LoadedSag, SupportLoad_CW, P_SpanWeight
LoadedSag_MW, SupportLoad_MW, P_SpanWeight_MW
"""

NominalSag = pd.DataFrame({
    'Stationing': np.append(
        Nominal.get('Stationing'), Nominal.get('Stationing')
        ),
    'Sag': np.append(
        Nominal.get('LoadedSag'), Nominal.get('LoadedSag_MW')
        ),
    'Cable Type': np.append(
        (Nominal.get('LoadedSag')*0), Nominal.get('LoadedSag_MW')*0 + 1
        )
    })
NominalSag=NominalSag.pivot(
    index='Stationing', columns='Cable Type', values='Sag'
    )
sns.lineplot(data=NominalSag)
NominalSag.to_csv('OutputData_Nominal.csv')

Uplift, cloops, NewCWSupportReaction = Flex.CatenarySag_FlexibleHA_Iterative(
    FullWireRun, 0*Conductors['Weight'], Conductors['Tension'], L_DesignHA,
    L_DesignBase, L_StaticCWLoading, Nominal)
""" Uplift dictionary keys
Stationing, HA_STA, HA_EL
LoadedSag, SupportLoad_CW, P_SpanWeight
LoadedSag_MW, SupportLoad_MW, P_SpanWeight_MW
"""

UpliftSag = pd.DataFrame({
    'Stationing': np.append(
        Uplift.get('Stationing'), Uplift.get('Stationing')
        ),
    'Sag': np.append(
        Uplift.get('LoadedSag'), Uplift.get('LoadedSag_MW')
        ),
    'Cable Type': np.append(
        (Uplift.get('LoadedSag')*0), Uplift.get('LoadedSag_MW')*0 + 1
        )
    })
UpliftSag = UpliftSag.pivot(
    index='Stationing', columns='Cable Type', values='Sag'
    )
sns.lineplot(data=UpliftSag)
UpliftSag.to_csv('OutputData_Uplift.csv')

DiffCW = (Uplift.get('LoadedSag') - Nominal.get('LoadedSag'))*12
DiffMW = (Uplift.get('LoadedSag_MW') - Nominal.get('LoadedSag_MW'))*12
DiffSag = pd.DataFrame({
    'Stationing': np.append(
        Uplift.get('Stationing'), Uplift.get('Stationing')
        ),
    'Sag': np.append(DiffCW, DiffMW),
    'Cable Type': np.append(
        (Uplift.get('LoadedSag')*0), Uplift.get('LoadedSag_MW')*0 + 1
        )
    })
DiffSag = DiffSag.pivot(
    index='Stationing', columns='Cable Type', values='Sag')
sns.lineplot(data=DiffSag)

FocusSPT = 2
StartSPT = FocusSPT - 1
EndSPT = FocusSPT + 1
Elasticity = Flex.CatenaryElasticity_FlexibleHA(
    FullWireRun, 0*Conductors['Weight'], Conductors['Tension'], 
    L_DesignHA, L_DesignBase, 25, 1, Nominal, StartSPT, EndSPT)
""" Elasticity dictionary keys
STAVal, DiffMIN, DiffMAX, CycleLoops
"""

ElasticityGraph = pd.DataFrame(
    {'Stationing': np.append(
        Elasticity.get('STAVal'), Elasticity.get('STAVal')
        ),
     'Uplift': np.append(
         Elasticity.get('DiffMIN')*12, Elasticity.get('DiffMAX')*12
         ),
     'Min/Max': np.append(
         (Elasticity.get('DiffMIN')*0), Elasticity.get('DiffMAX')*0 + 1
         )
     })
ElasticityGraph=ElasticityGraph.pivot(
    index='Stationing', columns='Min/Max', values='Uplift'
    )
sns.lineplot(data=ElasticityGraph)
ElasticityGraph.to_csv('OutputData_Elasticity.csv')

# OUTPUT

