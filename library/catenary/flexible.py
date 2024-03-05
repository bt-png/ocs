# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:36:43 2024

@author: TharpBM
"""
import numpy as np
import sys
import ../general as GenFun

dimensionalTolerance = 0.001*12
loadTolerance = 0.01

def SagSpanData(STAList, LoadList, TensionList, HangerStationing, HangerElevation, STARound):
    """
    Determine the final elevation for each point using sum of moments method.
    
    Parameters:
        STAList (floats): x-axis values for determining elevations
        LoadList (floats): vertical loading at each STAList
        TensionList (floats): horizontal loading at each STAList
        HangerStationing (floats): controlled elevations
        HangerElevation (floats): corresponding elevations
        STARound (float): used to match HangerStationing to STAList
    Returns:
        floats: Elevations corresponding to the input STAList
    """
    #EL = np.empty_like(STAList)
    EL = STAList*0
    aList = GenFun.DistanceFromHA(STAList, HangerStationing, STARound)
    RightSupportLoadList = GenFun.SupportLoadRight(STAList, LoadList, TensionList, HangerStationing, HangerElevation, STARound)
    LeftSupportLoadList = GenFun.SupportLoadLeft(STAList, LoadList, HangerStationing, RightSupportLoadList, STARound)
    DiscreteMomentList = GenFun.DiscreteMoment(STAList, LoadList, HangerStationing, STARound)
    for i in range(0,len(HangerStationing)-1):
        Step = (STAList >= HangerStationing[i]) * (STAList < HangerStationing[i+1])
        sag = (LeftSupportLoadList[i] * aList - DiscreteMomentList) * (1 / TensionList)
        EL += (HangerElevation[i]-sag)*Step
        j, = np.where(HangerStationing[i] == STAList)
        EL[j] = HangerElevation[i]
    EL[-1] = HangerElevation[-1]
    return EL

def ELDiff_FreeSag_NegativeLoadedHA_Simple(Cycle_Stationing, Cycle_P_SpanWeight, Cycle_H_SpanTension,
                                           Cycle_HA_STA, New_HA_EL, minCW_P, Cycle_SupportLoad_CW, STARound):
    nEL = np.copy(New_HA_EL)
    #ELStep = np.empty_like(New_HA_EL)
    ELStep = New_HA_EL*0
    STEP = Cycle_SupportLoad_CW < minCW_P
    if sum(STEP) > 0:
        HA_STA_List = Cycle_HA_STA[np.nonzero(Cycle_HA_STA*STEP)]
        for i in range(0,len(HA_STA_List)):
            HA_N, = np.where(HA_STA_List[i] == Cycle_HA_STA)
            if (HA_N == 0) or (HA_N == len(Cycle_HA_STA)):
                nEL[HA_N] = New_HA_EL[HA_N]
            else:
                D_PriorSTA = Cycle_HA_STA[HA_N-1]
                D_PriorEL = New_HA_EL[HA_N-1]
                try:
                    D_NextSTA = Cycle_HA_STA[HA_N+1]
                    D_NextEL = New_HA_EL[HA_N-1]
                except:
                    D_NextSTA = D_PriorSTA
                    D_NextEL = D_PriorEL
                D_Step = (Cycle_Stationing >= D_PriorSTA)*(Cycle_Stationing <= D_NextSTA)
                D_STAList = Cycle_Stationing[D_Step != 0]
                D_LoadList = Cycle_P_SpanWeight[D_Step != 0]
                D_H_SpanTension = Cycle_H_SpanTension[D_Step != 0]
                D_HangerStationing = np.array([D_PriorSTA,D_NextSTA])
                D_HangerElevation = np.array([D_PriorEL,D_NextEL])
                D_LoadedSag = SagSpanData(D_STAList, D_LoadList, D_H_SpanTension, D_HangerStationing, D_HangerElevation, STARound)
                try:
                    nEL[HA_N] = D_LoadedSag[(HA_STA_List[i] == D_STAList) != 0]
                except:
                    continue
        ELStep = nEL - New_HA_EL
    return ELStep

def CatenarySag_Flexible(WireRun, L_Weight, L_Tension, L_SpanLoading, L_DesignHA, L_DesignBase):
    P_DiscreteLoad_CW, STA_DiscreteLoad_CW, P_DiscreteLoad_MW, STA_DiscreteLoad_MW = [L_SpanLoading[i] for i in (0,1,2,3)]
    MaxHASpacing, minCW_P, MinHALength, HA_Accuracy = [L_DesignHA[i] for i in (0,1,2,3)]
    xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength = [L_DesignBase[i] for i in (0,1,2,3,4)]
    WR = GenFun.ScaleWireRun(WireRun, xMultiplier, yMultiplier)
    Stationing = GenFun.x_SPAN_LIST(WireRun, xStep, xRound)
    nHA = GenFun.L_HA_QTY(WR, MaxHASpacing*xMultiplier)
    HA_STA = GenFun.L_HA_STA(WR,nHA,xRound)
    HA_EL = GenFun.L_HA_EL(WR,HA_STA,L_Weight[1]*yMultiplier/xMultiplier,L_Tension[1]*xMultiplier,xRound)
    P_SpanWeight = GenFun.LoadSpanWeight(Stationing, L_Weight[1]*yMultiplier/xMultiplier)
    P_SpanWeight = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight, Stationing,
                                                 P_DiscreteLoad_CW*yMultiplier/xMultiplier, STA_DiscreteLoad_CW*xMultiplier,xRound)
    H_SpanTension = GenFun.LoadSpanTension(Stationing,L_Tension[1]*xMultiplier)
    subLoop = 1
    while subLoop < 25:
        #print('subloop =',subLoop)
        SupportLoad_CW = GenFun.SupportLoad(Stationing, P_SpanWeight, H_SpanTension, HA_STA, HA_EL, xRound)
        Cycle_MinLoad = min(SupportLoad_CW[1:-1])
        if GenFun.RoundVal(Cycle_MinLoad,0) >= GenFun.RoundVal(minCW_P,0):
            break
        ELStep = ELDiff_FreeSag_NegativeLoadedHA_Simple(Stationing, P_SpanWeight, H_SpanTension,
                                                        HA_STA, HA_EL, minCW_P, SupportLoad_CW, xRound)
        #print('Cycle_MinLoad =',Cycle_MinLoad,'l(HA_EL) =',len(HA_EL),'l(ELStep) =',len(ELStep))
        HA_EL += 1*ELStep
        subLoop += 1
    #print(subLoop)
    LoadedSag = SagSpanData(Stationing, P_SpanWeight, H_SpanTension, HA_STA, HA_EL, xRound)

    P_SpanWeight_MW = GenFun.LoadSpanWeight(Stationing, (L_Weight[0]+L_Weight[2])*yMultiplier/xMultiplier)
    P_SpanWeight_MW = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight_MW, Stationing,
                                                 SupportLoad_CW[SupportLoad_CW>0], HA_STA[SupportLoad_CW>0],xRound)
    P_SpanWeight_MW = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight_MW, Stationing,
                                                 P_DiscreteLoad_MW*yMultiplier/xMultiplier, STA_DiscreteLoad_MW*xMultiplier,xRound)
    H_SpanTension_MW = GenFun.LoadSpanTension(Stationing,L_Tension[0]*xMultiplier)
    SupportLoad_MW = GenFun.SupportLoad(Stationing, P_SpanWeight_MW, H_SpanTension_MW, np.array(WR['STA']), np.array(WR['Rail EL']+WR['MW Height']), xRound)
    LoadedSag_MW = SagSpanData(Stationing, P_SpanWeight_MW, H_SpanTension_MW, np.array(WR['STA']), np.array(WR['Rail EL']+WR['MW Height']), xRound)
    return {'Stationing': Stationing/xMultiplier, 'HA_STA': HA_STA/xMultiplier, 'HA_EL': HA_EL/xMultiplier,
            'LoadedSag': LoadedSag/yMultiplier, 'SupportLoad_CW': SupportLoad_CW*xMultiplier/yMultiplier,
            'P_SpanWeight': P_SpanWeight*xMultiplier/yMultiplier,
            'LoadedSag_MW': LoadedSag_MW/yMultiplier, 'SupportLoad_MW': SupportLoad_MW*xMultiplier/yMultiplier,
            'P_SpanWeight_MW': P_SpanWeight_MW*xMultiplier/yMultiplier}

def CatenarySag_FlexibleHA_Iterative(WireRun, L_WeightChange, L_Tension, L_DesignHA, L_DesignBase, L_StaticCWUplift, ORIGINALDESIGN):
    MaxHASpacing, minCW_P, MinHALength, HA_Accuracy = [L_DesignHA[i] for i in (0,1,2,3)]
    xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength = [L_DesignBase[i] for i in (0,1,2,3,4)]
    INPUT_StaticUplift_P, INPUT_StaticUpliftSTA = [L_StaticCWUplift[i] for i in (0,1)]
    #ORIGINALDESIGN is the return dictionary from CatenarySag
    RadialLoad = GenFun.RadialLoad(np.array(WireRun['Deviation Angle']),L_Tension[1])
    PriorCWSupportReaction = np.copy(RadialLoad*0)
    Cycle_HA_EL = np.copy(ORIGINALDESIGN.get('HA_EL'))
    Stationing = np.copy(ORIGINALDESIGN.get('Stationing'))
    PriorHALoad = np.copy(ORIGINALDESIGN.get('SupportLoad_CW'))
    HA_STA = np.copy(ORIGINALDESIGN.get('HA_STA'))
    WR = GenFun.ScaleWireRun(WireRun,xMultiplier,yMultiplier)
    H_SpanTension = GenFun.LoadSpanTension(Stationing,L_Tension[1]*xMultiplier)
    H_SpanTension_MW = GenFun.LoadSpanTension(Stationing,L_Tension[0]*xMultiplier)
    P_SpanWeightDiff = GenFun.LoadSpanWeight(Stationing,L_WeightChange[1]*yMultiplier/xMultiplier)
    P_SpanWeightDiff_MW = GenFun.LoadSpanWeight(Stationing,(L_WeightChange[0]+L_WeightChange[2])*yMultiplier/xMultiplier)
    P_SpanWeight = ORIGINALDESIGN.get('P_SpanWeight') + P_SpanWeightDiff
    P_SpanWeight_MW = ORIGINALDESIGN.get('P_SpanWeight_MW') + P_SpanWeightDiff_MW
    P_SpanWeight = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight, Stationing,
                                                 INPUT_StaticUplift_P*yMultiplier/xMultiplier, INPUT_StaticUpliftSTA*xMultiplier,xRound)
    P_SpanWeight_MW = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight_MW, Stationing,
                                                 -PriorHALoad[PriorHALoad>=minCW_P]*yMultiplier/xMultiplier,
                                                 HA_STA[PriorHALoad>=minCW_P]*xMultiplier,xRound)

    TsubLoop1=0
    TsubLoop2=0
    loops = 0
    while loops < 25:
        #Resolve to the correct solution for Resetting Forces at Supports
        subLoop1 = 0
        while subLoop1 < 25:
            Cycle_P_SpanWeight = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight, Stationing,
                                                               PriorCWSupportReaction*yMultiplier/xMultiplier,
                                                               np.array(WR['STA']),xRound)
            Cycle_LoadedSag = SagSpanData(Stationing, Cycle_P_SpanWeight, H_SpanTension, HA_STA, Cycle_HA_EL, xRound)
            Cycle_CWELDiff, Cycle_CWELDiffSTA = GenFun.CWSupportELDifference(Stationing, ORIGINALDESIGN.get('LoadedSag'),
                                                          Cycle_LoadedSag, np.array(WR['STA']))
            Cycle_CWSupportReaction = GenFun.SteadyArm_VerticalResettingForce(RadialLoad, np.array(WR['STA']), Cycle_CWELDiff/yMultiplier, Cycle_CWELDiffSTA, SteadyArmLength)
            Cycle_CWSupportReaction = Cycle_CWSupportReaction*yMultiplier/xMultiplier
            LoadDiff = Cycle_CWSupportReaction-PriorCWSupportReaction
            if max(np.nanmax(LoadDiff),abs(np.nanmin(LoadDiff))) <= loadTolerance:
                break
            PriorCWSupportReaction += 0.8*LoadDiff
            subLoop1 +=1
        TsubLoop1 += subLoop1
        #Resolve to the correct solution for Hangers without downforce
        subLoop2 = 0
        while subLoop2 < 25:
            Cycle_SupportLoad_CW = GenFun.SupportLoad(Stationing, Cycle_P_SpanWeight, H_SpanTension,
                                                      HA_STA, Cycle_HA_EL, xRound)
            Cycle_MinLoad = np.nanmin(Cycle_SupportLoad_CW[1:-1])
            if Cycle_MinLoad >= minCW_P:
                break
            #Hangers with down force cannot occur with flexible hangers, therefore the CW height is increased slightly at the HA
            #location until it reaches a point of equilibrium and is not/slightly loaded
            ELStep = ELDiff_FreeSag_NegativeLoadedHA_Simple(Stationing, Cycle_P_SpanWeight, H_SpanTension,
                                                                   HA_STA, Cycle_HA_EL, minCW_P, Cycle_SupportLoad_CW, xRound)
            Cycle_HA_EL += 0.8*ELStep
            subLoop2 += 1
        TsubLoop2 += subLoop2
        Cycle_P_SpanWeight_MW = GenFun.AddDiscreteLoadstoSpan(P_SpanWeight_MW, Stationing,
                                                              Cycle_SupportLoad_CW[Cycle_SupportLoad_CW>minCW_P],
                                                              HA_STA[Cycle_SupportLoad_CW>minCW_P], xRound)
        Cycle_LoadedSag_MW = SagSpanData(Stationing, Cycle_P_SpanWeight_MW, H_SpanTension_MW, np.array(WR['STA']),
                                         np.array(WR['Rail EL']+WR['MW Height']), xRound)
        #Resolve to the correct solution for Hanger lengths. They are allowed to shrink and unload, but hanger lengths cannot get bigger.
        #CW HA elevations need to be adjusted so they do not increase in length.
        HA_L_Diff = GenFun.HangerLengthDifference(Stationing, ORIGINALDESIGN.get('LoadedSag'), ORIGINALDESIGN.get('LoadedSag_MW'),
                                                  Cycle_LoadedSag, Cycle_LoadedSag_MW, HA_STA)
        HA_L_Diff_grown = HA_L_Diff*(HA_L_Diff>0)
        HA_L_Diff_shrunk = HA_L_Diff*(Cycle_SupportLoad_CW >= 0)*(HA_L_Diff<0)
        if max(np.max(HA_L_Diff_grown),abs(np.min(HA_L_Diff_shrunk))) <= HA_Accuracy:
            if subLoop1 == 0 and subLoop2 == 0:
                break
        #Since HA's are allowed to slack, stepping up slowly to an unloaded state without overshooting resolves the quickest
        Cycle_HA_EL += 0.8*HA_L_Diff*(HA_L_Diff>0)
        #Slacking of hangers (shortening) is only allowed if the force is in uplift
        #HA_EL_Diff = GenFun.CWSupportELDifference(Stationing, ORIGINALDESIGN.get('LoadedSag'), Cycle_LoadedSag, HA_STA)
        Cycle_HA_EL += 0.8*HA_L_Diff*(HA_L_Diff<0)*(Cycle_SupportLoad_CW>0)
        loops += 1
    Cycle_SupportLoad_MW = GenFun.SupportLoad(Stationing, Cycle_P_SpanWeight_MW, H_SpanTension_MW,
                                              np.array(WR['STA']), np.array((WR['Rail EL']+WR['MW Height'])), xRound)
    TotalLoops = loops*10000+TsubLoop2*100+TsubLoop1
    #print('Main Loops=', loops, 'subLoop2=', TsubLoop2, 'subLoop1=', TsubLoop1)
    return {'Stationing': Stationing/xMultiplier, 'HA_STA': HA_STA/xMultiplier, 'HA_EL': Cycle_HA_EL/xMultiplier,
            'LoadedSag': Cycle_LoadedSag/yMultiplier, 'SupportLoad_CW': Cycle_SupportLoad_CW*xMultiplier/yMultiplier,
            'P_SpanWeight': Cycle_P_SpanWeight*xMultiplier/yMultiplier,
            'LoadedSag_MW': Cycle_LoadedSag_MW/yMultiplier, 'SupportLoad_MW': Cycle_SupportLoad_MW*xMultiplier/yMultiplier,
            'P_SpanWeight_MW': Cycle_P_SpanWeight_MW*xMultiplier/yMultiplier}, TotalLoops, Cycle_CWSupportReaction*xMultiplier/yMultiplier

# def CatenaryElasticity_FlexibleHA(WireRun, L_WeightDiff, L_Tension, L_DesignHA, L_DesignBase, UpliftLoad,
#                                   LoadSteps, ORIGINALDESIGN, StartSPT, EndSPT):
#     #ORIGINALDESIGN is the return dictionary from CatenarySag
#     StartSTA = WireRun['STA'].iloc[StartSPT]
#     EndSTA = WireRun['STA'].iloc[EndSPT]
#     STAVal = np.arange(StartSTA, EndSTA, LoadSteps, dtype=float)
#     print('Checking elasticity from STA', StartSTA, 'to STA', EndSTA)
#     print(' | ', int(len(STAVal)), 'total cycles')
#     DiffMIN = np.copy(STAVal*0)
#     DiffMAX = np.copy(STAVal*0)
#     CycleLoops = np.copy(STAVal*0)
#     i = 0
#     for UpliftSTA in STAVal:
#         print(i, ' | calculating uplift at STA =', UpliftSTA)
#         L_StaticCWUplift = np.array([-UpliftLoad, UpliftSTA])
#         CYCLE, cloops, NewCWSupportReaction = CatenarySag_FlexibleHA_Iterative(WireRun, L_WeightDiff, L_Tension, L_DesignHA,
#                                                                                L_DesignBase, L_StaticCWUplift, ORIGINALDESIGN)
#         EL_DIFF = CYCLE.get('LoadedSag') - ORIGINALDESIGN.get('LoadedSag')
#         DiffMIN[i] = np.nanmin(EL_DIFF)
#         DiffMAX[i] = np.nanmax(EL_DIFF)
#         CycleLoops = cloops
#         i += 1
#     return {'STAVal': STAVal, 'DiffMIN': DiffMIN, 'DiffMAX': DiffMAX, 'CycleLoops': CycleLoops}
