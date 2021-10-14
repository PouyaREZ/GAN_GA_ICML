# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 2020


@Author: Author

____________________________________________________
Plots to produce:
1. LCC of equipment for each scenario for all the individuals
2, GHG of equipment for each scenario for all the individuals

3. GHG vs LCC scatter plot.

4. GHG vs chiller type
5. GHG vs CHP type,
6. LCC vs chiller type
7. GHG vs CHP type

8. Traces of building types across all the runs
____________________________________________________

"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys







# Constants
max_site_area = 10**6
max_FAR = 5
max_gfa = max_site_area*max_FAR




# Constants for plotting the filtered plots
# LCC_Cutoff = 10 # float('inf') # k$/m2
CO2_Cutoff = float('inf') # T-CO2/m2




# Constants for evaluating the solutions
# Num_Sites = 4
Num_Buildings = 21
Type_Max = 7
# len_of_indiv = 11 # After adding the plant location


LCC_Var = Num_Buildings+5
CO2_Var = Num_Buildings+6
LCC_w_o_EV_from_Grid_Var = Num_Buildings+7
# WalkScore_Var = Num_Sites+15
GFA_Var = Num_Buildings+7+Type_Max+3
FAR_Var = Num_Buildings+7+Type_Max+1
EV_Ratio_Var = Num_Buildings+3


CHPType_Var = Num_Buildings
ChillerType_Var = Num_Buildings+1


bldg_types = ['Res','Off','Ret','Sup','Rest','Edu','Med','Lod','Ind']


verbose = True






## CHP/Chiller/Solar Types used in the individual neighborhood
CHP_Types = {}
CHP_Types[1] = 'Gas1'
CHP_Types[2] = 'Gas2'
CHP_Types[3] = 'Gas3'
CHP_Types[4] = 'Gas4'
CHP_Types[5] = 'Gas5'
CHP_Types[6] = 'Micro1'
CHP_Types[7] = 'Micro2'
CHP_Types[8] = 'Micro3'
CHP_Types[9] = 'Recipro1'
CHP_Types[10] = 'Recipro2'
CHP_Types[11] = 'Recipro3'
CHP_Types[12] = 'Recipro4'
CHP_Types[13] = 'Recipro5'
CHP_Types[14] = 'Steam1'
CHP_Types[15] = 'Steam2'
CHP_Types[16] = 'Steam3'
CHP_Types[17] = 'FC1'
CHP_Types[18] = 'FC2'
CHP_Types[19] = 'FC3'
CHP_Types[20] = 'FC4'
CHP_Types[21] = 'FC5'
CHP_Types[22] = 'FC6'
CHP_Types[23] = 'Bio1'
CHP_Types[24] = 'Bio2'
CHP_Types[25] = 'Bio3'
CHP_Types[26] = 'Bio4'
CHP_Types[27] = 'Bio5'
CHP_Types[28] = 'Bio6'
CHP_Types[29] = 'Bio7'
CHP_Types[30] = 'Bio8'
CHP_Types[31] = 'Bio9'
CHP_Types[32] = 'Bio10'


Chiller_Types = {}
Chiller_Types[1] = 'EC1'
Chiller_Types[2] = 'EC2'
Chiller_Types[3] = 'EC3'
Chiller_Types[4] = 'EC4'
Chiller_Types[5] = 'EC5'
Chiller_Types[6] = 'EC6'
Chiller_Types[7] = 'EC7'
Chiller_Types[8] = 'EC8'
Chiller_Types[9] = 'EC9'
Chiller_Types[10] = 'AC1'
Chiller_Types[11] = 'AC2'
Chiller_Types[12] = 'AC3'
Chiller_Types[13] = 'AC4'
Chiller_Types[14] = 'AC5'
Chiller_Types[15] = 'AC6'
Chiller_Types[16] = 'AC7'
Chiller_Types[17] = 'AC8'










def DF_Filter(filename):    
    inputDF = pd.read_csv(filename, header=None, index_col=None)
    
    error_tol = 1.15
    
#    print('GFA stats:')
#    print(inputDF.iloc[:,38].describe())
    print('+++++ processing %s +++++\n'%(filename))
    
    print('Count duplicates:')
    condition1 = inputDF.duplicated() == True
    print(inputDF[condition1][GFA_Var].count())
    
    
    print('Count under the min GFA:') # Count non-trivial neighborhoods
    condition2 = inputDF[GFA_Var] <= 1/error_tol#<=647497/10
    print(inputDF[condition2][GFA_Var].count())
    
    
    print('Count over the max GFA:')
    condition3 = inputDF[GFA_Var] >= max_gfa*error_tol
    print(inputDF[condition3][GFA_Var].count())
    
    
    print('Count over the max Site GFA:')
    condition4 = inputDF[GFA_Var]/inputDF[FAR_Var] >= max_site_area*error_tol
    print(inputDF[condition4][GFA_Var].count())
    
    
    print('Count valid answers:')
    print(len(inputDF) - inputDF[condition1 | condition2 | condition3 | condition4][GFA_Var].count())
    
#    print('------------------')
    # Filtering the inadmissible results
    Filtered = ~(condition1 | condition2 | condition3 | condition4)
    inputDF = inputDF[Filtered]
    inputDF.reset_index(inplace=True, drop=True)
    
#    print('Annual energy demand stats (MWh):')
    inputDF[LCC_Var] /= inputDF[GFA_Var] # Normalizing LCC ($/m2)
    inputDF[LCC_w_o_EV_from_Grid_Var] /= inputDF[GFA_Var] # Normalizing LCC on Grid
    inputDF[CO2_Var] /= inputDF[GFA_Var] # Normalizing GHG ($/m2)
    # inputDF[40] /= (10**3*inputDF[GFA_Var]) # Normalizing total energy demand (MWh/m2)
    # inputDF[41] /= inputDF[GFA_Var] # Normalizing total wwater treatment demand (L/m2)
    for i in range(Num_Buildings+8,Num_Buildings+8+Type_Max): # Converting percent areas to integer %
        inputDF[i] = inputDF[i] * 100
#    print(inputDF[40].describe())
    
    return inputDF
    


### MAIN FUNCTION
def main():
    nameOfFile = sys.argv[1]
    LCC_Cutoff = float(sys.argv[2])
    
    print('loading data')
    # filenames = ['../Results/Results.csv']
    filenames = ['../Results/' + nameOfFile + '.csv']
    DFNames = [nameOfFile]
    DFs = {}
    for i in range(len(filenames)):
        DFs[DFNames[i]] = DF_Filter(filenames[i])
    
    
    
    
    # plt.style.use('ggplot') ## TODO
    colors_rb = {DFNames[0]:'black'} # {DFNames[0]:'r', DFNames[1]:'b'}
    
    
    
    
    ## CHP, Chiller and WWT name assignments
    # CHP = {}
    # Chiller = {}
    # WWT = {}
    for DFName in DFNames:
        # CHP[DFName] = np.array([CHP_Types[int(i)] for i in DFs[DFName][21]]) # Making strings of CHP names instead of integers
        DFs[DFName][21] = np.array([CHP_Types[int(i)] for i in DFs[DFName][21]]) # Making strings of CHP names instead of integers
        # Chiller[DFName] = np.array([Chiller_Types[int(i)] for i in DFs[DFName][22]]) # Making strings of Chiller names instead of integers
        DFs[DFName][22] = np.array([Chiller_Types[int(i)] for i in DFs[DFName][22]]) # Making strings of Chiller names instead of integers
        # WWT[DFName] = np.array([WWT_Types[int(i)] for i in DFs[DFName][24]]) # Making strings of WWT module names instead of integers
        # DFs[DFName][24] = np.array([WWT_Types[int(i)] for i in DFs[DFName][24]]) # Making strings of WWT module names instead of integers
    
    
    # =============================================================================
    
    
    
    
    
    ######################## PLOTS ##########################
    
    
    
    
    if verbose: print('\nPlotting!\n')
    
    
    
    # LCC vs GHG for the filtered data
    for DFName in DFNames:
        DF = DFs[DFName][DFs[DFName][LCC_Var]/10**3 < LCC_Cutoff]
        # LCC vs CO2
        if verbose: print('Plotting LCC vs GHG')
        plt.figure(figsize=(10,5))
        rgba_colors = list(zip([0]*len(DF), [0]*len(DF), [0]*len(DF), list(np.minimum(DF[EV_Ratio_Var]+30, 100)/100))) # R, G, B, A (black with alphas proportional to EV load ratio)
        plt.scatter(x=DF[LCC_Var]/10**3,y=DF[CO2_Var], label=DFName, s=1, color=rgba_colors)#, marker='x')
        # plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
        
        
        plt.xlabel(r'LCC (k\$/$m^2$)')
        plt.ylabel(r'GHG (T-$CO_{2e}$/$m^2$)')
        # plt.legend()
        plt.title('GHG vs LCC_the more transparent, the closer to 0% EV load on CCHP')
        plt.savefig('GHG_vs_LCC'+DFNames[0]+'.png', dpi=400, bbox_inches='tight')
    
    
    
    
    
    # LCC vs EV Load Ratio
    for DFName in DFNames:
        DF = DFs[DFName][DFs[DFName][LCC_Var]/10**3 < LCC_Cutoff]
        # LCC vs CO2
        if verbose: print('Plotting LCC vs EV Load Ratio')
        plt.figure(figsize=(10,5))
        # rgba_colors = list(zip([0]*len(DF), [0]*len(DF), [0]*len(DF), list(DF[EV_Ratio_Var]/100))) # R, G, B, A (black with alphas proportional to EV load ratio)
        plt.scatter(x=DF[LCC_Var]/10**3,y=DF[EV_Ratio_Var], label=DFName, s=1, color='black')#, marker='x')
        # plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
        
        
        plt.xlabel(r'LCC (k\$/$m^2$)')
        plt.ylabel(r'EV Load Ratio on CCHP (%)')
        # plt.legend()
        # plt.title('GHG vs LCC_the more transparent, the closer to 0% EV load on CCHP')
        plt.savefig('LCC_vs_EVLoadRatio'+DFNames[0]+'.png', dpi=400, bbox_inches='tight')
    
    
    
    # GHG vs EV Load Ratio
    for DFName in DFNames:
        DF = DFs[DFName][DFs[DFName][CO2_Var]/10**3 < LCC_Cutoff]
        # LCC vs CO2
        if verbose: print('Plotting GHG vs EV Load Ratio')
        plt.figure(figsize=(10,5))
        # rgba_colors = list(zip([0]*len(DF), [0]*len(DF), [0]*len(DF), list(DF[EV_Ratio_Var]/100))) # R, G, B, A (black with alphas proportional to EV load ratio)
        plt.scatter(x=DF[CO2_Var],y=DF[EV_Ratio_Var], label=DFName, s=1, color='black')#, marker='x')
        # plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
        
        
        plt.xlabel(r'GHG (T-$CO_{2e}$/$m^2$)')
        plt.ylabel(r'EV Load Ratio on CCHP (%)')
        # plt.legend()
        # plt.title('GHG vs LCC_the more transparent, the closer to 0% EV load on CCHP')
        plt.savefig('GHG_vs_EVLoadRatio'+DFNames[0]+'.png', dpi=400, bbox_inches='tight')
    
    
    
    
    
    
    # 3d plot GHG vs LCC vs EV Load 
    for DFName in DFNames:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    
        DF = DFs[DFName][DFs[DFName][LCC_Var]/10**3 < LCC_Cutoff]
        # LCC vs CO2
        if verbose: print('Plotting LCC vs GHG vs EV Load Ratio')
        plt.figure(figsize=(10,5))
        # rgba_colors = list(zip([0]*len(DF), [0]*len(DF), [0]*len(DF), list(DF[EV_Ratio_Var]/100))) # R, G, B, A (black with alphas proportional to EV load ratio)
        ax.scatter(DF[LCC_Var]/10**3, DF[CO2_Var], DF[EV_Ratio_Var], s=1)
        
        # plt.scatter(x=DF[LCC_Var]/10**3,y=DF[CO2_Var], label=DFName, s=1, color=rgba_colors)#, marker='x')
        # plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
        
        ax.set_xlabel(r'LCC (k\$/$m^2$)')
        ax.set_ylabel(r'GHG (T-$CO_{2e}$/$m^2$)')
        ax.set_zlabel('EV Load Ratio')
        # plt.legend()
        plt.title('GHG vs LCC vs EV Load Ratio')
        fig.savefig('GHG_vs_LCC_vs_EVLoadRatio'+DFNames[0]+'.png', dpi=400, bbox_inches='tight')
    
    
    
    
    
    # 3d histogram of LCC and GHG
    for DFName in DFNames:
        if verbose: print('Plotting LCC vs GHG Histogram')
        
        plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111, projection='3d')
        
    
        DF = DFs[DFName][DFs[DFName][LCC_Var]/10**3 < LCC_Cutoff]
        
        hist, xedges, yedges = np.histogram2d(DF[LCC_Var]/10**3, DF[CO2_Var], bins=10)
        
        
        # Construct arrays for the anchor positions of the 16 bars.
        delta_x = 0.5*(xedges[1] - xedges[0])
        delta_y = 0.5*(yedges[1] - yedges[0])
        xpos, ypos = np.meshgrid(xedges[:-1] + delta_x, yedges[:-1] + delta_y, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        
        # Construct arrays with the dimensions for the 16 bars.
        dx = delta_x * np.ones_like(zpos)
        dy = delta_y * np.ones_like(zpos)
        dz = hist.ravel()
        
        ax.set_xlabel(r'LCC (k\$/$m^2$)')
        ax.set_ylabel(r'GHG (T-$CO_{2e}$/$m^2$)')
        ax.set_zlabel('Count')
        
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        
    
        # plt.legend()
        plt.title('2dHist_GHG vs LCC')
        fig.savefig('2D_Hist_GHG_vs_LCC'+DFNames[0]+'.png', dpi=400, bbox_inches='tight')
        
        
    
    
    
    
    
    exit()
    
    
    
    
    
    
    
    
    #############################################
    
    print('plotting overall LCC and GHG graphs')
    # LCC
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=26, ascending=True).reset_index(drop=True)
        plt.scatter(x=sortedDF.index,y=(sortedDF[LCC_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel('Rank')
    plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.title('LCC')
    plt.legend()
    plt.savefig('LCC_Ascending.png', dpi=400, bbox_inches='tight')
    
    
    
    # GHG
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=27, ascending=True).reset_index(drop=True)
        plt.scatter(x=sortedDF.index,y=(sortedDF[CO2_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel('Rank')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    # plt.title('GHG')
    plt.legend()
    plt.savefig('GHG_Ascending.png', dpi=400, bbox_inches='tight')
    
    plt.close('all')
    
    
    
    #############################################
    print('plotting LCC and GHG box plots')
    
    print('\n#############################################')
    print('Stats of LCC ($/m2) for Disintegrated Case:\n',(DFs[DFNames[0]][26]).describe())
    print('Stats of LCC ($/m2) for Integrated Case:\n',(DFs[DFNames[1]][26]).describe())
    print('Stats of GHG ($/m2) for Disintegrated Case:\n',(DFs[DFNames[0]][27]).describe())
    print('Stats of GHG ($/m2) for Integrated Case:\n',(DFs[DFNames[1]][27]).describe())
    print('#############################################\n')
    
    # =============================================================================
    # # LCC
    # plt.figure(figsize=(10,5))
    # # for DFName in DFNames:
    # plt.boxplot(x=[(DFs[DFNames[0]][26]/10**3), (DFs[DFNames[1]][26]/10**3)])
    # #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    # # plt.xlabel('Rank')
    # plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.xticks([1,2],[DFNames[0],DFNames[1]])
    # # plt.title('LCC')
    # plt.savefig('LCC_Boxplot.png', dpi=400, bbox_inches='tight')
    # 
    # 
    # 
    # # GHG
    # plt.figure(figsize=(10,5))
    # # for DFName in DFNames:
    # plt.boxplot(x=[(DFs[DFNames[0]][27]/10**3), (DFs[DFNames[1]][27]/10**3)])
    # #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    # # plt.xlabel('Rank')
    # plt.ylabel(r'GHG (k\$/$m^2$)')
    # plt.xticks([1,2],[DFNames[0],DFNames[1]])
    # # plt.title('LCC')
    # plt.savefig('GHG_Boxplot.png', dpi=400, bbox_inches='tight')
    # 
    # plt.close('all')
    # =============================================================================
    
    
    '''
    #############################################
    print('plotting LCC/GHG vs total neighborhood energy and ww graphs')
    
    print('\n#############################################')
    print('Stats of Total Energy Demand (MWh/m2) for Disintegrated Case:\n',(DFs[DFNames[0]][40]).describe())
    print('Stats of Total Energy Demand (MWh/m2) for Integrated Case:\n',(DFs[DFNames[1]][40]).describe())
    print('Stats of Total Wastewater Treatment Demand (m3/m2) for Disintegrated Case:\n',(DFs[DFNames[0]][41]/10**3).describe())
    print('Stats of Total Wastewater Treatment Demand (m3/m2) for Integrated Case:\n',(DFs[DFNames[1]][41]/10**3).describe())
    print('#############################################\n')
    
    # LCC vs Neighborhood's Total Energy Use
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
        plt.scatter(x=(sortedDF[40]),y=(sortedDF[LCC_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.title('LCC')
    plt.legend()
    plt.savefig('LCC_vs_Energy_Demand.png', dpi=400, bbox_inches='tight')
    
    
    # LCC vs Neighborhood's Total WWater Demand
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
        plt.scatter(x=(sortedDF[41]/10**3),y=(sortedDF[LCC_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.title('LCC')
    plt.legend()
    plt.savefig('LCC_vs_WWater_Demand.png', dpi=400, bbox_inches='tight')
    
    
    
    # GHG vs Neighborhood's Total Energy Use
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
        plt.scatter(x=(sortedDF[40]),y=(sortedDF[CO2_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    # plt.title('LCC')
    plt.legend()
    plt.savefig('GHG_vs_Energy_Demand.png', dpi=400, bbox_inches='tight')
    
    
    # GHG vs Neighborhood's Total WWater Demand
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
        plt.scatter(x=(sortedDF[41]/10**3),y=(sortedDF[CO2_Var]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    # plt.title('LCC')
    plt.legend()
    plt.savefig('GHG_vs_WWater_Demand.png', dpi=400, bbox_inches='tight')
    
    plt.close('all')
    
    #############################################
    
    print('plotting building mix vs neighborhood energy and ww graphs')
    
    # Building Mix vs Neighborhood's Total WWater Demand (integrated)
    DFName = 'CCHP+WWT'
    bldg_types = ['Res','Off','Com','Ind','Hos','Med','Edu']
    colors = ['m','b','c','g','y','orange','r']
    columns = list(range(29,36))
    plt.figure(figsize=(10,5))
    sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
    for i in range(len(bldg_types)):
        plt.scatter(x=(sortedDF[41]/10**3),y=DFs[DFName].iloc[:,columns[i]],
                    s=0.5, label=bldg_types[i], c=colors[i], alpha=0.5)
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.ylabel('Percent of Total GFA (%)')
    plt.ylim(0, 100)
    plt.xlim(0,11)
    # plt.title('LCC')
    plt.legend()
    plt.savefig('Bldg_Mix_vs_WWater_Demand_Integ.png', dpi=400, bbox_inches='tight')
    
    
    
    # Building Mix vs Neighborhood's Total WWater Demand (Disintegrated)
    DFName = 'CCHP|CWWTP'
    bldg_types = ['Res','Off','Com','Ind','Hos','Med','Edu']
    colors = ['m','b','c','g','y','orange','r']
    columns = list(range(29,36))
    plt.figure(figsize=(10,5))
    sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
    for i in range(len(bldg_types)):
        plt.scatter(x=(sortedDF[41]/10**3),y=DFs[DFName].iloc[:,columns[i]],
                    s=0.5, label=bldg_types[i], c=colors[i], alpha=0.5)
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.ylabel('Percent of Total GFA (%)')
    # plt.title('LCC')
    plt.ylim(0, 100)
    plt.xlim(0,11)
    plt.legend()
    plt.savefig('Bldg_Mix_vs_WWater_Demand_Disinteg.png', dpi=400, bbox_inches='tight')
    
    
    
    
    # Building Mix vs Neighborhood's Total Energy Demand (integrated)
    DFName = 'CCHP+WWT'
    bldg_types = ['Res','Off','Com','Ind','Hos','Med','Edu']
    colors = ['m','b','c','g','y','orange','r']
    columns = list(range(29,36))
    plt.figure(figsize=(10,5))
    sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
    for i in range(len(bldg_types)):
        plt.scatter(x=(sortedDF[40]),y=DFs[DFName].iloc[:,columns[i]],
                    s=0.5, label=bldg_types[i], c=colors[i], alpha=0.5)
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.ylabel('Percent of Total GFA (%)')
    # plt.title('LCC')
    plt.ylim(0, 100)
    plt.xlim(0,1)
    plt.legend()
    plt.savefig('Bldg_Mix_vs_Energy_Demand_Integ.png', dpi=400, bbox_inches='tight')
    
    
    
    # Building Mix vs Neighborhood's Total Energy Demand (Disintegrated)
    DFName = 'CCHP|CWWTP'
    bldg_types = ['Res','Off','Com','Ind','Hos','Med','Edu']
    colors = ['m','b','c','g','y','orange','r']
    columns = list(range(29,36))
    plt.figure(figsize=(10,5))
    sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
    for i in range(len(bldg_types)):
        plt.scatter(x=(sortedDF[40]),y=DFs[DFName].iloc[:,columns[i]],
                    s=0.5, label=bldg_types[i], c=colors[i], alpha=0.5)
    #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    plt.xlabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.ylabel('Percent of Total GFA (%)')
    # plt.title('LCC')
    plt.ylim(0, 100)
    plt.xlim(0,1)
    plt.legend()
    plt.savefig('Bldg_Mix_vs_Energy_Demand_Disinteg.png', dpi=400, bbox_inches='tight')
    
    plt.close('all')
    
    #############################################
    print('plotting Supply type vs total neighborhood energy and ww graphs')
    
    # Total Energy Demand vs CHP
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
        plt.scatter(x=DFs[DFName][21],y=(sortedDF[40]),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'CHP Type')
    plt.ylabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.legend()
    plt.savefig('Total_Energy_vs_CHP.png', dpi=400, bbox_inches='tight')
    
    
    # Total WWater Demand vs CHP
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
        plt.scatter(x=DFs[DFName][21],y=(sortedDF[41]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'CHP Type')
    plt.ylabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.legend()
    plt.savefig('Total_WWater_vs_CHP.png', dpi=400, bbox_inches='tight')
    
    
    # Total Energy Demand vs Chiller
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
        plt.scatter(x=DFs[DFName][22],y=(sortedDF[40]),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'Chiller Type')
    plt.ylabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.legend()
    plt.savefig('Total_Energy_vs_Chiller.png', dpi=400, bbox_inches='tight')
    
    
    # Total WWater Demand vs Chiller
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
        plt.scatter(x=DFs[DFName][22],y=(sortedDF[41]/10**3),label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'Chiller Type')
    plt.ylabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.legend()
    plt.savefig('Total_WWater_vs_Chiller.png', dpi=400, bbox_inches='tight')
    
    
    # Total Energy Demand vs WWT (integrated)
    plt.figure(figsize=(10,5))
    DFName = 'CCHP+WWT'
    sortedDF = DFs[DFName].sort_values(by=40, ascending=True).reset_index(drop=True)
    plt.scatter(x=DFs[DFName][24],y=(sortedDF[40]),s=2, c=colors_rb[DFName])
    plt.xlabel(r'WWT Type')
    plt.ylabel(r'Total Energy Demand (MWh/$m^2$)')
    plt.legend()
    plt.savefig('Total_Energy_vs_WWT_Integ.png', dpi=400, bbox_inches='tight')
    
    
    # Total WWater Demand vs WWT (integrated)
    plt.figure(figsize=(10,5))
    DFName = 'CCHP+WWT'
    sortedDF = DFs[DFName].sort_values(by=41, ascending=True).reset_index(drop=True)
    plt.scatter(x=DFs[DFName][24],y=(sortedDF[41]/10**3), s=2, c=colors_rb[DFName])
    plt.xlabel(r'WWT Type')
    plt.ylabel(r'Total Wastewater Treatment Demand ($m^3$/$m^2$)')
    plt.savefig('Total_Wwater_vs_WWT_Integ.png', dpi=400, bbox_inches='tight')
    '''
    plt.close('all')
    
    #############################################
    print('plotting pareto fronts')
    
    # LCC vs CO2
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        plt.scatter(x=DFs[DFName][26]/10**3,y=DFs[DFName][39],label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'LCC (k\$/$m^2$)')
    plt.ylabel(r'Lifecycle $CO_{2e}$ (T/$m^2$)')
    plt.legend()
    plt.savefig('CO2_vs_LCC.png', dpi=400, bbox_inches='tight')
    
    
    
    
    #############################################
    
    # LCC vs GHG
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        plt.scatter(x=DFs[DFName][26]/10**3,y=DFs[DFName][27]/10**3,label=DFName, s=2, alpha=0.5, c=colors_rb[DFName])
    plt.xlabel(r'LCC (k\$/$m^2$)')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    plt.legend()
    plt.savefig('GHG_vs_LCC.png', dpi=400, bbox_inches='tight')
    
    
    # LCC vs GHG w Generation-based transparency
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        alphas = np.linspace(0.1, 1, len(DFs[DFName]))
        rgba_colors = np.zeros((len(DFs[DFName]),4))
        if DFName == DFNames[0]:
            rgba_colors[:,0] = 1.0 # red
        else:
            rgba_colors[:,2] = 1.0 # blue
        rgba_colors[:,3] = alphas
        plt.scatter(x=DFs[DFName][26]/10**3,y=DFs[DFName][27]/10**3,label=DFName, s=1, c=rgba_colors)
    plt.xlabel(r'LCC (k\$/$m^2$)')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    plt.legend()
    plt.savefig('GHG_vs_LCC_Gen_Colorcoded.png', dpi=400, bbox_inches='tight')
    
    
    # LCC vs GHG w Generation-based transparency and elite-filtered
    plt.figure(figsize=(10,5))
    for DFName in DFNames:
        DF = DFs[DFName][DFs[DFName][26]/10**3 <= 500]
        DF = DF[DFs[DFName][27]/10**3 <= 0.1]
        alphas = np.linspace(0.1, 1, len(DF))
        rgba_colors = np.zeros((len(DF),4))
        if DFName == DFNames[0]:
            rgba_colors[:,0] = 1.0 # red
        else:
            rgba_colors[:,2] = 1.0 # blue
        rgba_colors[:,3] = alphas
        plt.scatter(x=DF[LCC_Var]/10**3,y=DF[CO2_Var]/10**3,label=DFName, s=1, c=rgba_colors)
    plt.xlabel(r'LCC (k\$/$m^2$)')
    plt.ylabel(r'GHG (k\$/$m^2$)')
    plt.legend()
    plt.savefig('GHG_vs_LCC_Gen_Colorcoded_Filtered.png', dpi=400, bbox_inches='tight')
    
    
    # =============================================================================
    # # LCC vs GHG (integrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP+WWT'
    # plt.scatter(x=DFs[DFName][26]/10**3,y=DFs[DFName][27]/10**3, s=2)
    # plt.xlabel(r'LCC (k\$/$m^2$)')
    # plt.ylabel(r'GHG (k\$/$m^2$)')
    # plt.savefig('GHG_vs_LCC_Integ.png', dpi=400, bbox_inches='tight')
    # 
    # 
    # # LCC vs GHG (disintegrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP|CWWTP'
    # plt.scatter(x=DFs[DFName][26]/10**3,y=DFs[DFName][27]/10**3, s=2)
    # #    (DFs[DFName][0][26]/10**6).plot(label=DFName)
    # plt.xlabel(r'LCC (k\$/$m^2$)')
    # plt.ylabel(r'GHG (k\$/$m^2$)')
    # # plt.title('LCC')
    # plt.savefig('GHG_vs_LCC_Disinteg.png', dpi=400, bbox_inches='tight')
    # 
    # =============================================================================
    
    #############################################
    print('plotting Supply type vs opt objectives')
    
    
    print('\n#############################################')
    Disinteg_Grpd_by_CHP_meanLCC = DFs[DFNames[0]].groupby(21)[26].mean()
    Disnteg_Grpd_by_CHP_medLCC = DFs[DFNames[0]].groupby(21)[26].median()
    Disnteg_Grpd_by_CHP_meanGHG = DFs[DFNames[0]].groupby(21)[27].mean()
    Disnteg_Grpd_by_CHP_medGHG = DFs[DFNames[0]].groupby(21)[27].median()
    Integ_Grpd_by_CHP_meanLCC = DFs[DFNames[1]].groupby(21)[26].mean()
    Integ_Grpd_by_CHP_medLCC = DFs[DFNames[1]].groupby(21)[26].median()
    Integ_Grpd_by_CHP_meanGHG = DFs[DFNames[1]].groupby(21)[27].mean()
    Integ_Grpd_by_CHP_medGHG = DFs[DFNames[1]].groupby(21)[27].median()
    items = [Disinteg_Grpd_by_CHP_meanLCC, Disnteg_Grpd_by_CHP_medLCC, Disnteg_Grpd_by_CHP_meanGHG,
     Disnteg_Grpd_by_CHP_medGHG, Integ_Grpd_by_CHP_meanLCC, Integ_Grpd_by_CHP_medLCC,
     Integ_Grpd_by_CHP_meanGHG, Integ_Grpd_by_CHP_medGHG]
    items_names = ['Disinteg_Grpd_by_CHP_meanLCC', 'Disnteg_Grpd_by_CHP_medLCC', 'Disnteg_Grpd_by_CHP_meanGHG',
     'Disnteg_Grpd_by_CHP_medGHG', 'Integ_Grpd_by_CHP_meanLCC', 'Integ_Grpd_by_CHP_medLCC',
     'Integ_Grpd_by_CHP_meanGHG', 'Integ_Grpd_by_CHP_medGHG']
    for i in range(len(items)):
        print(items_names[i], items[i])
    print('#############################################\n')
    
    
    
    # shapes = {DFNames[0]: '+', DFNames[1]: 'x'}
    
    
    # LCC vs CHP
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=21)
        plt.scatter(x=DF[21], y=DF[LCC_Var]/10**3,label=DFName, s=2, alpha=0.5)#, c=colors_rb[DFName])#, marker=shapes[DFName])
        plt.xlabel(r'CHP Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'LCC (k\$/$m^2$)')
        plt.ylim(-5, 500)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('LCC_vs_CHP_disinteg.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('LCC_vs_CHP_integ.png', dpi=400, bbox_inches='tight')
    
    
    # GHG vs CHP
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=21)
        plt.scatter(x=DF[21], y=DF[CO2_Var]/10**3,label=DFName, s=2, alpha=0.5)#, c=colors_rb[DFName])
        plt.xlabel(r'CHP Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'GHG (k\$/$m^2$)')
        plt.ylim(-0.01, 0.1)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('GHG_vs_CHP_disinteg.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('GHG_vs_CHP_integ.png', dpi=400, bbox_inches='tight')
    
    
    # GHG vs CHP with LCC-oriented transparency
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=21)
        DF = DF[(DF[LCC_Var]<=100) & (DF[CO2_Var]<=100)]
        print('number of indivs plotted: ', len(DF))
        alphas = 1.2 - DF[LCC_Var]/DF[LCC_Var].max() # Normalized LCCs (lowest LCC: 1; highest LCC: 0)
        # alphas = np.linspace(0.1, 1, len(DFs[DFName]))
        rgba_colors = np.zeros((len(DF),4))
        rgba_colors[:,3] = alphas
        plt.scatter(x=DF[21],y=DF[CO2_Var]/10**3,label=DFName, s=1, c=rgba_colors)
        plt.xlabel(r'CHP Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'GHG (k\$/$m^2$)')
        plt.ylim(-0.01, 0.1)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('GHG_vs_CHP_disinteg_colorCoded.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('GHG_vs_CHP_integ_colorCoded.png', dpi=400, bbox_inches='tight')
            
        
    
    # =============================================================================
    # # LCC vs CHP (integrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP+WWT'
    # plt.scatter(x=DFs[DFName][21], y=DFs[DFName][26]/10**3, s=2)
    # plt.xlabel(r'CHP Type')
    # plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.savefig('LCC_vs_CHP_Integ.png', dpi=400, bbox_inches='tight')
    # 
    # 
    # # LCC vs CHP (disintegrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP|CWWTP'
    # plt.scatter(x=DFs[DFName][21], y=DFs[DFName][26]/10**3, s=2)
    # plt.xlabel(r'CHP Type')
    # plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.savefig('LCC_vs_CHP_Disinteg.png', dpi=400, bbox_inches='tight')
    # =============================================================================
    
    
    
    # LCC vs Chiller
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=22)
        plt.scatter(x=DF[22], y=DF[LCC_Var]/10**3,label=DFName, s=2, alpha=0.5)#, c=colors_rb[DFName])
        plt.xlabel(r'Chiller Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'LCC (k\$/$m^2$)')
        plt.ylim(-5, 500)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('LCC_vs_Chiller_disinteg.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('LCC_vs_Chiller_integ.png', dpi=400, bbox_inches='tight')
    
    
    # GHG vs Chiller
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=22)
        plt.scatter(x=DF[22], y=DF[CO2_Var]/10**3,label=DFName, s=2, alpha=0.5)#, c=colors_rb[DFName])
        plt.xlabel(r'Chiller Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'GHG (k\$/$m^2$)')
        plt.ylim(-0.01, 0.1)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('GHG_vs_Chiller_disinteg.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('GHG_vs_Chiller_integ.png', dpi=400, bbox_inches='tight')
            
            
    # GHG vs Chiller with LCC-oriented transparency
    for DFName in DFNames:
        plt.figure(figsize=(10,5))
        DF = DFs[DFName].sort_values(by=22)
        DF = DF[(DF[LCC_Var]<=100) & (DF[CO2_Var]<=0.5)]
        print('number of indivs plotted: ', len(DF))
        alphas = 1 - DF[LCC_Var]/DF[LCC_Var].max() # Normalized LCCs (lowest LCC: 1; highest LCC: 0)
        # alphas = np.linspace(0.1, 1, len(DFs[DFName]))
        rgba_colors = np.zeros((len(DF),4))
        rgba_colors[:,3] = alphas
        plt.scatter(x=DF[22],y=DF[CO2_Var]/10**3,label=DFName, s=1, c=rgba_colors)
        plt.xlabel(r'Chiller Type')
        plt.xticks(rotation=75)
        plt.ylabel(r'GHG (k\$/$m^2$)')
        plt.ylim(-0.01, 0.1)
        # plt.legend()
        if DFName == 'CCHP|CWWTP':
            plt.savefig('GHG_vs_Chiller_disinteg_colorCoded.png', dpi=400, bbox_inches='tight')
        else:
            plt.savefig('GHG_vs_Chiller_integ_colorCoded.png', dpi=400, bbox_inches='tight')
    
    
    
    # =============================================================================
    # # LCC vs Chiller (integrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP+WWT'
    # plt.scatter(x=DFs[DFName][22], y=DFs[DFName][26]/10**3, s=2)
    # plt.xlabel(r'Chiller Type')
    # plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.savefig('LCC_vs_Chiller_Integ.png', dpi=400, bbox_inches='tight')
    # 
    # 
    # # LCC vs Chiller (disintegrated)
    # plt.figure(figsize=(10,5))
    # DFName = 'CCHP|CWWTP'
    # plt.scatter(x=DFs[DFName][22], y=DFs[DFName][26]/10**3, s=2)
    # plt.xlabel(r'Chiller Type')
    # plt.ylabel(r'LCC (k\$/$m^2$)')
    # plt.savefig('LCC_vs_Chiller_Disinteg.png', dpi=400, bbox_inches='tight')
    # =============================================================================
    
    
    
    # LCC vs WWT (integrated)
    plt.figure(figsize=(10,5))
    DFName = 'CCHP+WWT'
    DF = DFs[DFName].sort_values(by=24)
    plt.scatter(x=DF[24], y=DF[LCC_Var]/10**3, s=2)#, c=colors_rb[DFName])
    plt.xlabel(r'WWT Type')
    plt.xticks(rotation=75)
    plt.ylabel(r'LCC (k\$/$m^2$)')
    plt.ylim(-5, 500)
    plt.savefig('LCC_vs_WWT_Integ.png', dpi=400, bbox_inches='tight')
    
    
    
    # GHG vs WWT (integrated)
    plt.figure(figsize=(10,5))
    DFName = 'CCHP+WWT'
    DF = DFs[DFName].sort_values(by=24)
    plt.scatter(x=DF[24], y=DF[CO2_Var]/10**3, s=2)#, c=colors_rb[DFName])
    plt.xlabel(r'WWT Type')
    plt.xticks(rotation=75)
    plt.ylabel(r'GHG (k\$/$m^2$)')
    plt.ylim(-0.01, 0.1)
    plt.savefig('GHG_vs_WWT_Integ.png', dpi=400, bbox_inches='tight')
    
    
    
    # GHG vs WWT with LCC-oriented transparency  (integrated)
    plt.figure(figsize=(10,5))
    DFName = 'CCHP+WWT'
    DF = DFs[DFName].sort_values(by=24)
    DF = DF[(DF[LCC_Var]<=100) & (DF[CO2_Var]<=0.5)]
    print('number of indivs plotted: ', len(DF))
    alphas = 1 - DF[LCC_Var]/DF[LCC_Var].max() # Normalized LCCs (lowest LCC: 1; highest LCC: 0)
    # alphas = np.linspace(0.1, 1, len(DFs[DFName]))
    rgba_colors = np.zeros((len(DF),4))
    rgba_colors[:,3] = alphas
    plt.scatter(x=DF[24],y=DF[CO2_Var]/10**3,s=1, c=rgba_colors)
    plt.xlabel(r'WWT Type')
    plt.xticks(rotation=75)
    plt.ylabel(r'GHG (k\$/$m^2$)')
    plt.ylim(-0.01, 0.1)
    plt.savefig('GHG_vs_WWT_Integ_colorCoded.png', dpi=400, bbox_inches='tight')
    
    plt.close('all')
    





if __name__ == "__main__":
    main()