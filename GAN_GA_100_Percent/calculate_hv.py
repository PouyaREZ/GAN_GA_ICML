# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:14:10 2021

@author: The Author
"""



# Import libraries
import numpy as np
import pandas as pd
import os

from Filter_Data import DF_Filter
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import minmax_scale



# Constants
# Num_Samplings = 20
ref_point = [1.0, 1.0, 1.0]

Num_Sites = 4
len_of_indiv = 11 # After adding the plant location

## Variables for the original dataset
LCC_Var = Num_Sites+11
CO2_Var = Num_Sites+14
WalkScore_Var = Num_Sites+15
## Labels for the generated dataset
LCC_Var_Gen = len_of_indiv+0
CO2_Var_Gen = len_of_indiv+1
WalkScore_Var_Gen = len_of_indiv+2

np.random.seed(42)







############################ Helper Functions #####################################
def results_total_finder(experiment):
    path = './'
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'resultsTotal'+experiment in i:
            files.append(i)
    return files



def load_combined_results(experiment):
    allResults = None
    for i, filename in enumerate(results_total_finder(experiment)):
        # print('processed %s'%filename)
        if i == 0:
            allResults = np.loadtxt(filename)
        else:
            allResults = np.append(allResults, np.loadtxt(filename), axis=0)
    
    print('Loaded %d points generated by GAN from combined results of %s'%(len(allResults), experiment))
    return allResults



############################ Main #####################################








# DECLARE THE EXPERIMENT TYPE
aggregate_results = True
for Experiment in ['WorstHalfCO2', 'WorstHalfLCC', 'WorstHalfWalkScore',
                    'WorstHalfAll', 'BestHalfAll', 'FullData']:
# Experiment = 'BestHalfAll'
    print('\n+++++++++++++++++')
    print('Results for the experiment %s:\n'%Experiment)
    
    
    # Load the proper set of filenames based on the experiment
    if Experiment == 'FullData':
        ## NewerFullData; NewFullData; FullData (better)
        fileNames = ['FullData_1000', 'FullData_5000', 'FullData_15000', 'FullData_39999'] # ['FullData_19456', 'FullData_19712', 'FullData_19968', 'FullData_19999'] # ['FullData_100', 'FullData_400', 'FullData_1000', 'FullData_1999_1']
        # fileNames = ['FullData_200', 'FullData_500', 'FullData_1100', 'FullData_1700', 'FullData_1999']
    # if Experiment == 'WorstHalfLCC':
    #     fileNames = ['WorstHalfLCC_200_2', 'WorstHalfLCC_600', 'WorstHalfLCC_999']# ['799_6']
    elif Experiment == 'WorstHalfLCC': ## NewWorstHalfLCC ## WorstHalfLCC
        fileNames = ['WorstHalfLCC_1000', 'WorstHalfLCC_5000', 'WorstHalfLCC_10000', 'WorstHalfLCC_15000', 'WorstHalfLCC_19999'] # ['WorstHalfLCC_200', 'WorstHalfLCC_799']# ['799_6']
    # elif Experiment == 'BestHalfLCC':
    #     fileNames = ['BestHalfLCC_100', 'BestHalfLCC_300', 'BestHalfLCC_500', 'BestHalfLCC_1000', 'BestHalfLCC_1999']# ['799_6']
    elif Experiment == 'WorstHalfCO2': ## NewWorstHalfCO2 ## WorstHalfCO2
        fileNames = ['WorstHalfCO2_1000', 'WorstHalfCO2_5000', 'WorstHalfCO2_10000', 'WorstHalfCO2_15000', 'WorstHalfCO2_19999'] # ['WorstHalfCO2_2_300', 'WorstHalfCO2_2_700'] 
    elif Experiment == 'WorstHalfWalkScore': ## NewWorstHalfWalkscore; WorstHalfWalkscore
        fileNames = ['WorstHalfWalkscore_1000', 'WorstHalfWalkscore_5000', 'WorstHalfWalkscore_10000', 'WorstHalfWalkscore_15000', 'WorstHalfWalkscore_23999'] # ['WorstHalfWalkscore_400', 'WorstHalfWalkscore_799']
    elif Experiment == 'WorstHalfAll':## NewWorstHalAll; NewerWorstHalfAll; WorstHalfAll
        fileNames = ['WorstHalfAll2_100', 'WorstHalfAll2_300', 'WorstHalfAll2_500', 'WorstHalfAll2_1500', 'WorstHalfAll2_3199'] # ['WorstHalfAll_100', 'WorstHalfAll_300', 'WorstHalfAll_1000', 'WorstHalfAll_2000', 'WorstHalfAll_3199']  #['WorstHalfAll_400', 'WorstHalfAll_1000']
    elif Experiment == 'BestHalfAll':## NewBestHalAll; BestHalfAll
        fileNames = ['BestHalfAll_150', 'BestHalfAll_600', 'BestHalfAll_2100', 'BestHalfAll_4050', 'BestHalfAll_4999'] # ['BestHalfAll_100', 'BestHalfAll_500', 'BestHalfAll_1000', 'BestHalfAll_1800', 'BestHalfAll_1999']
    # elif Experiment == 'BestHalfAny':
    #     fileNames = ['FullData_19456', 'FullData_19712', 'FullData_19968', 'FullData_19999']
    
    
    
    # Load the datasets
    if not aggregate_results:
        # resultsTotal = np.loadtxt('resultsTotal.txt')
        resultsTotal = pd.DataFrame(np.loadtxt('resultsTotal'+fileNames[0]+'.txt'))
        print('Loaded points generated by GAN from %s'%('resultsTotal'+fileNames[0]+'.txt'))
    else:
        resultsTotal = pd.DataFrame(load_combined_results(Experiment))
    # resultsTotal = pd.DataFrame(np.loadtxt('resultsTotal'+fileNames[0]+'.txt'))
    # print('Loaded points generated by GAN from %s'%('resultsTotal'+fileNames[0]+'.txt'))
    
    # print('loading data generated by the optimization algorithm')
    filename = './IILP_Toy_Optimization_TestRuns.txt'
    # DFName = 'CCHP+Network'
    DF = DF_Filter(filename, experiment=Experiment, verbose=0)
    
    
    
    # Calculate the hypervolume
    ## modify the input parameters to make them suitable for calculating the HV w.r.t. the reference point (0,0,0)
    maxLCC = max(np.max(DF[LCC_Var]), np.max(resultsTotal[LCC_Var_Gen]))
    maxCO2 = max(np.max(DF[CO2_Var]), np.max(resultsTotal[CO2_Var_Gen]))
    maxWalkScore = max(np.max(DF[WalkScore_Var]), np.max(resultsTotal[WalkScore_Var_Gen]))
    
    minLCC = min(np.min(DF[LCC_Var]), np.min(resultsTotal[LCC_Var_Gen]))
    minCO2 = min(np.min(DF[CO2_Var]), np.min(resultsTotal[CO2_Var_Gen]))
    minWalkScore = min(np.min(DF[WalkScore_Var]), np.min(resultsTotal[WalkScore_Var_Gen]))
    
    
    ## Reverse the data columns
    def reverse_data(DF, column, maxValue): # Subtract each value from the given max value in a column of a dataframe/np.array
        DF[column] = maxValue - DF[column]
        
    # reverse_data(DF, LCC_Var, maxLCC)
    # reverse_data(resultsTotal, LCC_Var_Gen, maxLCC)
    # reverse_data(DF, CO2_Var, maxCO2)
    # reverse_data(resultsTotal, CO2_Var_Gen, maxCO2)
    reverse_data(DF, WalkScore_Var, maxWalkScore)
    reverse_data(resultsTotal, WalkScore_Var_Gen, maxWalkScore)
    
    
    ## Normalize the data
    def normalize_data(DF, column, minValue, maxValue):
        DF[column] = (DF[column] - minValue)/(maxValue - minValue)
    
    normalize_data(DF, LCC_Var, minLCC, maxLCC)
    normalize_data(DF, CO2_Var, minCO2, maxCO2)
    normalize_data(DF, WalkScore_Var, minWalkScore, maxWalkScore)
    
    normalize_data(resultsTotal, LCC_Var_Gen, minLCC, maxLCC)
    normalize_data(resultsTotal, CO2_Var_Gen, minCO2, maxCO2)
    normalize_data(resultsTotal, WalkScore_Var_Gen, minWalkScore, maxWalkScore)
    
    
    
    ## Calculate the hv
    from pymoo.factory import get_performance_indicator
    hv = get_performance_indicator("hv", ref_point=np.array(ref_point))#[0.5, 0.5, 0.5]#ref_point=np.array([maxLCC, maxCO2, maxWalkScore]))
    
    
    ## GIVES MEMORY ERROR IF USED DIRECTLY ##
    array1 = np.array(DF[[LCC_Var, CO2_Var, WalkScore_Var]])
    # print("hv for the original solutions", hv.calc(array1))
    
    
    array2 = np.array(resultsTotal[[LCC_Var_Gen, CO2_Var_Gen, WalkScore_Var_Gen]])
    generatedArea = hv.calc(array2)
    # print("hv for the generated solutions", generatedArea)
    
    
    originalAreas = []
    # prevArr = None
    Num_Samplings = int(len(array1)/len(array2))+1
    for i in range(Num_Samplings):
        choices = np.random.randint(low=0, high=len(array1), size=len(array2))
        array1_2 = np.array(DF[[LCC_Var, CO2_Var, WalkScore_Var]])[choices, :]
        # if i != 0:
            # if np.all(prevArr == array1_2): print('Same array selected twice!!') 
        originalAreas.append(hv.calc(array1_2))
        # prevArr = array1_2
    # print("hv for the original solutions", originalArea)
    
    meanOriginalArea = np.mean(originalAreas)
    maxOriginalArea = np.max(originalAreas)
    
    

    
    print('Maximum generated hv:%.2e'%generatedArea)
    print('Maximum original hv:%.2e'%maxOriginalArea)
    print('Mean original hv:%.2e'%meanOriginalArea)
    if generatedArea > meanOriginalArea:
        print('Generated solutions have on average a hv %.2f%% larger than the original solutions'%((generatedArea - meanOriginalArea)/generatedArea*100))
    else:
        print('Original solutions have on average a hv %.2f%% larger than the generated solutions'%((meanOriginalArea - generatedArea)/meanOriginalArea*100))
        
        
    if generatedArea > maxOriginalArea:
        print('Generated solutions have a hv %.2f%% larger than the best of original solutions'%((generatedArea - maxOriginalArea)/generatedArea*100))
    else:
        print('Original solutions have at best a hv %.2f%% larger than the generated solutions'%((maxOriginalArea - generatedArea)/maxOriginalArea*100))