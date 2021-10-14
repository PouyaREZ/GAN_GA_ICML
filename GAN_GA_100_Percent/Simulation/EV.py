# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:52:19 2021

@author: PouyaRZ
"""
import numpy as np

def EV_Load_Calculator(Num_Buildings, Type, Diversifier_Peak, Building_Vars, Demand_Types,
                       EV_Input, Residential_Electricity_Use_SF, NonResidential_Electricity_Use_SF,
                       Total_Electricity_Use_SF, MWh_to_kWh, MT_to_lbs, Grid_Emissions, Buy_Price,
                       EV_Load_Ratio, Aggregate_Demand):
    
    Total_EV_Demand = 0 # kWh
    EV_Charging_GHG = 0 # Ton CO2
    EV_Charging_LCC = 0 # $
    Peak_EV_Demand = 0 # kW

    residential_bldgs_electricity = 0
    non_residential_bldgs_electricity = 0
    total_bldgs_electricity = 0
    for i in range(Num_Buildings):
        j = i+1
        
        
        if Type[j] == 1 or j == 20: # Residential or Mixed Use type 1
            residential_bldgs_electricity += np.sum(Diversifier_Peak*(Building_Vars[j]*Demand_Types[j][:,0]))
        else:
            non_residential_bldgs_electricity += np.sum(Diversifier_Peak*(Building_Vars[j]*Demand_Types[j][:,0]))
    
    total_bldgs_electricity = residential_bldgs_electricity + non_residential_bldgs_electricity
        
    ## The hourly load of EV charging from, respectively, residential, non-residential, and public chargers
    EV_Hourly_Charging_Load =\
        residential_bldgs_electricity/Residential_Electricity_Use_SF * (EV_Input[:,0] + EV_Input[:,1]) +\
            non_residential_bldgs_electricity/NonResidential_Electricity_Use_SF * EV_Input[:,4] +\
                total_bldgs_electricity/Total_Electricity_Use_SF * (EV_Input[:,2] + EV_Input[:,3]) # in MW
    
    EV_Hourly_Charging_Load *= MWh_to_kWh # convert to kW
    Total_EV_Demand = np.sum(EV_Hourly_Charging_Load) # in kWh
    Peak_EV_Demand = np.max(EV_Hourly_Charging_Load)
    
    EV_Hourly_Charging_Load_on_CCHP = EV_Load_Ratio/100 * EV_Hourly_Charging_Load # in kW
    EV_Hourly_Charging_Load_on_Grid = EV_Hourly_Charging_Load - EV_Hourly_Charging_Load_on_CCHP # in kW
    
    ## Add the electric load on CCHP to the electric demand from the buildings
    Aggregate_Demand[:,0] += EV_Hourly_Charging_Load_on_CCHP # kW
    ## Calculate the emissions from the load sent to the grid
    EV_Charging_GHG = np.sum(EV_Hourly_Charging_Load_on_Grid * Grid_Emissions) / MT_to_lbs # Ton CO2
    EV_Charging_LCC = np.sum(Buy_Price*EV_Hourly_Charging_Load_on_Grid) # $
    
    return Total_EV_Demand, EV_Charging_GHG, EV_Charging_LCC, Peak_EV_Demand