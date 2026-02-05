######################################################################################################################
# Copyright 2024 Alice Milne, Jonah Prout, Kevin Coleman (Original Fortran to Python translation)
# Copyright 2025 Terra Madre (Modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE:
# This file is a modified version of the RothC Python implementation.
# Original translation from Fortran by Alice Milne, Jonah Prout, and Kevin Coleman (29/02/2024)
# 
# Modifications made in 2025:
# - Removed all radiocarbon age calculations and related parameters
# - Simplified function signatures to focus on soil carbon pools only
# - Removed redundant type conversions
# - Adapted for integration with Terra+ carbon balance system
######################################################################################################################

import pandas as pd
import numpy as np


# Calculates the rate modifying factor for temperature (RMF_Tmp)
def RMF_Tmp (TEMP):
         
    if(TEMP<-5.0):
        RM_TMP=0.0
    else:
        RM_TMP=47.91/(np.exp(106.06/(TEMP+18.27))+1.0)
   
    return RM_TMP


# Calculates the rate modifying factor for moisture (RMF_Moist)
def RMF_Moist (RAIN, PEVAP, clay, depth, PC, SWC):
        
    RMFMax = 1.0
    RMFMin = 0.2

# calc soil water functions properties
    SMDMax=-(20+1.3*clay-0.01*(clay*clay))
    SMDMaxAdj = SMDMax * depth / 23.0
    SMD1bar = 0.444 * SMDMaxAdj
    SMDBare = 0.556 * SMDMaxAdj
      
    DF = RAIN - 0.75 * PEVAP

    minSWCDF=np.min (np.array([0.0, SWC[0]+DF]))
    minSMDBareSWC=np.min (np.array([SMDBare, SWC[0]]))
      
    if(PC==1):
        SWC[0] = np.max(np.array([SMDMaxAdj, minSWCDF]))
    else:
        SWC[0] = np.max(np.array([minSMDBareSWC, minSWCDF]))
      
    if(SWC[0]>SMD1bar): 
        RM_Moist = 1.0
    else:
        RM_Moist = (RMFMin + (RMFMax - RMFMin) * (SMDMaxAdj - SWC[0]) / (SMDMaxAdj - SMD1bar) )   
    
    return RM_Moist


# Calculates the plant retainment modifying factor (RMF_PC)
def RMF_PC (PC):
     
    if (PC==0):
        RM_PC = 1.0
    else:
        RM_PC = 0.6

    return RM_PC


# Calculates the decomposition of soil carbon pools
def decomp(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, RateM, clay, C_Inp, FYM_Inp, DPM_RPM):

# rate constant are params so don't need to be passed
    DPM_k = 10.0
    RPM_k = 0.3
    BIO_k = 0.66
    HUM_k = 0.02 

    tstep = 1.0/timeFact    # monthly 1/12, or daily 1/365 
 
# decomposition
    DPM1 = DPM[0] * np.exp(-RateM*DPM_k*tstep)
    RPM1 = RPM[0] * np.exp(-RateM*RPM_k*tstep)      
    BIO1 = BIO[0] * np.exp(-RateM*BIO_k*tstep)      
    HUM1 = HUM[0] * np.exp(-RateM*HUM_k*tstep) 
      
    DPM_d = DPM[0] - DPM1
    RPM_d = RPM[0] - RPM1      
    BIO_d = BIO[0] - BIO1
    HUM_d = HUM[0] - HUM1 
    
    x=1.67*(1.85+1.60*np.exp(-0.0786*clay))
                    
# proportion C from each pool into CO2, BIO and HUM      
    DPM_co2 = DPM_d * (x / (x+1))
    DPM_BIO = DPM_d * (0.46 / (x+1))
    DPM_HUM = DPM_d * (0.54 / (x+1))
      
    RPM_co2 = RPM_d * (x / (x+1))
    RPM_BIO = RPM_d * (0.46 / (x+1))
    RPM_HUM = RPM_d * (0.54 / (x+1))    
      
    BIO_co2 = BIO_d * (x / (x+1))
    BIO_BIO = BIO_d* (0.46 / (x+1))
    BIO_HUM = BIO_d * (0.54 / (x+1))
      
    HUM_co2 = HUM_d * (x / (x+1))
    HUM_BIO = HUM_d * (0.46 / (x+1))
    HUM_HUM = HUM_d * (0.54 / (x+1))  
      
    # update C pools  
    DPM[0] = DPM1
    RPM[0] = RPM1
    BIO[0] = BIO1 + DPM_BIO + RPM_BIO + BIO_BIO + HUM_BIO
    HUM[0] = HUM1 + DPM_HUM + RPM_HUM + BIO_HUM + HUM_HUM    
      
    # split plant C to DPM and RPM 
    PI_C_DPM = DPM_RPM / (DPM_RPM + 1.0) * C_Inp
    PI_C_RPM =     1.0 / (DPM_RPM + 1.0) * C_Inp

    # split FYM C to DPM, RPM and HUM 
    FYM_C_DPM = 0.49*FYM_Inp
    FYM_C_RPM = 0.49*FYM_Inp      
    FYM_C_HUM = 0.02*FYM_Inp   
      
    # add Plant C and FYM_C to DPM, RPM and HUM   
    DPM[0] = DPM[0] + PI_C_DPM + FYM_C_DPM
    RPM[0] = RPM[0] + PI_C_RPM + FYM_C_RPM  
    HUM[0] = HUM[0] + FYM_C_HUM
      
    # calculate total SOC
    SOC[0] = DPM[0] + RPM[0] + BIO[0] + HUM[0] + IOM[0]     
    
    return


def rothc(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, clay, depth, TEMP, RAIN, PEVAP, PC, DPM_RPM, C_Inp, FYM_Inp, SWC, RM_TILL=1.0):     
     
    # Calculate RMFs     
    RM_TMP = RMF_Tmp(TEMP)
    RM_Moist = RMF_Moist(RAIN, PEVAP, clay, depth, PC, SWC)
    RM_PC = RMF_PC(PC)

    # Combine RMF's into one.      
    RateM = RM_TMP*RM_Moist*RM_PC*RM_TILL
    # print(f"\ntimeFact={timeFact}, DPM={DPM[0]:.2f}, RPM={RPM[0]:.2f}, BIO={BIO[0]:.2f}, HUM={HUM[0]:.2f}, IOM={IOM[0]:.2f}, SOC={SOC[0]:.2f}, "
    #       f"RateM={RateM:.2f}, clay={clay:.2f}, C_Inp={C_Inp:.2f}, FYM_Inp={FYM_Inp:.2f}, DPM_RPM={DPM_RPM:.2f}")
    decomp(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, RateM, clay, C_Inp, FYM_Inp, DPM_RPM)
   
    return


def rothc_spinup(som, clay, depth, monthly):
    """Run RothC model to equilibrium to calculate initial soil carbon pools.
    
    Runs the model iteratively until carbon pools stabilize (test < 1E-6).
    Used during farm initialization to establish baseline soil carbon state.

    Args:
        som: Soil organic matter content (tC/ha)
        clay: Clay content (%)
        depth: Soil depth (cm)
        monthly: DataFrame with 12 months of data containing columns:
                 't_year', 't_month', 't_tmp' (°C), 't_rain' (mm), 't_evap' (mm),
                 't_C_Inp' (tC/ha/month), 't_FYM_Inp' (tC/ha/month), 't_PC' (plant cover),
                 't_DPM_RPM' (decomposable/resistant plant material ratio)
                 
    Returns:
        dict: Initial soil carbon pools at equilibrium (all in tC/ha):
              'DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC'
    """
    
    # set initial pool values   
    DPM = [0.0]
    RPM = [0.0]
    BIO = [0.0]
    HUM = [0.0]
    SOC = [0.0]
    IOM = [0.049*(som**1.139)]  ############### IOM calculation from Falloon et al., 1998

    # set initial soil water content (deficit) 
    SWC = [0.0]
    TOC1 = 0.0

    k = -1
    j = -1
        
    SOC[0] = DPM[0]+RPM[0]+BIO[0]+HUM[0]+IOM[0]
    
    timeFact = 12
    test = 1.0

    while (test > 1E-6):
        k = k + 1
        j = j + 1 

        if( k == timeFact):
            k = 0
        
        TEMP = monthly.t_tmp[k]
        RAIN = monthly.t_rain[k]
        PEVAP = monthly.t_evap[k]
        
        PC = monthly.t_PC[k]
        DPM_RPM = monthly.t_DPM_RPM[k]
        
        C_Inp = monthly.t_C_Inp[k]
        FYM_Inp = monthly.t_FYM_Inp[k]

        rothc(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, clay, depth, TEMP, RAIN, PEVAP, PC, DPM_RPM, C_Inp, FYM_Inp, SWC)  
            
        # Each year calculates the difference between the previous year and current year (counter =12 monthly model)
        if (np.mod(k+1, timeFact)== 0):
            TOC0 = TOC1
            TOC1 = DPM[0]+RPM[0]+BIO[0]+HUM[0]
            test = abs(TOC1-TOC0)  
    
    # Create a dictionary of results to return
    results = {
        'DPM': DPM[0],
        'RPM': RPM[0],
        'BIO': BIO[0],
        'HUM': HUM[0],
        'IOM': IOM[0],
        'SOC': SOC[0]
    }
    return results

def rothc_transient(year, clay, depth, monthly, initial_pools, tillage_modifier=1.0):
    """Run RothC model for a single year to calculate updated soil carbon pools.
    
    Advances soil carbon pools forward one year based on inputs, climate, and management.
    Used during yearly calculations to track soil carbon changes over time.

    Args:
        year: Simulation year (int or str)
        clay: Clay content (%)
        depth: Soil depth (cm)
        monthly: DataFrame with 12 months of data containing columns:
                 't_year', 't_month', 't_tmp' (°C), 't_rain' (mm), 't_evap' (mm),
                 't_C_Inp' (tC/ha/month), 't_FYM_Inp' (tC/ha/month), 't_PC' (plant cover),
                 't_DPM_RPM' (decomposable/resistant plant material ratio)
        initial_pools: Dict with starting carbon pools (tC/ha): 'DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC'
        tillage_modifier: Decomposition rate modifier for tillage intensity (default: 1.0)
                          Values > 1.0 increase decomposition, < 1.0 decrease it
                          
    Returns:
        dict: Updated soil carbon pools after one year (all in tC/ha):
              'DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC'
    """

    # set initial pool values   
    DPM = [initial_pools['DPM']]
    RPM = [initial_pools['RPM']]
    BIO = [initial_pools['BIO']]
    HUM = [initial_pools['HUM']]
    SOC = [initial_pools['SOC']]
    IOM = [initial_pools['IOM']]

    # set initial soil water content (deficit) 
    SWC = [0.0]
    TOC1 = 0.0
    RM_TILL = tillage_modifier

    nsteps = 12
    
    timeFact = 12

    for i in range(nsteps):
        
        TEMP = monthly.t_tmp[i]
        RAIN = monthly.t_rain[i]
        PEVAP = monthly.t_evap[i]
        
        PC = monthly.t_PC[i]
        DPM_RPM = monthly.t_DPM_RPM[i]
        
        C_Inp = monthly.t_C_Inp[i]
        FYM_Inp = monthly.t_FYM_Inp[i]

        rothc(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, clay, depth, TEMP, RAIN, PEVAP, PC, DPM_RPM, C_Inp, FYM_Inp, SWC, RM_TILL)  

    # Create a dictionary of results to return
    results = {
        'DPM': DPM[0],
        'RPM': RPM[0],
        'BIO': BIO[0],
        'HUM': HUM[0],
        'IOM': IOM[0],
        'SOC': SOC[0]
    }
    return results


def prepare_rothc_pools(soil_data, type, initial_pools=None):
    """Prepare carbon inputs and initial pool values for RothC model.
    
    Args:
        soil_data (pd.DataFrame): DataFrame with carbon input data per plot
        type (str): "spinup" or "transient" - determines how pools are initialized
        initial_pools (pd.DataFrame, optional): DataFrame with initial pool values (columns: plot_name, DPM, RPM, BIO, HUM, IOM, SOC)
    
    Returns:
        pd.DataFrame: Carbon inputs and pool values ready for RothC

    """

    rothc_pools = soil_data.copy()
    rothc_pools.rename(columns={'rothc_soc30_t_ha': 'SOC'}, inplace=True)
    clay_pct = rothc_pools['rothc_clay_pct'].values
    rothc_pools.drop(columns=['rothc_clay_pct'], inplace=True)

    pool_cols = ['DPM', 'RPM', 'BIO', 'HUM', 'IOM', 'SOC']

    # If period is "spinup", pools start at zero (will be calculated)
    if type == "spinup":
        for col in pool_cols:
            rothc_pools[col] = 0.0
        return rothc_pools
        
    required_cols = ['case'] + pool_cols
    
    # Use initial_pools if provided
    if initial_pools is not None:
        init_pools = initial_pools.copy()
        rothc_pools = rothc_pools.merge(
            init_pools[required_cols],
            on='case',
            how='left'
        )
        return rothc_pools
    else:
        # If no initial_pools provided, use soc to calculate pool sizes
        # Pool distribution: Calculated using pedotransfer functions (Weihermüller et al., 2013):
        # RPM = (0.1847 × SOC + 0.1555) × (clay + 1.2750)^(-0.1158)
        # HUM = (0.7148 × SOC + 0.5069) × (clay + 0.3421)^(0.0184)
        # BIO = (0.0140 × SOC + 0.0075) × (clay + 8.8473)^(0.0567)
        # IOM = 0.049 × SOC^1.139
        # DPM = 0 (at equilibrium)
        rothc_pools['RPM'] = (0.1847 * rothc_pools['SOC'] + 0.1555) * (clay_pct + 1.2750)**(-0.1158)
        rothc_pools['HUM'] = (0.7148 * rothc_pools['SOC'] + 0.5069) * (clay_pct + 0.3421)**(0.0184)
        rothc_pools['BIO'] = (0.0140 * rothc_pools['SOC'] + 0.0075) * (clay_pct + 8.8473)**(0.0567)
        rothc_pools['IOM'] = 0.049 * rothc_pools['SOC']**1.139
        rothc_pools['DPM'] = 0.0
        return rothc_pools
    

            
    

    
