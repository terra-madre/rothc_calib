from rothc import rothc
def run_rothc(cases_treatments_df, climate_df, initial_pools_df, soil_depth_cm=30) :

    # Initialize soil carbon pools
    DPM = [initial_pools_df.loc[0, 'DPM']]
    RPM = [initial_pools_df.loc[0, 'RPM']]
    BIO = [initial_pools_df.loc[0, 'BIO']]
    HUM = [initial_pools_df.loc[0, 'HUM']]
    IOM = [initial_pools_df.loc[0, 'IOM']]
    SOC = [DPM[0] + RPM[0] + BIO[0] + HUM[0] + IOM[0]]

    # Loop through each case and run RothC for control and treatments
    for _, case in cases_treatments_df.iterrows():

        clay = case['rothc_clay_pct']
        timeFact = 12
        
        # Prepare monthly data
        monthly = climate_df.copy()
        monthly['t_FYM_Inp'] = 0.0
        monthly['t_PC'] = 1

        print( j, DPM[0], RPM[0], BIO[0], HUM[0], IOM[0], SOC[0], Total_Delta)

        year_list = [[1, j+1, DPM[0], RPM[0], BIO[0], HUM[0], IOM[0], SOC[0], Total_Delta[0]]]

        month_list = []        

        for  i in range(timeFact, nsteps):
        
            TEMP = df.t_tmp[i]
            RAIN = df.t_rain[i]
            PEVAP =df.t_evap[i]
            
            PC = df.t_PC[i]
            DPM_RPM = df.t_DPM_RPM[i]
            
            C_Inp = df.t_C_Inp[i]
            FYM_Inp = df.t_FYM_Inp[i]
            depth = soil_depth_cm
            
            rothc(timeFact, DPM, RPM, BIO, HUM, IOM, SOC, clay, depth, TEMP, RAIN, PEVAP, PC, DPM_RPM, C_Inp, FYM_Inp, SWC, RM_TILL=1.0)
                
            Total_Delta = (np.exp(-Total_Rage[0]/8035.0) - 1.0) * 1000.0
            
            print(C_Inp, FYM_Inp, TEMP, RAIN, PEVAP, SWC[0],  PC,  DPM[0],RPM[0],BIO[0],HUM[0], IOM[0], SOC[0])
            
            month_list.insert(i-timeFact, [df.loc[i,"t_year"],df.loc[i,"t_month"], DPM[0],RPM[0],BIO[0],HUM[0], IOM[0], SOC[0], Total_Delta[0]])
                
            if(df.t_month[i] == timeFact):
                timeFact_index = int(i/timeFact)   
                year_list.insert(timeFact_index, [df.loc[i,"t_year"],df.loc[i,"t_month"], DPM[0],RPM[0],BIO[0],HUM[0], IOM[0], SOC[0], Total_Delta[0]])
                print( i, DPM, RPM, BIO, HUM, IOM, SOC, Total_Delta)

        output_years = pd.DataFrame(year_list, columns=["Year","Month","DPM_t_C_ha","RPM_t_C_ha","BIO_t_C_ha","HUM_t_C_ha","IOM_t_C_ha","SOC_t_C_ha","deltaC"])     
        output_months = pd.DataFrame(month_list, columns=["Year","Month","DPM_t_C_ha","RPM_t_C_ha","BIO_t_C_ha","HUM_t_C_ha","IOM_t_C_ha","SOC_t_C_ha","deltaC"])

        output_years.to_csv("year_results.csv", index = False)
        output_months.to_csv("month_results.csv", index = False)
