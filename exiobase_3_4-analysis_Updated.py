"""
Tasks:

    1. Load from pickle  
    2. Calculate for a specific country territorial CO2 emissions 
    3. Calculate for a specific country consumption-based CO2 emissions    
    4. Identify top-5 sectors that contribute to territorial emissions of a specific country
    5. Identify top-5 final demand expenditure on products that contribute to consumption-based emissions of a specific country
    6. Identify top-5 countries where the specific country displaced its emissions to
"""
#%%
import numpy as np
import time
import pickle as pkl
import os
from pathlib import Path
import pandas as pd

np.set_printoptions(precision=2)

tstart = time.time()
##############################################
##############################################
# TASK 1: Load MRIO from Pickle and create variables

##############################################
# Folder settings: Change the current working directory to reflect the location in your computer 
# (run os.getcwd() to find out what that is)
# Folder to read MRIO from
os.getcwd()
mrio_dir = Path(r"C:\Users\patta\OneDrive\Desktop\EIOA\IGA\Analysis")  # Fill " " with your working directory path 

##############################################
# Load MRIO

tstart = time.time()

mrio_name = 'mrio.pkl'  
mrio_str = mrio_dir.joinpath(mrio_name)   
pkl_in = open(mrio_str,"rb")                                                        # 'open' the mrio.pkl file
mrio = pkl.load(pkl_in)                                                             # load the mrio.pkl file to mrio
pkl_in.close()

tend = time.time()
print('Done reading in %5.2f s\n'% (tend - tstart))

# category counts

nr = mrio['label']['region'].count()[0]
ns = mrio['label']['product'].count()[0]
ny = mrio['label']['final'].count()[0]



# MRIO matrix/vector variables
L = mrio['L']
Y = mrio['Y']
V = mrio['V']
F = mrio['F']
F_co2 = mrio['F_co2']
H = mrio['H']
H_co2 = mrio['H_co2']
A = mrio['A']


# Creating the 'original scenario'

# calculate total output 
x = L @ Y.sum(1)

# calculate direct emissions intensity
f_co2 = F_co2/x                                                                     # resulting in na or inf when x = 0 
f_co2 = np.nan_to_num(f_co2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)           # replace na and inf with 0


##############################################
##############################################
# Configuration for all following tasks


# change to choose different region: c in 0 to 48
# position 20 is Netherlands
# explore mrio['label']['region'] to find others
c = 20

# To locate country columns or rows in A and L: 'c*ns : (c+1)*ns' 
# To locate country columns in Y: 'c*ny : (c+1)*ny'

# convert units from kg to Mt (*1e-9)
conv = 1e-9

##############################################
##############################################
# TASK 2. Calculate territorial emissions for a specific country


# We want to report three cols: production-related, households and total, plus a global contribution percentage

pba = np.zeros((1,4))
pba[:,0] = F_co2[c*ns : (c+1)*ns].sum() *conv                                       # direct emissions by producers                 
pba[:,1] = H_co2[c*ny : (c+1)*ny].sum() *conv                                       # direct emissions by households
pba[:,2] = pba[:,0:2].sum() 
pba[:,3] = pba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100

    
##############################################
##############################################
# TASK 3. Calculate for a specific country consumption-based CO2 emissions 


# We want to report three cols: production-generated, households and total, plus a global contribution percentage

cba = np.zeros((1,4))                        
y = Y[:, c*ny : (c+1)*ny].sum(1)                                                    # aggregate final demand to one column

cba[:,0] = f_co2 @ L @ y *conv                                                      # footprint traced to production emissions
cba[:,1] = H_co2[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum() 
cba[:,3] = cba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100


##############################################
##############################################
# TASK 4. Identify top-5 sectors that contribute to production-based emissions

# pba_s: we want to report in rows each sector plus household

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
pba_s = F_co2[c*ns : (c+1)*ns] *conv
pba_s = np.append(pba_s, pba[0,1])                                                  # append household direct emissions to the end

# pba_s_sorted: we want to report in rows the 'sorted' (in descending order) sectoral plus household results; 
# 3 cols: sectoral emissions, percent, and cumulative percent
pba_s_sorted = np.zeros((ns+1,3))
rank = np.argsort(pba_s)                                                            # returns indices that'd sort the array low to high
rank = np.flip(rank)                                                                # flip so it goes 'high to low'
        
pba_s_sorted[:,0] = np.sort(pba_s)                                                  # sort the values from 'low to high'
pba_s_sorted[:,0] = np.flip(pba_s_sorted[:,0])                                      # flip so the values go from high to low
pba_s_sorted[:,1] = pba_s_sorted[:,0]/pba[0,2]*100                                  # percentage contribution
pba_s_sorted[:,2] = np.cumsum(pba_s_sorted[:,1])                                    # cumulative percentage contribution          

# sector names of the top 5 emitters
for i in range(0,5) :                
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], ");")
        

##############################################
##############################################
# TASK 5. Identify top-5 final demand spending sectors that contribute to consumption-based emissions
# TASK 6. Identify top-5 countries where the specific country displaced emissions to

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_s = f_co2 @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions

# cba_s_sort: we want to report in rows each final demand sector(product) plus household
# 3 cols: sectoral emissions, percent, and cumulative percent
cba_s_sorted = np.zeros((ns+1,3))
rank = np.argsort(cba_s)                                                            # sort from 'low to high' and returns indices 
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

cba_s_sorted[:,0] = np.sort(cba_s)                                                  # sort the values from 'low to high'
cba_s_sorted[:,0] = np.flip(cba_s_sorted[:,0])                                      # flip so the values go from high to low
cba_s_sorted[:,1] = cba_s_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_s_sorted[:,2] = np.cumsum(cba_s_sorted[:,1])                                    # cumulative percentage contribution    

# sector names of the top 5 final demand product categories
for i in range(0,5) :                
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], ");")
        
        
# cba_r_sort: we want to report in rows each country of origin
# 3 cols: country emissions, percent, and cumulative percent
label_r = list(mrio['label']['region']['CountryName']) 
cba_r_sorted = np.zeros((nr,3))
rank = np.argsort(cba_r)                                                            # sort from 'low to high' and returns indices 
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

cba_r_sorted[:,0] = np.sort(cba_r)                                                  # sort the values from 'low to high'
cba_r_sorted[:,0] = np.flip(cba_r_sorted[:,0])                                      # flip so the values go from high to low
cba_r_sorted[:,1] = cba_r_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_r_sorted[:,2] = np.cumsum(cba_r_sorted[:,1])                                    # cumulative percentage contribution    

# country names of the top 5 displacement locations
if c in rank[0:5] :
    for i in range(0,6) :  
        if rank[i] != c :    
            print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], ");")
else :
    for i in range(0,5) :    
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], ");")
 
#%%
f = F/x
#%%
f = np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
#%%
og_F4= np.diag(f.sum(0)) @ L @ np.diag(y)

#%% Scenario where the impact of COVID 19 is calculated on the hospitality sector in the Netherlands
# hospitality sector for Netherlands is line 4155

# Implementing reduction in Y
Y_covid = mrio['Y_covid']
Y_reduction = Y_covid[4155,:] * 0.4 # 60% reduction of the final demand of the hotel sector
Y_covid[4155] = Y_reduction

y_covid = Y_covid.sum(1)

# calculate x_covid (Total Output)
Z = A * x
x_covid = Y_covid.sum(1) + Z.sum(1)

# Value Added Covid
V_covid = x_covid - Z.sum(0)

# Calculate Direct Effects (F_covid) based on original F
#calculate multiplier
f_covid = F/x_covid
f_covid = np.nan_to_num(f_covid, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  #nan to 0

# F calculation
F_covid = f_covid @ L @ np.diag(y_covid)
F_covid = np.nan_to_num(F_covid, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  #nan to 0

F_va_covid = F_covid[0:9]  #VA section of F
F_employment_covid = F_covid[9:23]

np.savetxt("F_covid.csv", F_covid, delimiter=",")

#%% Scenario where the impact of the NOW package is calculated on the hospitality sector in the Netherlands

# Implement Package NOW1 - this calculates the % govt payed by taking into account wages
NOW_1 = F[2:4, 4155].sum() * 0.45

# Calculating final demand including the NOW1 package
Y_NOW1 = mrio['Y_NOW1']
Y_NOW1[4155,142] = Y_NOW1[4155,142] + NOW_1  # Govt exp + NOW1 on hospitality sector

# Calculating Total expenditure and Value added with implementation of NOW1 package
x_NOW1 = Y_NOW1.sum(1) + Z.sum(1)
V_NOW1= x_NOW1 - Z.sum(0)

# Calculate Direct Effects (F_NOW1) based on original F
#calculate multiplier
f_NOW1 = F/x_NOW1
f_NOW1 = np.nan_to_num(f_NOW1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  #nan to 0

# F calculation
F_NOW1 = f_NOW1  @ L @ Y_NOW1
F_NOW1 = np.nan_to_num(F_NOW1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)  #nan to 0
F_va_NOW1 = F_NOW1[0:22]  #VA section of F

