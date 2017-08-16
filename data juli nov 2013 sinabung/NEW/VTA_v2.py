# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:41:44 2017

@author: Amajid Sinar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

t = 0.01

#LAST 13100600
#NOT DONE 13100601

#---------------------------------------------------------------------------------
#13070420_44 Sukameriah
data = pd.read_csv("13070420_44.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13070420_44 = []
for i in range(330):
    SH13070420_44.append(z[100+i:800+i])
zeros = np.zeros((len(SH13070420_44),1))
SH13070420_44 = np.array(np.append(zeros,SH13070420_44,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070420_43 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13070420_44[329][1:])
plt.savefig('VTA/0SH13070420_44-TEST.png', bbox_inches='tight')
#---------------------------------------------------------------------------------
#13070322 Sukanalu
data = pd.read_csv("13070322.CDM",delim_whitespace=True)
z = data.iloc[:,0].values


SU13070322 = []
for i in range(381):
    SU13070322.append(z[100+i:800+i])
zeros = np.zeros((len(SU13070322),1))
SU13070322 = np.array(np.append(zeros,SU13070322,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070322 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13070322[380][1:])
plt.savefig('VTA/1SU13070322-TEST.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13070404 Sukameriah
data = pd.read_csv("13070404.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13070404 = []
for i in range(450):
    SH13070404.append(z[50+i:750+i])
zeros = np.zeros((len(SH13070404),1))
SH13070404 = np.array(np.append(zeros,SH13070404,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070404 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13070404[449][1:])
plt.savefig('VTA/2SH13070404-TEST.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13070415_06 Sukanalu
data = pd.read_csv("13070415_06.CDM",delim_whitespace=True)
z = data.iloc[:,0].values

SU13070415_06 = []
for i in range(371):
    SU13070415_06.append(z[0+i:700+i])
zeros = np.zeros((len(SU13070415_06),1))
SU13070415_06 = np.array(np.append(zeros,SU13070415_06,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070415_06 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13070415_06[370][1:])
plt.savefig('VTA/3SU13070415_06-TEST.png', bbox_inches='tight')

#-----------------------------------------------------------------------------
#13070417 Sukanalu
data = pd.read_csv("13070417.CDM",delim_whitespace=True)
z = data.iloc[:,0].values

SU13070417 = []
for i in range(420):
    SU13070417.append(z[0+i:700+i])
zeros = np.zeros((len(SU13070417),1))
SU13070417 = np.array(np.append(zeros,SU13070417,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070417 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13070417[419][1:])
plt.savefig('VTA/4SU13070417-TEST.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
##13070422 Sukanalu
#data = pd.read_csv("13070422.CDM",delim_whitespace=True)
#z = data.iloc[:,0].values
#
#SU13070422 = []
#for i in range(371):
#    SU13070422.append(z[0+i:700+i])
#zeros = np.zeros((len(SU13070422),1))
#SU13070422 = np.array(np.append(zeros,SU13070422,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13070422 Sukanalu")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,SU13070422[370][1:])
#plt.savefig('VTA/SU13070422-TRAINING.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13101207_22 Sukameriah
data = pd.read_csv("13101207_22.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13101207_22 = []
for i in range(301):
    SH13101207_22.append(z[0+i:700+i])
zeros = np.zeros((len(SH13101207_22),1))
SH13101207_22 = np.array(np.append(zeros,SH13101207_22,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13101207_22 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13101207_22[300][1:])
plt.savefig('VTA/SH13101207_22-TRAINING.png', bbox_inches='tight')

#---------------------------------------------------------------------------------

#13070518_50 Sukanalu
data = pd.read_csv("13070518_50.CDM",delim_whitespace=True)
z = data.iloc[:,0].values

SU13070518_50 = []
for i in range(381):
    SU13070518_50.append(z[0+i:700+i])
zeros = np.zeros((len(SU13070518_50),1))
SU13070518_50 = np.array(np.append(zeros,SU13070518_50,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070518_50 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13070518_50[380][1:])
plt.savefig('VTA/5SU13070518_50-TEST.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13070518_50 Laukawar
data = pd.read_csv("13070518_50.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

L13070518_50 = []
for i in range(400):
    L13070518_50.append(z[0+i:700+i])
zeros = np.zeros((len(L13070518_50),1))
L13070518_50 = np.array(np.append(zeros,L13070518_50,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070518_50 Laukawar")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,L13070518_50[399][1:])
plt.savefig('VTA/L13070518_50-TRAINING.png', bbox_inches='tight')

#---------------------------------------------------------------------------
#13070607 Sukanalu
data = pd.read_csv("13070607.CDM",delim_whitespace=True)
z = data.iloc[:,0].values


SU13070607 = []
for i in range(351):
    SU13070607.append(z[0+i:700+i])
zeros = np.zeros((len(SU13070607),1))
SU13070607 = np.array(np.append(zeros,SU13070607,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070607 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13070607[350][1:])
plt.savefig('VTA/6SU13070607-TEST.png', bbox_inches='tight')

#--------------------------------------BELAKANGNYA ANEH-------------------------------------
#13070607 Sukameriah
data = pd.read_csv("13070607.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13070607 = []
for i in range(371):
    SH13070607.append(z[0+i:700+i])
zeros = np.zeros((len(SH13070607),1))
SH13070607 = np.array(np.append(zeros,SH13070607,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13070607 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13070607[370][1:])
plt.savefig('VTA/7SH13070607-TEST.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13070609_46 Sukanalu
#data = pd.read_csv("13070609_46.CDM",delim_whitespace=True)
#z = data.iloc[:,0].values
#
#SU13070609_46 = []
#for i in range(371):
#    SU13070609_46.append(z[0+i:700+i])
#zeros = np.zeros((len(SU13070609_46),1))
#SU13070609_46 = np.array(np.append(zeros,SU13070609_46,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13070609_46 Sukanalu")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,SU13070609_46[370][1:])
#plt.savefig('VTA/SU13070609_46-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13102002_14 Sukameriah
data = pd.read_csv("13102002_14.CDM",delim_whitespace=True)
z = data.iloc[:,2].values


SH13102002_14 = []
for i in range(371):
    SH13102002_14.append(z[0+i:700+i])
zeros = np.zeros((len(SH13102002_14),1))
SH13102002_14 = np.array(np.append(zeros,SH13102002_14,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13102002_14 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13102002_14[370][1:])
plt.savefig('VTA/SH13102002_14-TRAINING.png', bbox_inches='tight')

#---------------------------------------------------------------------------------
#13102000 Sukameriah
data = pd.read_csv("13102007_25.CDM",delim_whitespace=True)
z = data.iloc[:,2].values


SH13102007_25 = []
for i in range(381):
    SH13102007_25.append(z[0+i:700+i])
zeros = np.zeros((len(SH13102007_25),1))
SH13102007_25 = np.array(np.append(zeros,SH13102007_25,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13102007_25 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13102007_25[380][1:])
plt.savefig('VTA/SH13102007_25-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13102100_43 Sukanalu
data = pd.read_csv("13102100_43.CDM",delim_whitespace=True)
z = data.iloc[:,0].values

SU13102100_43 = []
for i in range(361):
    SU13102100_43.append(z[0+i:700+i])
zeros = np.zeros((len(SU13102100_43),1))
SU13102100_43 = np.array(np.append(zeros,SU13102100_43,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13102100_43 Sukanalu")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13102100_43[360][1:])
plt.savefig('VTA/SU13102100_43-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13102100_43 Laukawar
data = pd.read_csv("13102100_43.CDM",delim_whitespace=True)
z = data.iloc[:,1].values

L13102100_43 = []
for i in range(361):
    L13102100_43.append(z[0+i:700+i])
zeros = np.zeros((len(L13102100_43),1))
L13102100_43 = np.array(np.append(zeros,L13102100_43,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13102100_43 Laukawar")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,L13102100_43[360][1:])
plt.savefig('VTA/SU13102100_43-TEST8.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13102100_43 Sukameriah
data = pd.read_csv("13102100_43.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13102100_43 = []
for i in range(400):
    SH13102100_43.append(z[0+i:700+i])
zeros = np.zeros((len(SH13102100_43),1))
SH13102100_43 = np.array(np.append(zeros,SH13102100_43,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13102100_43 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13102100_43[399][1:])
plt.savefig('VTA/SH13102100_43-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------
#13091510_4 Laukawar
data = pd.read_csv("13091510_4.CDM",delim_whitespace=True)
z = data.iloc[:,1].values

L13091510_4 = []
for i in range(301):
    L13091510_4.append(z[0+i:700+i])
zeros = np.zeros((len(L13091510_4),1))
L13091510_4 = np.array(np.append(zeros,L13091510_4,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13091510_4 Laukawar")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,L13091510_4[300][1:])
plt.savefig('VTA/L13091510_4-TEST9.png', bbox_inches='tight')

#------------------------------------------------------------------------------
##13091517_26 Sukameriah
#data = pd.read_csv("13091517_26.CDM",delim_whitespace=True)
#z = data.iloc[:,2].values
#
#SH13091517_26 = []
#for i in range(381):
#    SH13091517_26.append(z[0+i:700+i])
#zeros = np.zeros((len(SH13091517_26),1))
#SH13091517_26 = np.array(np.append(zeros,SH13091517_26,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13091517_26 Sukameriah")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,SH13091517_26[380][1:])
#plt.savefig('VTA/SH13091517_26-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13091519_45 Laukawar
data = pd.read_csv("13091519_45.CDM",delim_whitespace=True)
z = data.iloc[:,1].values

L13091519_45 = []
for i in range(301):
    L13091519_45.append(z[0+i:700+i])
zeros = np.zeros((len(L13091519_45),1))
L13091519_45 = np.array(np.append(zeros,L13091519_45,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13091519_45 Laukawar")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,L13091519_45[300][1:])
plt.savefig('VTA/L13091519_45-TEST10.png', bbox_inches='tight')

#------------------------------------------------------------------------------
#13091519_45 Sukameriah
data = pd.read_csv("13091519_45.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13091519_45 = []
for i in range(301):
    SH13091519_45.append(z[0+i:700+i])
zeros = np.zeros((len(SH13091519_45),1))
SH13091519_45 = np.array(np.append(zeros,SH13091519_45,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13091519_45 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13091519_45[300][1:])
plt.savefig('VTA/SH13091519_45-TRAINING.png', bbox_inches='tight')

#-----------------------------TERBARU COOOY-------------------------------------
#13091519_45 Sukameriah
data = pd.read_csv("13100200_23.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13100200_23 = []
for i in range(340):
    SH13100200_23.append(z[0+i:700+i])
zeros = np.zeros((len(SH13100200_23),1))
SH13100200_23 = np.array(np.append(zeros,SH13100200_23,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13100200_23 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13100200_23[339][1:])
plt.savefig('VTA/SH13100200_23-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
#13091519_45 Sukameriah
data = pd.read_csv("13100219_24.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SU13100219_24 = []
for i in range(351):
    SU13100219_24.append(z[0+i:700+i])
zeros = np.zeros((len(SU13100219_24),1))
SU13100219_24 = np.array(np.append(zeros,SU13100219_24,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("SU13100219_24 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SU13100219_24[350][1:])
plt.savefig('VTA/SU13100219_24-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
##13091519_45 Sukanalu
#data = pd.read_csv("13100204_13.CDM",delim_whitespace=True)
#z = data.iloc[:,0].values
#
#SU13100204_13 = []
#for i in range(351):
#    SU13100204_13.append(z[0+i:700+i])
#zeros = np.zeros((len(SU13100204_13),1))
#SU13100204_13 = np.array(np.append(zeros,SU13100204_13,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13100204_13 Sukanalu")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,SU13100204_13[350][1:])
#plt.savefig('VTA/SU13100204_13-TRAINING.png', bbox_inches='tight')


#----------------------------------------------------------------------
##13091519_45 Sukanalu
#data = pd.read_csv("13100402_42.CDM",delim_whitespace=True)
#z = data.iloc[:,1].values
#
#L13100402_42 = []
#for i in range(351):
#    L13100402_42.append(z[0+i:700+i])
#zeros = np.zeros((len(L13100402_42),1))
#L13100402_42 = np.array(np.append(zeros,L13100402_42,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13100402_42 Laukawar")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,L13100402_42[350][1:])
#plt.savefig('VTA/L13100402_42-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
##13091519_45 Sukanalu
#data = pd.read_csv("13100320_53.CDM",delim_whitespace=True)
#z = data.iloc[:,1].values
#
#SU13100204_13 = []
#for i in range(221):
#    SU13100204_13.append(z[0+i:700+i])
#zeros = np.zeros((len(SU13100204_13),1))
#SU13100204_13 = np.array(np.append(zeros,SU13100204_13,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13100204_13 Sukanalu")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,SU13100204_13[220][1:])
#plt.savefig('VTA/SU13100204_13-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
##13091519_45 Sukanalu
#data = pd.read_csv("13100400_38.CDM",delim_whitespace=True)
#z = data.iloc[:,1].values
#
#L13100400_38 = []
#for i in range(301):
#    L13100400_38.append(z[0+i:700+i])
#zeros = np.zeros((len(L13100400_38),1))
#L13100400_38 = np.array(np.append(zeros,L13100400_38,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13100400_38 Laukawar")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,L13100400_38[300][1:])
#plt.savefig('VTA/L13100400_38-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
#13091519_45 Sukanalu
data = pd.read_csv("13100400_39.CDM",delim_whitespace=True)
z = data.iloc[:,2].values

SH13100400_39 = []
for i in range(360):
    SH13100400_39.append(z[0+i:700+i])
zeros = np.zeros((len(SH13100400_39),1))
SH13100400_39 = np.array(np.append(zeros,SH13100400_39,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13100400_39 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13100400_39[359][1:])
plt.savefig('VTA/SH13100400_39-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
#13091519_45 Sukanalu
#data = pd.read_csv("13100400_42.CDM",delim_whitespace=True)
#z = data.iloc[:,1].values
#
#L13100400_42 = []
#for i in range(360):
#    L13100400_42.append(z[0+i:700+i])
#zeros = np.zeros((len(L13100400_42),1))
#L13100400_42 = np.array(np.append(zeros,L13100400_42,axis=1),dtype=int)
#
##Check by plotting
#t1 = np.linspace(0,t*700,700)
#plt.title("13100400_42 Laukawar")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.plot(t1,L13100400_42[359][1:])
#plt.savefig('VTA/13100400_42-TRAINING.png', bbox_inches='tight')

#----------------------------------------------------------------------
#13091519_45 Sukanalu
data = pd.read_csv("13100519_25.CDM",delim_whitespace=True)
z = data.iloc[:,1].values

SH13100400_39 = []
for i in range(360):
    SH13100400_39.append(z[0+i:700+i])
zeros = np.zeros((len(SH13100400_39),1))
SH13100400_39 = np.array(np.append(zeros,SH13100400_39,axis=1),dtype=int)

#Check by plotting
t1 = np.linspace(0,t*700,700)
plt.title("13100400_39 Sukameriah")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t1,SH13100400_39[0][1:])
plt.savefig('VTA/SH13100400_39-TRAINING.png', bbox_inches='tight')

#------------------------------------------------------------------------------
training_set = np.concatenate((SH13101207_22, L13070518_50,
                               SH13102002_14, SH13102007_25, SU13102100_43,
                               SH13102100_43, SH13091519_45, SH13100200_23, SU13100219_24,
                               SH13100400_39))

#training_set = np.concatenate((SU13070422,SH13101207_22, L13070518_50,
#                               SU13070609_46, SH13102002_14, SH13102007_25, SU13102100_43,
#                               SH13102100_43, SH13091519_45, SH13100200_23, SU13100219_24,
#                               SH13100400_39))

test_set = np.concatenate((SH13070420_44[[300],:], SU13070322[[300],:], SH13070404[[300],:], 
                           SU13070415_06	[[300],:], SU13070417[[300],:], SU13070518_50[[300],:],
                           SU13070607[[300],:], SH13070607[[300],:], SU13102100_43[[300],:],
                           L13091510_4[[300],:], L13091519_45[[300],:]))

np.savetxt("VTA-training-set.csv",training_set,delimiter=";",fmt="%s")
np.savetxt("VTA-test-set.csv",test_set,delimiter=";",fmt="%s")
