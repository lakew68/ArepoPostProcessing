"""
Computes the closest DM halo counterpart of Gas primary objects
Usage: python match.py 1.4Mpc 11.8kms 10
Returns: file with dictionary containing matched information
Notes: change numsnaps depending on simulation
"""

from __future__ import print_function, division
from sys import argv
import numpy as np
import hdf5lib 
import readsubfHDF5
import snapHDF5
try:
   import cPickle as pickle
except:
   import pickle

NUMSNAPS = 13 



def dx_wrap(dx,box):
	#wraps to account for period boundary conditions. This mutates the original entry
	idx = dx > +box/2.0
	dx[idx] -= box
	idx = dx < -box/2.0
	dx[idx] += box 
	return dx
def dist2(dx,dy,dz,box):
	#Calculates distance taking into account periodic boundary conditions
	return dx_wrap(dx,box)**2 + dx_wrap(dy,box)**2 + dx_wrap(dz,box)**2

# Units
GRAVITY_cgs = 6.672e-8
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
KPC = 3.085678e21
ZSUN = 0.0127
UnitLength_in_cm = 3.085678e21 # code length unit in cm/h
UnitMass_in_g = 1.989e43       # code length unit in g/h
UnitVelocity_in_cm_per_s = 1.0e5
UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitDensity_in_cgs = UnitMass_in_g/ np.power(UnitLength_in_cm,3)
UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / np.power(UnitTime_in_s,2)
UnitEnergy_in_cgs = UnitMass_in_g * np.power(UnitLength_in_cm,2) / np.power(UnitTime_in_s,2)
GCONST = GRAVITY_cgs / np.power(UnitLength_in_cm,3) * UnitMass_in_g *  np.power(UnitTime_in_s,2)
critical_density = 3.0 * .1 * .1 / 8.0 / np.pi / GCONST #.1 is to convert 100/Mpc to 1/kpc, this is in units of h^2
hubbleparam = .71 #hubble constant

#Should be run with a snap number input
script, res, vel,  snapnum = argv
snapnum = int(snapnum) 
s_vel = vel.replace(".","")
s_res = res.replace(".","")
#File paths
filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for snapHDF5

#Read header information
header = snapHDF5.snapshot_header(filename3)
red = header.redshift
atime = header.time
boxSize = header.boxsize
Omega0 = header.omega0
OmegaLambda = header.omegaL

#Read halo catalog
catGas = readsubfHDF5.subfind_catalog(filename + "GasOnly_FOF", snapnum)
catDM = readsubfHDF5.subfind_catalog(filename + "DM_FOF", snapnum)

#Get CM & R200 of all halos w/ >300 DM particles, >100 gas particles
GroupPos_Gas = catGas.GroupPos[catGas.GroupLenType[:,0]>100]
GroupPos_DM = catDM.GroupPos[catDM.GroupLenType[:,1]>300]
R200_DM = catDM.Group_R_Crit200[catDM.GroupLenType[:,1]>300]
M200_DM = catDM.Group_M_Crit200[catDM.GroupLenType[:,1]>300]

#Filter for nonzero R200
GroupPos_DM = GroupPos_DM[R200_DM!=0.]
M200_DM = M200_DM[R200_DM!=0.]
R200_DM = R200_DM[R200_DM!=0.]


#Allocate arrays 
matchingHalos = np.array([None for groupPos in GroupPos_Gas])
Rmin = np.array([None for groupPos in GroupPos_Gas])
R200dm = np.array([None for groupPos in GroupPos_Gas])
M200dm = np.array([None for groupPos in GroupPos_Gas])

#For each gas object, calculate the distance to all DM objects
#Get the minimum and record that as the separation 

for igas, groupPos in list(enumerate(GroupPos_Gas)):
	dists = dist2(GroupPos_DM[:,0]-groupPos[0], GroupPos_DM[:,1]-groupPos[1], GroupPos_DM[:,2]-groupPos[2],boxSize)
	idx = np.where(dists == np.min(dists))[0][0]
	matchingHalos[igas] = idx
	Rmin[igas] = np.sqrt(np.min(dists)) #dists is actually squared
	M200dm[igas] = M200_DM[idx]
	R200dm[igas] = R200_DM[idx]

#Print information
matched = {} #Initialize dict of results
matched['red'] = red
matched['matchingHalos'] = matchingHalos
matched['Rmin'] = Rmin
matched['R200dm'] = R200dm
matched['M200dm'] = M200dm

"""
with open("match"+s_res+"_"+s_vel+"_"+str(snapnum)+".dat", 'wb') as f:
    pickle.dump(matched, f)
"""

#np.save('match14Mpc_118kms_10.npy',matched)
np.savez('match14Mpc_118kms_10.npz',red=red, matchingHalos=matchingHalos,Rmin=Rmin,R200dm=R200dm,M200dm=M200dm)
