"""
Computes the closest DM halo counterpart of Gas primary objects
Returns: pickled file with dictionary containing matched information

Updated 10/12/23
"""

from __future__ import print_function, division

import numpy as np
import readsubfHDF5
import snapHDF5
try:
   import cPickle as pickle
except:
   import pickle


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
baryonfraction = .044 / .27 #OmegaB/Omega0

#Should be run with a snap number input
res = '14Mpc'
vel = 'Sig2'
snapkey = range(154)
for snapnum in snapkey:
    snapnum = int(snapnum)
    s_vel = vel.replace(".","")
    s_res = res.replace(".","")

    #File paths
    filename = "D:/Star_Movie_768/"
    filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
    filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for snapHDF5

    #Read header information
    header = snapHDF5.snapshot_header(filename3)
    red = header.redshift
    
    atime = header.time
    boxSize = header.boxsize
    Omega0 = header.omega0
    OmegaLambda = header.omegaL
    massDMParticle = header.massarr[1] #all DM particles have same mass

    #redshift evolution of critical_density
    critical_density *= Omega0 + atime**3 * OmegaLambda
    critical_density_gas = critical_density * baryonfraction
    #Read halo catalog
    catDM = readsubfHDF5.subfind_catalog(filename + "DM_FOF", snapnum)
    catGas = readsubfHDF5.subfind_catalog(filename + "GasOnly_FOF", snapnum)
    

    #Get CM & R200 of all halos w/ >300 DM particles, >100 gas particles
    GroupPos_Gas = catGas.GroupPos[catGas.GroupLenType[:,0]+catGas.GroupLenType[:,4]>100]
    GroupPos_DM = catDM.GroupPos[catDM.GroupLenType[:,1]>300]
    R200_DM = catDM.Group_R_Crit200[catDM.GroupLenType[:,1]>300]
    M200_DM = catDM.Group_M_Crit200[catDM.GroupLenType[:,1]>300]
    print(np.ndim(GroupPos_Gas))
    print(np.shape(GroupPos_Gas))
    np.savetxt("foo.csv", catGas.GroupLenType[:,0]>100, delimiter=",")

    #Filter for nonzero R200
    GroupPos_DM = GroupPos_DM[R200_DM!=0.]
    M200_DM = M200_DM[R200_DM!=0.]
    R200_DM = R200_DM[R200_DM!=0.]


    #Allocate arrays 
    matchingHalos = [] 
    Rmin = []
    R200dm = [] 
    M200dm = []
    matchingHalos2 = [] 
    Rmin2 = []
    R200dm2 = []
    M200dm2 = []

    #For each gas object, calculate the distance to all DM objects
    #Get the minimum and record that as the separation 

    for igas, groupPos in list(enumerate(GroupPos_Gas)):
        dists = dist2(GroupPos_DM[:,0]-groupPos[0], GroupPos_DM[:,1]-groupPos[1], GroupPos_DM[:,2]-groupPos[2],boxSize)
        virDists = np.sqrt(dists) / R200_DM
        idx = np.where(virDists == np.min(virDists))[0][0]
        matchingHalos += [idx]
        Rmin += [np.sqrt(dists[idx])] #dists is actually squared
        M200dm += [M200_DM[idx]]
        R200dm += [R200_DM[idx]]
        idx2 = sorted(list(enumerate(virDists)), key=lambda x: x[1])[1][0] #Gets index of second largest distance
        matchingHalos2 += [idx2]
        Rmin2 += [np.sqrt(dists[idx2])] #dists is actually squared
        M200dm2 += [M200_DM[idx2]]
        R200dm2 += [R200_DM[idx2]]

    #Print information
    matched = {} #Initialize dict of results
    matched['red'] = np.array(red)
    matched['matchingHalos'] = np.array(matchingHalos)
    matched['Rmin'] = np.array(Rmin)
    matched['R200dm'] = np.array(R200dm)
    matched['M200dm'] = np.array(M200dm)
    matched['matchingHalos2'] = np.array(matchingHalos2)
    matched['Rmin2'] = np.array(Rmin2)
    matched['R200dm2'] = np.array(R200dm2)
    matched['M200dm2'] = np.array(M200dm2)


    with open(filename + "match"+s_res+"_"+s_vel+"_"+str(snapnum)+"_2.dat", 'wb') as f:
        pickle.dump(matched, f)

