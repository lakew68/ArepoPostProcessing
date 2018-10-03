"""
Computes the indices for all particles in the spherical overdensity (DM/G) halos.
Usage: ipython particleindicesgeneral.py res vel snapnum
Returns: npy file wiht array of indices per halo.

read npy file with 

goodidx, allgasidx, alldmidx = np.load("particleindexingoodhaloz10.npy")
"""
from __future__ import print_function, division
from sys import argv
import numpy as np
import readsubfHDF5
import snapHDF5 
import hdf5lib
from gadget import *
from gadget_subfind import *
import calcGrid

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
UnitTime_in_s= UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitDensity_in_cgs= UnitMass_in_g/ np.power(UnitLength_in_cm,3)
UnitPressure_in_cgs= UnitMass_in_g/ UnitLength_in_cm/ np.power(UnitTime_in_s,2)
UnitEnergy_in_cgs= UnitMass_in_g * np.power(UnitLength_in_cm,2) / np.power(UnitTime_in_s,2)
GCONST=GRAVITY_cgs/ np.power(UnitLength_in_cm,3) * UnitMass_in_g *  np.power(UnitTime_in_s,2)
critical_density = 3.0*.1 * .1 / 8.0/np.pi/GCONST #.1 is for 1/Mpc to 1/kpc, also in units of h^2


#Read snapshot arguments
script, res, vel, snapnum = argv
snapnum = int(snapnum) 
s_vel = vel.replace(".","") 
s_res = res.replace(".","") 


#Read filepaths
"""
filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
"""

prefix = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
run = 'DM_FOF'
path = prefix + run
base = 'snap-groupordered_'

#Load snapshot
s = gadget_readsnap(snapnum, snappath=path, snapbase=base, hdf5=True, loadonlytype=[0,1], loadonly=['pos','mass','vel'])
cat = load_subfind(snapnum, dir=path + '/', hdf5=True)

#Setup boxsize quantities
s.boxsize *= s.time / s.hubbleparam #put boxsize in physical units
boxSizeVel = s.boxsize * .1 * UnitLength_in_cm/UnitVelocity_in_cm_per_s * np.sqrt(s.omega0/s.time/s.time/s.time + s.omegalambda)


#indices of haloes >300 dm particles and r200 not 0
r200 = cat.data['frc2']
haloPos = cat.data['fpos']
over300idx, = np.where(np.logical_and(np.greater(cat.data['flty'][:,1],300),np.not_equal(r200,0.)))

#Implementation for all halos
allgasindex = np.empty(np.size(r200),dtype=list)
alldmindex = np.empty(np.size(r200),dtype=list)

#Read in particles
gasidx, = np.where(s.data['type']==0)
dmidx, = np.where(s.data['type']==1)

massgas = s.data['mass'][gasidx]
massdm = s.data['mass'][dmidx]
posgas = s.data['pos'][gasidx]
posdm = s.data['pos'][dmidx]
velgas = s.data['vel'][gasidx]
veldm = s.data['vel'][dmidx]

for j in over300idx:
	#Calculate indices of particles within r200
	indgas, = np.where(dist2(posgas[:,0]-haloPos[j][0],posgas[:,1]-haloPos[j][1],posgas[:,2]-haloPos[j][2],boxSize)  < r200[j]**2)
	allgasindex[j] = indgas
	inddm, = np.where(dist2(posdm[:,0]-haloPos[j][0],posdm[:,1]-haloPos[j][1],posdm[:,2]-haloPos[j][2],boxSize)  < r200[j]**2)
	alldmindex[j] = inddm
np.save("particleindex_" + s_res +'_'+ s_vel + '_'+  str(snapnum) + ".npy", (goodidx, allgasindex[goodidx], alldmindex[goodidx]))


