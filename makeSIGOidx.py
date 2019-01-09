"""
Program to computes SIGO indices.
Usage: python makeSIGOidx.py res vel snapnum
Returns: pickled file with np array containing SIGO indices
1/9/19
"""

from __future__ import print_function, division
from sys import argv
import numpy as np
import readsubfHDF5
import snapHDF5 
try:
   import cPickle as pickle
except:
   import pickle

from match14Mpc_118kms import *
from yeoushrinker14Mpc_118kms_10 import *


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
script, res, vel,  snapnum = argv
snapnum = int(snapnum) 
s_vel = vel.replace(".","")
s_res = res.replace(".","")

#File paths
filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for snapHDF5

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

#load particle indices and catalogs
pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
cat = readsubfHDF5.subfind_catalog(filename2, snapnum)

halo100_indices= np.where(cat.GroupLenType[:,0] >100)[0]		
startAllGas = []
endAllGas   = []
for i in halo100_indices:
	startAllGas += [np.sum(cat.GroupLenType[:i,0])]
	endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]


SIGOidx = []
"""
#Load shrinker and match data
with open('shrinker'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
	shrunken = pickle.load(f)
with open('match'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
	matched = pickle.load(f)
"""
for i in halo100_indices:
	"""	
	cm = shrunken['cm'][i]
	rotation = shrunken['rotation'][i]
	radii = shrunken['radii'][i]
	mDM = shrunken['mDM'][i]
	DMinEll = shrunken['DMindices'][i]
	Rclosest = matched['Rmin'][i]
	R200dm = matched['R200dm'][i]
	"""
	exec("cm = cm_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("rotation = rotation_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("radii = radii_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("mDM=mDM_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("DMinEll=DMindices_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))	
	exec("Rclosest=Rmin_%s_%s[snapnum-10][i]"%(s_res,s_vel))
	exec("R200dm=R200dm_%s_%s[snapnum-10][i]"%(s_res,s_vel))

	if radii[0] > 0.: #In case of shrinker errors
		
		#Check if CM is buggy
		if np.sum(cm == np.array([0., 0., 0.]))==3:
			# it's probbaly an error; recompute com	
			totalGas = np.sum(mGas[startAllGas[i]: endAllGas[i]])
			cm = np.array([np.sum(pGas[startAllGas[i]: endAllGas[i], j]*mGas[startAllGas[i]: endAllGas[i]])/totalGas for j in range(3)])

		# Get positions of gas particles
		P = pGas[startAllGas[i]: endAllGas[i]]
		M = mGas[startAllGas[i]: endAllGas[i]]
		Pdm = pDM[DMinEll]
		# Shift coordinate system to center on the center of the ellipsoid
		Precentered = dx_wrap(P - cm,boxSize)
		PrecenteredDM = dx_wrap(Pdm - cm,boxSize)
		# Rotate coordinated to the the axes point along x,y,z directions:
		Precentered = np.array([np.dot(pp, rotation.T) for pp in Precentered])
		PrecenteredDM = np.array([np.dot(pp, rotation.T) for pp in PrecenteredDM])

		# Figure out which particles are inside the ellipsoid
		inEll = (Precentered[:,0]**2./radii[0]**2. + Precentered[:,1]**2./radii[1]**2 + Precentered[:,2]**2./radii[2]**2)<=1.
		if np.size(P[inEll]) > 100: #Only consider SIGOs with greater than 100 particles
			if (np.sum(M[inEll])/(np.sum(M[inEll])+mDM)>.4) and (Rclosest/R200dm>1.):
				SIGOidx += [i]	

with open("SIGOidx"+s_res+"_"+s_vel+"_"+str(snapnum)+".dat",'wb') as f:
	pickle.dump(np.array(SIGOidx), f)

