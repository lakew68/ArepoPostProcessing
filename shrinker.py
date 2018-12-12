"""
Program to fit ellipsoid to FOF object, then shrinks the ellipsoid.
Usage: python shrinker.py 1.4Mpc 11.8kms 10
Returns: pickled file with dictionary containing ellipsoid information
Note: pickled dictionary will contain numpy object arrays. Need to convert to float64

12/11/18
"""

from __future__ import print_function, division
from sys import argv
import numpy as np
import readsubfHDF5
import snapHDF5 
from annikaEllipsoid import *
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
mDM = header.massarr[1] #all DM particles have same mass


#load particle indices
pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)

halo100_indices= np.where(cat.GroupLenType[:,0] >100)[0]		
startAllGas = []
endAllGas   = []
for i in halo100_indices:
	startAllGas += [np.sum(cat.GroupLenType[:i,0])]
	endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]

CM_112Mpc_Sig0 = cat.GroupPos[halo100_indices]

overRadii_112Mpc_Sig0 = []
radii_112Mpc_Sig0 = []
rotation_112Mpc_Sig0 = []
cm_112Mpc_Sig0 = []
mDM_112Mpc_Sig0 = []
mGas_112Mpc_Sig0 = []
gasFrac_112Mpc_Sig0 = []
DMindices_112Mpc_Sig0 = []


for idx in halo100_indices:
	cm = CM_112Mpc_Sig0[idx] 
	startGas = startAllGas[idx]
	endGas = endAllGas[idx]	
	if np.sum(cm == np.array([0., 0., 0.]))==3:
		# it's probbaly an error; recompute com
		totalGas = np.sum(mGas[startGas: endGas])
		cm = np.array([np.sum(pGas[startGas: endGas, i]*mGas[startGas: endGas])/totalGas for i in range(3)])
					
#Reposition particles to take into account CM
	P = pGas[startGas:endGas]
	M = mGas[startGas:endGas]
	Precentered = dx_wrap(P - cm,boxSize)
	dists = np.sqrt(dist2(P[:,0]-cm[0],P[:,1]-cm[1],P[:,2]-cm[2],boxSize))
	maxAxis = np.max(dists)
	ratios, evecs = axis(Precentered,maxAxis,axes_out=True,quiet=True)
	
	if ratios[0] > 0. and ratios[1] > 0.: #accounts for erros in fit
		evecs = np.array(evecs)
		Precentered = np.array([np.dot(pp,evecs.T) for pp in Precentered])
		tempAxis = maxAxis
		inEll = Precentered[:,0]**2/ratios[0]**2+Precentered[:,1]**2/ratios[1]**2+Precentered[:,2]**2 <= maxAxis**2
		numGasOrig = np.sum(inEll)		
		while True:		
			inEll = Precentered[:,0]**2/ratios[0]**2+Precentered[:,1]**2/ratios[1]**2+Precentered[:,2]**2 <= tempAxis**2
			numGasTemp = np.sum(inEll) 
			if tempAxis/maxAxis > 1.*numGasTemp/numGasOrig or 1.*numGasTemp/numGasOrig <= .8:
				break
			tempAxis-= .005*maxAxis
		over =  np.sum(M[inEll])/(4.*np.pi/3.*ratios[0]*ratios[1]*tempAxis**3.)/rhoCritGas_112Mpc_Sig0[snapnum-10]

		mGas_112Mpc_Sig0 += [np.sum(M[inEll])]
		overRadii_112Mpc_Sig0 += [over]
		radii = np.array([tempAxis*ratios[0], tempAxis*ratios[1], tempAxis])
		rotation = np.array(evecs)
		radii_112Mpc_Sig0 += [list(radii)]	
		rotation_112Mpc_Sig0 += [[list(r) for r in rotation]]
		cm_112Mpc_Sig0 += [list(cm)]	
			
		
		#Calculate DM mass
		tempPosDM = dx_wrap(pDM-cm,boxSize)			
		nearidx, = np.where(dist2(pDM[:,0]-cm[0],pDM[:,1]-cm[1],pDM[:,2]-cm[2],boxSize)<=tempAxis**2)
		DMindices_112Mpc_Sig0 += [nearidx] 
		tempPosDM = tempPosDM[nearidx]
		tempPosDM = np.array([np.dot(pp,evecs.T) for pp in tempPosDM])
		DMinEll = tempPosDM[:,0]**2/ratios[0]**2 + tempPosDM[:,1]**2/ratios[1]**2 + tempPosDM[:,2]**2 <= tempAxis**2 
		mDM_112Mpc_Sig0 += [1.*np.sum(DMinEll)*mDM]	
		gasFrac_112Mpc_Sig0 += [np.sum(M[inEll])/(np.sum(M[inEll]) + 1.*np.sum(DMinEll)*mDM)]	
	else:
		overRadii_112Mpc_Sig0 += [-1.]
		radii_112Mpc_Sig0 += [[-1.,-1.,-1.]]
		rotation_112Mpc_Sig0 +=[[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]] 
		cm_112Mpc_Sig0 += [[-1.,-1.,-1.]]
		mDM_112Mpc_Sig0 += [-1.]
		mGas_112Mpc_Sig0 += [-1.]
		gasFrac_112Mpc_Sig0 += [-1.]
		DMindices_112Mpc_Sig0 += [[-1.]]

#Print information
shrunken = {} #Initialize dict of results
shrunken['radii'] = [list(r) for r in radii_112Mpc_Sig0]
shrunken['rotation'] = [[list(r[0]), list(r[1]), list(r[2])] for r in rotation_112Mpc_Sig0]
shrunken['cm'] = [list(c) for c in cm_112Mpc_Sig0]
shrunken['overRadii'] = list(overRadii_112Mpc_Sig0)
shrunken['mDM'] = list(mDM_112Mpc_Sig0)
shrunken['mGas'] = list(mGas_112Mpc_Sig0)
shrunken['gasFrac'] = list(gasFrac_112Mpc_Sig0)
shrunken['DMindices'] = [list(d) for d in DMindices_112Mpc_Sig0]

with open("shrinker"+s_res+"_"+s_vel+"_"+str(snapnum)+".dat",'wb') as f:
	pickle.dump(matched, f)

"""
print "import numpy as np"
print "radii_112Mpc_Sig0 = np.array([None])"
print "rotation_112Mpc_Sig0 = np.array([None])"
print "cm_112Mpc_Sig0 = np.array([None])"
print "overRadii_112Mpc_Sig0 = np.array([None])"
print "mDM_112Mpc_Sig0 = np.array([None])"
print "mGas_112Mpc_Sig0 = np.array([None])"
print "gasFrac_112Mpc_Sig0 = np.array([None])"
print "DMindices_112Mpc_Sig0 = np.array([None])"

print "radii_112Mpc_Sig0[0]    = ", [list(r) for r in radii_112Mpc_Sig0]
print "rotation_112Mpc_Sig0[0] = ", [[list(r[0]), list(r[1]), list(r[2])] for r in rotation_112Mpc_Sig0]
print "cm_112Mpc_Sig0[0]       = ", [list(c) for c in cm_112Mpc_Sig0]
print "overRadii_112Mpc_Sig0[0]  = ", list(overRadii_112Mpc_Sig0)
print "mDM_112Mpc_Sig0[0]  = ", list(mDM_112Mpc_Sig0)
print "mGas_112Mpc_Sig0[0]  = ", list(mGas_112Mpc_Sig0)
print "gasFrac_112Mpc_Sig0[0]  = ", list(gasFrac_112Mpc_Sig0)
print "DMindices_112Mpc_Sig0[0] = ", [list(d) for d in DMindices_112Mpc_Sig0]
print "radii_112Mpc_Sig0[0] = np.array(radii_112Mpc_Sig0[0])"
print "rotation_112Mpc_Sig0[0] = np.array(rotation_112Mpc_Sig0[0])"
print "cm_112Mpc_Sig0[0] = np.array(cm_112Mpc_Sig0[0])"
print "overRadii_112Mpc_Sig0[0] = np.array(overRadii_112Mpc_Sig0[0])"
print "mDM_112Mpc_Sig0[0] = np.array(mDM_112Mpc_Sig0[0])"
print "mGas_112Mpc_Sig0[0] = np.array(mGas_112Mpc_Sig0[0])"
print "gasFrac_112Mpc_Sig0[0] = np.array(gasFrac_112Mpc_Sig0[0])"
print "DMindices_112Mpc_Sig0[0] = np.array(DMindices_112Mpc_Sig0[0])"
"""
