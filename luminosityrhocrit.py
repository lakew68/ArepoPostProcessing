"""
Program to plot luminosity vs radius using only objects that satisfy rho > rho_crit
"""
from __future__ import print_function, division
import matplotlib
#matplotlib.use('Agg')
import pylab
from gadget import *
from gadget_subfind import *
import calcGrid
from sys import argv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import readsubfHDF5
import snapHDF5 
import hdf5lib
try:
	import cPickle as pickle
except:
	import pickle

np.random.seed(42)

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


#H hardness in units of log. Z= [0, 10^-7, 10^-5]
Q_H = {'A': [46.98, 46.94, 46.90], 'B':[47.1, 46.65, 46.99], 'C': [47.98, 48.01, 48.02]}

#Line em coefs. T = [30kK, 10kK]
c_LA = [1.04e-11, 1.04e-11] #Used with Q_H
c_He2 = [5.67e-12, 6.4e-12] #Used with Q_HeP
c_HA = [1.21e-12, 1.37e-12] #Used with Q_H

#Shaerer starbust model parameteres
IMF = 'A'
Z = 0
T = 0
f_esc = .5 #photon escape fraciton
f_star= .1 #precent to convert to stellar mass


#SIGO

#File paths
filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5


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
pGas = snapHDF5.read_block(filename3,"POS ", parttype=0)
mGas = snapHDF5.read_block(filename3,"MASS", parttype=0)
vGas = snapHDF5.read_block(filename3,"VEL ", parttype=0)
rGas = snapHDF5.read_block(filename3,"RHO ",parttype=0) 
uGas = snapHDF5.read_block(filename3,"U   ",parttype=0)
pDM = snapHDF5.read_block(filename3,"POS ",parttype=1)
catGas = readsubfHDF5.subfind_catalog(filename2, snapnum)


halo100_indices= np.where(catGas.GroupLenType[:,0] >100)[0]		
startAllGas = []
endAllGas   = []
for i in halo100_indices:
	startAllGas += [np.sum(catGas.GroupLenType[:i,0])]
	endAllGas   += [startAllGas[-1] + catGas.GroupLenType[i,0]]

cms = catGas.GroupPos / hubbleparam / atime #convert to physical
cvel = catGas.GroupVel / atime

#Initialize arrays
#some radii are errors and  negative, will have a value of 1 to be excluded
negradii = np.zeros(np.size(halo100_indices)) 
rmax = np.zeros(np.size(halo100_indices))
gasfrac = np.zeros(np.size(halo100_indices))
gasmass = np.zeros(np.size(halo100_indices))
rmin =  np.zeros(halo100_indices.size)
rhocritGP = np.zeros(halo100_indices.size)
rhoGP = np.zeros(halo100_indices.size)

#get SIGO indices
with open('SIGOidx'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
	SIGOidx =  pickle.load(f)
#Load shrinker and match data
with open('shrinker'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
	shrunken = pickle.load(f)
with open('match'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
	matched = pickle.load(f)


#Calculate properties
for i in halo100_indices:
	cm = shrunken['cm'][i]
	rotation = shrunken['rotation'][i]
	radii = shrunken['radii'][i]
	mDM = shrunken['mDM'][i]
	DMinEll = shrunken['DMindices'][i]
	Rclosest = matched['Rmin'][i]
	R200dm = matched['R200dm'][i]

	if radii[0] <= 0.:
		negradii[i] = 1.
	else:	
		#Check if CM is buggy
		if np.sum(cm == np.array([0., 0., 0.]))==3:
			# it's probbaly an error; recompute com	
			totalGas = np.sum(mGas[startAllGas[i]: endAllGas[i]])
			cm = np.array([np.sum(pGas[startAllGas[i]: endAllGas[i], j]*mGas[startAllGas[i]: endAllGas[i]])/totalGas for j in range(3)])

		# Get positions of gas particles
		P = pGas[startAllGas[i]: endAllGas[i]]
		M = mGas[startAllGas[i]: endAllGas[i]]
		R = rGas[startAllGas[i]: endAllGas[i]]	
		U = uGas[startAllGas[i]: endAllGas[i]]
		V = vGas[startAllGas[i]: endAllGas[i]]
		Pdm = pDM[DMinEll]
		# Shift coordinate system to center on the center of the ellipsoid
		Precentered = dx_wrap(P - cm,boxSize)
		PrecenteredDM = dx_wrap(Pdm - cm,boxSize)
		# Rotate coordinated to the the axes point along x,y,z directions:
		Precentered = np.array([np.dot(pp, rotation.T) for pp in Precentered])
		PrecenteredDM = np.array([np.dot(pp, rotation.T) for pp in PrecenteredDM])

		# Figure out which particles are inside the ellipsoid
		inEll = (Precentered[:,0]**2./radii[0]**2. + Precentered[:,1]**2./radii[1]**2 + Precentered[:,2]**2./radii[2]**2)<=1.


		#convert internal energy to temperature 
		u = 1.0e10 * U[inEll] #it's a velocity squared to be converted in cgs
		mu = 1.22 #Primordial composition 
		temp = GAMMA_MINUS1 /  BOLTZMANN * u * PROTONMASS * mu	
			
		#turn into sound speed
		cs = np.sqrt(GAMMA * BOLTZMANN * temp / mu / PROTONMASS)
		tempvel = (V[inEll] * np.sqrt(atime) - cvel[i]) * 1.0e5 #convert to cgs
		#Turn first v_x into |v|
		velmag = np.linalg.norm(tempvel,axis=1)
		#turn into mach number
		mach = velmag / cs	
		#using rmax
		L_cloud = (2.*radii[2]*atime/hubbleparam*KPC) #in cm
		#L_cloud = ((radii[2]+radii[0])*s.time/s.hubbleparam*KPC) #use avg of rmin/rmax
			
		rhocritGP[i] = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / L_cloud**2.
		#using rmin	
		#rhocrit = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / (radii[0]*s.time/s.hubbleparam*KPC)**2.
		
		SIGOrho = np.sum(M[inEll]) /  (4./3. * np.pi * radii[0]*radii[1]*radii[2]) / (atime**3 / hubbleparam**2) * 1.e10 * MSUN / KPC **3.
		SIGOrhocell = np.mean(R[inEll]) * 1.0e10  *MSUN/ KPC**3.0 / (atime**3. / hubbleparam**2)
		rhoGP[i] = SIGOrho	

		#Calculate jeans length/sonic length		
		lamb_sonic = L_cloud / np.mean(mach)**2.
		lamb_jeans = np.sqrt(np.pi * np.mean(cs)**2. / GRAVITY_cgs / SIGOrho)

		m_BE = 1.18 / np.pi**(1.5) * SIGOrho * lamb_jeans**3.
		SIGOmass = M[inEll].sum() * 1.e10 / hubbleparam
		gasmass[i] = np.sum(M[inEll])
		gasfrac[i] = np.sum(M[inEll])/(np.sum(M[inEll])+mDM)
		#Make stellar density to  long axis
		#densradius[i] = .5 * (radii[0] + radii[2])
		rmax[i] = radii[2]
		rmin[i] = radii[0]

#Adjust for bad ellipsoids
nonzero, = np.where(negradii==0.)
#SIGOs
if s_vel == '118kms':	
	rmaxSIGO = rmax[SIGOidx] * atime /hubbleparam
	rminSIGO = rmin[SIGOidx] * atime /hubbleparam
	gasfracSIGO = gasfrac[SIGOidx]
	gasmassSIGO = gasmass[SIGOidx] *10.**10/hubbleparam
	stellarmassSIGO = f_star * gasmassSIGO

rmax = rmax[nonzero]
rmin = rmin[nonzero]
gasfrac = gasfrac[nonzero]
gasmass = gasmass[nonzero]
rmax *= atime / hubbleparam #convert to kpc
rmin *= atime / hubbleparam
gasmass *= 10.**10/hubbleparam


stellarmass = f_star * gasmass
Q_Hmassive = 10.**(Q_H[IMF][Z]) * stellarmass #unnormalize from one stellar mass
#ZAMS luminosity
luminosity = c_LA[T] * (1.-f_esc) * Q_Hmassive
#SIGOs
if s_vel == '118kms':
	Q_HmassiveSIGO = 10.**(Q_H[IMF][Z]) * stellarmassSIGO #unnormalize from one stellar mass
	#ZAMS luminosity
	luminositySIGO = c_LA[T] * (1.-f_esc) * Q_HmassiveSIGO

rhoSIGO = rhoGP[SIGOidx]
rhocritSIGO = rhocritGP[SIGOidx]
rhoGP = rhoGP[nonzero]
rhocritGP = rhocritGP[nonzero]

#Calculate indices of supercritical SIGOs
rhoGTrhocritindex, = np.where(rhoGP > rhocritGP)
rhoGTrhocritANDSIGO, = np.where(rhoSIGO > rhocritSIGO) 

#Print information
GasPrimary = {}
GasPrimary['rmax'] = rmax
GasPrimary['rmin'] = rmin
GasPrimary['gasfrac'] = gasfrac
GasPrimary['gasmass'] = gasmass
GasPrimary['luminosity'] = luminosity
GasPrimary['rhoGP'] = rhoGP
GasPrimary['rhocritGP'] = rhocritGP
GasPrimary['rhoGTrhocritindex'] = rhoGTrhocritindex
if s_vel == '118kms':
	GasPrimary['rhoSIGO'] = rhoSIGO
	GasPrimary['rhocritSIGO'] = rhocritSIGO
	GasPrimary['rhoGTrhocritANDSIGO'] = rhoGTrhocritANDSIGO
	GasPrimary['rmaxSIGO'] = rmaxSIGO
	GasPrimary['rminSIGO'] = rminSIGO
	GasPrimary['gasfracSIGO'] = gasfracSIGO
	GasPrimary['gasmassSIGO'] = gasmassSIGO
	GasPrimary['luminositySIGO'] = luminositySIGO

with open('GP_luminosity'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','wb') as f:
	pickle.dump(GasPrimary, f)

#DMG

#File paths
filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
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
pGas = snapHDF5.read_block(filename3,"POS ",parttype=0)
mGas = snapHDF5.read_block(filename3,"MASS",parttype=0)
vGas = snapHDF5.read_block(filename3,"VEL ",parttype=0)
rGas = snapHDF5.read_block(filename3,"RHO ",parttype=0)
uGas = snapHDF5.read_block(filename3,"U   ",parttype=0)
pDM = snapHDF5.read_block(filename3,"POS ",parttype=1)
mDM = snapHDF5.read_block(filename3,"MASS", parttype=1)
catDM = readsubfHDF5.subfind_catalog(filename2, snapnum)

cms = catDM.GroupPos / hubbleparam / atime
cvel = catDM.GroupVel / atime

#load DM/G indices
over300idx, indgas, inddm = np.load('particleindex_'+s_res+'_'+s_vel+'_'+str(snapnum)+'.npy')
over300idx = over300idx.astype(int)


#initialize arrays
rhocritDMG = np.zeros(catDM.Group_R_Crit200.size)
rhoDMG = np.zeros(catDM.Group_R_Crit200.size)
densradius= np.zeros(catDM.Group_R_Crit200.size)
maxrho = np.zeros(catDM.Group_R_Crit200.size)
gasmass = np.zeros(catDM.Group_R_Crit200.size)
gasfrac = np.zeros(catDM.Group_R_Crit200.size)



for i,j in enumerate(over300idx):
	if indgas[i].size > 100: #Only care about DM/G with >100 gas cells
		RHO = rGas[indgas[i]]	
		gasfrac[i] = mGas[indgas[i]].sum()/(mGas[indgas[i]].sum()+mDM[inddm[i]].sum())  
		gasmass[i] = mGas[indgas[i]].sum()
		densradius[i] = catDM.Group_R_Crit200[j]	

		#convert internal energy to temperature 
		u = 1.0e10 * uGas[indgas[i]] #it's a velocity squared to be converted in cgs
		mu = 1.22 #Primordial composition 
		temp = GAMMA_MINUS1 /  BOLTZMANN * u * PROTONMASS * mu	
			
		#turn into sound speed
		cs = np.sqrt(GAMMA * BOLTZMANN * temp / mu / PROTONMASS)
		tempvel = (vGas[indgas[i]] * np.sqrt(atime) - cvel[j]) * 1.0e5 #convert to cgs
		#Turn first v_x into |v|
		velmag = np.linalg.norm(tempvel,axis=1)
		#turn into mach number
		mach = velmag / cs	
		L_cloud = 2.*(catDM.Group_R_Crit200[j]*atime/hubbleparam*KPC) 
			
		rhocritDMG[i] = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / L_cloud**2.
		rhoDMG[i] = np.sum(mGas[indgas[i]]) / (4./3. * np.pi * catDM.Group_R_Crit200[j])/(atime**3 / hubbleparam**2) * 1.e10 * MSUN / KPC **3.

densradius *= atime / hubbleparam
gasmass *= 10.**10/hubbleparam

densradius = densradius[over300idx]
gasfrac = gasfrac[over300idx]
gasmass = gasmass[over300idx]
rhoDMG = rhoDMG[over300idx]
rhocritDMG = rhocritDMG[over300idx]



gasmass = gasmass[densradius!=0.]
gasfrac = gasfrac[densradius!=0.]
rhoDMG = rhoDMG[densradius!=0.]
rhocritDMG = rhocritDMG[densradius!=0.]
densradius = densradius[densradius!=0.]

stellarmass = f_star * gasmass
Q_Hmassive = 10.**(Q_H[IMF][Z]) * stellarmass #unnormalize from one stellar mass

#ZAMS luminosity
luminosity = c_LA[T] * (1.-f_esc) * Q_Hmassive

rhoGTrhocritindex, = np.where(rhoDMG > rhocritDMG)



#Print information
DMG = {} 
DMG['densradius'] = densradius
DMG['gasfrac'] = gasfrac
DMG['gasmass'] = gasmass
DMG['rhoDMG'] = rhoDMG
DMG['rhocritDMG'] = rhocritDMG
DMG['luminosity'] = luminosity
DMG['rhoGTrhocritindex'] = rhoGTrhocritindex

with open('DMG_luminosity'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','wb') as f:
	pickle.dump(DMG, f)
