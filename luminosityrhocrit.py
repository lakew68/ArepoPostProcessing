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

from headerInfo import *
from gasDomIdx import *
from yeoushrinker14Mpc_Sig0_10 import *
from yeoushrinker14Mpc_118kms_10 import *
from yeoushrinker14Mpc_Sig0_22 import *
from yeoushrinker14Mpc_118kms_22 import *

from match14Mpc_Sig0 import *
from match14Mpc_118kms import *

params = {
	'axes.labelsize' : 24,
	'font.size': 24,
	'xtick.labelsize': 24,
	'ytick.labelsize': 24,
	'figure.figsize' : [8,8],
	'legend.fontsize': 24
	}
matplotlib.rcParams.update(params)


np.random.seed(42)

#Run parameters
'''
res = '1.4Mpc'
s_res = '14Mpc'
vel = '11.8kms'
s_vel = '118kms'
snapnum = 10
'''
'''
res = '1.4Mpc'
s_res = '14Mpc'
vel = 'Sig0'
s_vel = 'Sig0'
snapnum = 10
'''


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


#Calc rhocrit for SIGO



prefix = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_"
run = "1.4Mpc_11.8kms"
cat = readsubfHDF5.subfind_catalog("/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_1.4Mpc_11.8kms/output/GasOnly_FOF",snapnum)
path = prefix+run+'/output/GasOnly_FOF/'
s = gadget_readsnap( snapnum, snappath=path, hdf5=True, loadonly=['pos','vel', 'mass', 'vol', 'rho', 'u'], loadonlytype=[0], forcesingleprec=False )
cms = cat.GroupPos
cms = cms / (s.hubbleparam / s.time)
cvel = cat.GroupVel
cvel = cvel / s.time

#boxSize hubble flow correction for halo CM velocity subtraction
boxSizeVel = s.boxsize * .1 * UnitLength_in_cm/UnitVelocity_in_cm_per_s * np.sqrt(s.omega0/s.time/s.time/s.time + s.omegalambda)

filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
########## CHANGED FILENAME3 TO GROUPORDERED IN GAS ONLY
filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
rGas = snapHDF5.read_block(filename3,"RHO ",parttype=0) 
pDM= snapHDF5.read_block(filename3,"POS ",parttype=1)


halo100_indices= np.where(cat.GroupLenType[:,0] >100)[0]		
startAllGas = []
endAllGas   = []
for i in halo100_indices:
	startAllGas += [np.sum(cat.GroupLenType[:i,0])]
	endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]

#get SIGO indices
gasDomIdx = gasDomIdxAll[s_vel+'_'+str(snapnum)]
#Initialize arrays
#some radii are errors and  negative, will have a value of 1 to be excluded
negradii = np.zeros(np.size(halo100_indices)) 
densradius = np.zeros(np.size(halo100_indices))
maxrho = np.zeros(np.size(halo100_indices))
totrho = np.zeros(np.size(halo100_indices))
gasfrac = np.zeros(np.size(halo100_indices))
gasmass = np.zeros(np.size(halo100_indices))
rmin =  np.zeros(halo100_indices.size)

rhocritGP = np.zeros(halo100_indices.size)
rhoGP = np.zeros(halo100_indices.size)

for i in halo100_indices:
	exec("cm = cm_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("rotation = rotation_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("radii = radii_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("mDM=mDM_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
	exec("DMinEll=DMindices_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))	
	exec("Rclosest=Rmin_%s_%s[snapnum-10][i]"%(s_res,s_vel))
	exec("R200dm=R200dm_%s_%s[snapnum-10][i]"%(s_res,s_vel))
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
		Pdm = pDM[DMinEll]
		# Shift coordinate system to center on the center of the ellipsoid
		Precentered = dx_wrap(P - cm,s.boxsize)
		PrecenteredDM = dx_wrap(Pdm - cm,s.boxsize)
		# Rotate coordinated to the the axes point along x,y,z directions:
		Precentered = np.array([np.dot(pp, rotation.T) for pp in Precentered])
		PrecenteredDM = np.array([np.dot(pp, rotation.T) for pp in PrecenteredDM])

		# Figure out which particles are inside the ellipsoid
		inEll = (Precentered[:,0]**2./radii[0]**2. + Precentered[:,1]**2./radii[1]**2 + Precentered[:,2]**2./radii[2]**2)<=1.



		#s.data['pos'] = s.pos[igas]
		#s.data['type'] = s.type[igas]
		#s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10 * MSUN / KPC**2.0 / mu / PROTONMASS * boxsize / res #put in cm^-2
		#s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10  *MSUN/ KPC**2.0 * boxsize / res #put in g/cm^2

		#s.data['vol'] = s.vol[igas]
		#s.data['mass'] = s.mass[igas]
		#convert internal energy to temperature 
		u = 1.0e10 * s.data['u'][startAllGas[i]: endAllGas[i]][inEll]#it's a velocity squared to be converted in cgs
		'''
		ne = s.data['ne'][:]
		metallicity  = 0 
		XH = s.data['gmet'][:, 0]
		yhelium = (1 - XH - metallicity) / (4. * XH);
		mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
		'''
		mu = 1.22 #Primordial composition 
		temp = GAMMA_MINUS1 /  BOLTZMANN * u * PROTONMASS * mu	
			
		#turn into sound speed
		cs = np.sqrt(GAMMA * BOLTZMANN * temp / mu / PROTONMASS)
		tempvel = (s.vel[startAllGas[i]: endAllGas[i]][inEll] - cvel[i]) * 1.0e5 #convert to cgs
		#Turn first v_x into |v|
		velmag = np.linalg.norm(tempvel,axis=1)
		#turn into mach number
		mach = velmag / cs	


		#print "Mach number: ", (np.mean(mach))
		#using rmax
		L_cloud = (2.*radii[2]*s.time/s.hubbleparam*KPC) #in cm
		#L_cloud = ((radii[2]+radii[0])*s.time/s.hubbleparam*KPC) #use avg of rmin/rmax
			
		rhocritGP[i] = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / L_cloud**2.
		#using rmin	
		#rhocrit = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / (radii[0]*s.time/s.hubbleparam*KPC)**2.

		#print "rho_crit: ", rhocrit, "g/cm^3"
		
		SIGOrho = np.sum(M[inEll]) /  (4./3. * np.pi * radii[0]*radii[1]*radii[2]) / (s.time**3 / s.hubbleparam**2) * 1.e10 * MSUN / KPC **3.
		SIGOrhocell = np.mean(R[inEll]) * 1.0e10  *MSUN/ KPC**3.0 / (s.time**3. / s.hubbleparam**2)
		rhoGP[i] = SIGOrho	
		#print "SIGO rho: ", SIGOrho, "g/cm^3"
		#print "SIGO rho cell: ",SIGOrhocell, "g/cm^3"

		#Calculate jeans length/sonic length

		
		lamb_sonic = L_cloud / np.mean(mach)**2.
		lamb_jeans = np.sqrt(np.pi * np.mean(cs)**2. / GRAVITY_cgs / SIGOrho)

		#print "L_cloud: ",L_cloud,"cm"
		#print "lamb sonic: ",lamb_sonic,'cm'
		#print "lamb jeans: ",lamb_jeans,"cm"
		#print "L_s / L_J: ", lamb_sonic/lamb_jeans 
		
		m_BE = 1.18 / np.pi**(1.5) * SIGOrho * lamb_jeans**3.
		SIGOmass = M[inEll].sum() * 1.e10 / s.hubbleparam
		#print "Bonner Ebert mass: ", m_BE/MSUN, 'MSUN'
		#print "SIGO mass: ", SIGOmass, "MSUN"
		gasmass[i] = np.sum(M[inEll])
		gasfrac[i] = np.sum(M[inEll])/(np.sum(M[inEll])+mDM)
		#Make stellar density to  long axis
		#densradius[i] = .5 * (radii[0] + radii[2])
		densradius[i] = radii[2]
		rmin[i] = radii[0]
		totrho[i] =  np.mean(R[inEll])
	#Adjust for bad ellipsoids

nonzero, = np.where(negradii==0.)
#GasDom
if vel == '11.8kms':	
	densradiusGD = densradius[gasDomIdx] * s.time /.71
	rminGD = rmin[gasDomIdx] * s.time /.71
	maxrhoGD = maxrho[gasDomIdx] *10.**10/(s.time**3./.71**2)
	totrhoGD = totrho[gasDomIdx] *10.**10/(s.time**3./.71**2)
	gasfracGD = gasfrac[gasDomIdx]
	gasmassGD = gasmass[gasDomIdx] *10.**10/.71
		
	#stellardensGD = f_star * totrhoGPGD
	#stellarmassGD = 4./3. * np.pi * densradiusGPGD**3. * stellardensGD
	#Try using just the gas mass
	stellarmassGD = f_star * gasmassGD

densradius = densradius[nonzero]
rmin = rmin[nonzero]
totrho = totrho[nonzero]
gasfrac = gasfrac[nonzero]
gasmass = gasmass[nonzero]
densradius *= s.time / .71 #convert to kpc
rmin *= s.time / .71
#maxrho *= 10.**10*MSUN/KPC**3/PROTONMASS/1.22* 1.0 / (s.time**3 /.71**2) #convert to cm^-3
#maxrho *= 10.**10/(s.time**3./.71**2) #convert to Msun/kpc^3
totrho *= 10.**10/(s.time**3./.71**2) #convert to Msun/kpc^3
gasmass *= 10.**10/.71

#stellardens = f_star * totrhoGP 
#stellarmass = 4./3. * np.pi * densradiusGP**3. * stellardens
stellarmass = f_star * gasmass


Q_Hmassive = 10.**(Q_H[IMF][Z]) * stellarmass #unnormalize from one stellar mass
#ZAMS luminosityGP
luminosity = c_LA[T] * (1.-f_esc) * Q_Hmassive
#Gas Dominated
if vel == '11.8kms':
	Q_HmassiveGD = 10.**(Q_H[IMF][Z]) * stellarmassGD #unnormalize from one stellar mass
	#ZAMS luminosityGP
	luminosityGD = c_LA[T] * (1.-f_esc) * Q_HmassiveGD

rhoSIGO = rhoGP[gasDomIdx]
rhocritSIGO = rhocritGP[gasDomIdx]
rhoGP = rhoGP[nonzero]
rhocritGP = rhocritGP[nonzero]



rhoGTrhocritindex, = np.where(rhoGP > rhocritGP)

rhoGTrhocritANDSIGO, = np.where(rhoSIGO > rhocritSIGO) 
#print rhoGTrhocritANDSIGO
#print rhoGTrhocritANDSIGO.size #works

#In theory can now just splice this in the luminsoity plotting, no need to do the luminosity
#print "import numpy as np"
#print "rhoGTrhocritindex = ", list(rhoGTrhocritindex)
#print "rhoGTrhocritANDSIGO = ", list(rhoGTrhocritANDSIGO)
"""

#plot luminosity vs radius

fig, ax = plt.subplots(figsize=(10,8))
#im = ax.scatter(DMG.densradius[rhoGTrhocritANDgood],DMG.luminosity[rhoGTrhocritANDgood],c=DMG.gasfrac[rhoGTrhocritiANDgood],label='DM/G',vmin=0.,vmax=.25)
#ax.scatter(GP.luminosity,GP.densradius,c=GP.gasfrac,marker='^',label='GP',alpha=.75,vmin=0,vmax=.25)
if vel == '11.8kms':
	#ax.scatter(GP.densradiusGD,GP.luminosityGD,c=GP.gasfracGD,marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)
	im = ax.scatter(densradiusGD[rhoGTrhocritANDSIGO],luminosityGD[rhoGTrhocritANDSIGO],c=gasfracGD[rhoGTrhocritANDSIGO],marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)




ax.legend(loc=2)
ax.set_xlim([0.,.4])
#print np.mean(GP.luminosityGD) #2.3e39
ax.set_ylim([5.0e38,2.3e40])
#ax.set_ylim([1.e39,1.e41])
ax.set_yscale('log')
ax.tick_params(axis='both',which='minor',length=6)
ax.tick_params(axis='both',which='major',length=8)
ax.tick_params(which='both',direction='in',top=True,right=True)


fig.subplots_adjust(right=.8)
cbar_ax=fig.add_axes([.81,.15,.05,.7])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label("Gas Fraction")

ax.set_xlabel("Radius (kpc)")
#ax.set_xlabel("Luminosity ($L_{\odot}$)")
ax.set_ylabel("Luminosity (erg/s)")
#plt.savefig('./plots/luminosityDMGasNoPeaks_'+s_vel+'.pdf')
plt.show()
"""

#DMG


prefix = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_"
run = "1.4Mpc_11.8kms"

name = 'clump'
cat = readsubfHDF5.subfind_catalog("/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_1.4Mpc_11.8kms/output/DM_FOF",snapnum)
path = prefix+run+'/output/'
s = gadget_readsnap( snapnum, snappath=path, snapbase='snap_',hdf5=True, loadonly=['pos','vel', 'mass', 'vol', 'rho', 'u'], loadonlytype=[0], forcesingleprec=False )
cms = cat.GroupPos
cms = cms / (s.hubbleparam / s.time)
cvel = cat.GroupVel
cvel = cvel / s.time

filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
########## CHANGED FILENAME3 TO GROUPORDERED IN GAS ONLY
filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
rGas = snapHDF5.read_block(filename3, "RHO ",parttype=0)
pDM= snapHDF5.read_block(filename3,"POS ",parttype=1)
mDM= snapHDF5.read_block(filename3,"MASS", parttype=1)

#boxSize hubble flow correction for halo CM velocity subtraction
boxSizeVel = s.boxsize * .1 * UnitLength_in_cm/UnitVelocity_in_cm_per_s * np.sqrt(s.omega0/s.time/s.time/s.time + s.omegalambda)

rhocritDMG = np.zeros(cat.Group_R_Crit200.size)
rhoDMG = np.zeros(cat.Group_R_Crit200.size)
densradius= np.zeros(cat.Group_R_Crit200.size)
maxrho = np.zeros(cat.Group_R_Crit200.size)
totrho = np.zeros(cat.Group_R_Crit200.size)
gasmass = np.zeros(cat.Group_R_Crit200.size)
gasfrac = np.zeros(cat.Group_R_Crit200.size)



#load particle indices

goodidx, indgas, inddm = np.load('particleindex_1.4Mpc_11.8kms_10.npy')
goodidx = goodidx.astype(int)

for i,j in enumerate(goodidx):
	if indgas[i].size > 100: #good idx doesn't know about how many gas particles there are
		tempposgas = dx_wrap(pGas[indgas[i]] - cms[j] * (s.hubbleparam / s.time),s.boxsize)
		RHO = rGas[indgas[i]]	
		totrho[i] = np.mean(RHO) 
		gasfrac[i] = mGas[indgas[i]].sum()/(mGas[indgas[i]].sum()+mDM[inddm[i]].sum())  
		gasmass[i] = mGas[indgas[i]].sum()
		densradius[i] = cat.Group_R_Crit200[j]	

		#over100global += 1
		#s.data['pos'] = s.pos[igas]
		#s.data['type'] = s.type[igas]
		#s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10 * MSUN / KPC**2.0 / mu / PROTONMASS * boxsize / res #put in cm^-2
		#s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10  *MSUN/ KPC**2.0 * boxsize / res #put in g/cm^2

		#s.data['vol'] = s.vol[igas]
		#s.data['mass'] = s.mass[igas]
		#convert internal energy to temperature 
		u = 1.0e10 * s.data['u'][indgas[i]]#it's a velocity squared to be converted in cgs
		'''
		ne = s.data['ne'][:]
		metallicity  = 0 
		XH = s.data['gmet'][:, 0]
		yhelium = (1 - XH - metallicity) / (4. * XH);
		mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
		'''
		mu = 1.22 #Primordial composition 
		temp = GAMMA_MINUS1 /  BOLTZMANN * u * PROTONMASS * mu	
			
		#turn into sound speed
		cs = np.sqrt(GAMMA * BOLTZMANN * temp / mu / PROTONMASS)
		tempvel = dx_wrap(s.vel[indgas[i]] - cvel[j],boxSizeVel) * 1.0e5 #convert to cgs
		#Turn first v_x into |v|
		velmag = np.linalg.norm(tempvel,axis=1)
		#turn into mach number
		mach = velmag / cs	


		#print "Mach number: ", (np.mean(mach))
		#using rmax
		#L_cloud = (2.*radii[2]*s.time/s.hubbleparam*KPC) #in cm
		L_cloud = 2.*(cat.Group_R_Crit200[j]*s.time/s.hubbleparam*KPC) 
			
		rhocritDMG[i] = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / L_cloud**2.
		#using rmin	
		#rhocrit = np.pi * np.mean(cs)**2. * np.mean(mach)**4. / GRAVITY_cgs / (radii[0]*s.time/s.hubbleparam*KPC)**2.

		#print "rho_crit: ", rhocrit, "g/cm^3"
		'''	
		SIGOrho = np.sum(M[inEll]) /  (4./3. * np.pi * radii[0]*radii[1]*radii[2]) / (s.time**3 / s.hubbleparam**2) * 1.e10 * MSUN / KPC **3.
		SIGOrhocell = np.mean(R[inEll]) * 1.0e10  *MSUN/ KPC**3.0 / (s.time**3. / s.hubbleparam**2)
		
		print "SIGO rho: ", SIGOrho, "g/cm^3"
		print "SIGO rho cell: ",SIGOrhocell, "g/cm^3"
		'''	
		rhoDMG[i] = np.sum(s.mass[indgas[i]]) / (4./3. * np.pi * cat.Group_R_Crit200[j])/(s.time**3 / s.hubbleparam**2) * 1.e10 * MSUN / KPC **3.
		#print "DMG rho: ", DMGrho, "g/cm^3"
		#Calculate jeans length/sonic length

		
		#lamb_sonic = L_cloud / np.mean(mach)**2.
		#lamb_jeans = np.sqrt(np.pi * np.mean(cs)**2. / GRAVITY_cgs / DMGrho)
densradius *= s.time / .71
totrho *= 10.**10/(s.time/.71**2)
gasmass *= 10.**10/.71

densradius = densradius[goodidx]
gasfrac = gasfrac[goodidx]
totrho = totrho[goodidx]
gasmass = gasmass[goodidx]
#maxrho = maxrho[densradius!=0.]
rhoDMG = rhoDMG[goodidx]
rhocritDMG = rhocritDMG[goodidx]



gasmass = gasmass[densradius!=0.]
gasfrac = gasfrac[densradius!=0.]
totrho = totrho[densradius!=0.]
rhoDMG = rhoDMG[densradius!=0.]
rhocritDMG = rhocritDMG[densradius!=0.]
densradius = densradius[densradius!=0.]

#stellardens = f_star * totrho
#stellarmass = 4./3. * np.pi * densradius**3. * stellardens

stellarmass = f_star * gasmass
Q_Hmassive = 10.**(Q_H[IMF][Z]) * stellarmass #unnormalize from one stellar mass

#ZAMS luminosity
luminosity = c_LA[T] * (1.-f_esc) * Q_Hmassive

rhoGTrhocritindex, = np.where(rhoDMG > rhocritDMG)
#rhoGTrhocritANDgood = np.intersect1d(rhoGTrhocritindex, goodidx)

#print luminosity[rhoGTrhocritindex] 
#print rhoGTrhocritindex.size
#print rhoGTrhocritindex
#print rhoGTrhocritANDgood.size 

#print "import numpy as np"
#print "rhoGTrhocritindexDMG = ", list(rhoGTrhocritindex)
#print "rhoGTrhocritANDgood = ", list(rhoGTrhocritANDgood)

#plot luminosity vs radius
"""
fig, ax = plt.subplots(figsize=(10,8))
im = ax.scatter(densradius[rhoGTrhocritindex],luminosity[rhoGTrhocritindex],c=gasfrac[rhoGTrhocritindex],label='DM/G',vmin=0.,vmax=.25)
#ax.scatter(GP.luminosity,GP.densradius,c=GP.gasfrac,marker='^',label='GP',alpha=.75,vmin=0,vmax=.25)
if vel == '11.8kms':
	#ax.scatter(GP.densradiusGD,GP.luminosityGD,c=GP.gasfracGD,marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)
	ax.scatter(np.random.rand(1)[0]*(densradiusGD[rhoGTrhocritANDSIGO]-rminGD[rhoGTrhocritANDSIGO])+rminGD[rhoGTrhocritANDSIGO],luminosityGD[rhoGTrhocritANDSIGO],c=gasfracGD[rhoGTrhocritANDSIGO],marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)




ax.legend(loc=2)
ax.set_xlim([0.,.4])
#print np.mean(GP.luminosityGD) #2.3e39
ax.set_ylim([5.0e38,2.3e40])
#ax.set_ylim([1.e39,1.e41])
ax.set_yscale('log')
ax.tick_params(axis='both',which='minor',length=6)
ax.tick_params(axis='both',which='major',length=8)
ax.tick_params(which='both',direction='in',top=True,right=True)


fig.subplots_adjust(right=.8)
cbar_ax=fig.add_axes([.81,.15,.05,.7])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label("Gas Fraction")

ax.set_xlabel("Radius (kpc)")
#ax.set_xlabel("Luminosity ($L_{\odot}$)")
ax.set_ylabel("Luminosity (erg/s)")
#plt.savefig('./plots/luminosityRhoCrit_'+s_vel+'randomrmaxrmin.pdf')
plt.show()
"""

print 'fraction of DM/G greater than rho crit: '+str(1.*np.size(densradius[rhoGTrhocritindex])/np.size(densradius))
print 'number of DM/G greater than rho crit: '+str(np.size(densradius[rhoGTrhocritindex]))

# plot mag vs radius

fig, ax = plt.subplots(figsize=(10,8))
DMGmag = -2.5 * np.log10(luminosity[rhoGTrhocritindex] * 10.**-7) + 71.197425

im = ax.scatter(densradius[rhoGTrhocritindex],DMGmag,c=gasfrac[rhoGTrhocritindex],label='DM/G',vmin=0.,vmax=.25)
#ax.scatter(GP.luminosity,GP.densradius,c=GP.gasfrac,marker='^',label='GP',alpha=.75,vmin=0,vmax=.25)
if vel == '11.8kms':
	GPGDmag = -2.5 * np.log10(luminosityGD[rhoGTrhocritANDSIGO] * 10.**-7) + 71.197425

	#ax.scatter(GP.densradiusGD,GP.luminosityGD,c=GP.gasfracGD,marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)
	ax.scatter(np.random.rand(1)[0]*(densradiusGD[rhoGTrhocritANDSIGO]-rminGD[rhoGTrhocritANDSIGO])+rminGD[rhoGTrhocritANDSIGO],GPGDmag,c=gasfracGD[rhoGTrhocritANDSIGO],marker='*',s=500,label='SIGO',alpha=.5,vmin=0,vmax=.25)




ax.legend(loc=4)
ax.set_xlim([0.,.4])
#print np.mean(GP.luminosityGD) #2.3e39
#ax.set_ylim([5.0e38,2.3e40])
ax.set_ylim([-7.8,-15])
#ax.set_yscale('log')
ax.tick_params(axis='both',which='minor',length=6)
ax.tick_params(axis='both',which='major',length=8)
ax.tick_params(which='both',direction='in',top=True,right=True)


fig.subplots_adjust(right=.8)
cbar_ax=fig.add_axes([.81,.15,.05,.7])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label("Gas Fraction")

ax.set_xlabel("Radius (kpc)")
#ax.set_xlabel("Luminosity ($L_{\odot}$)")
ax.set_ylabel("Bolometric magnitude")
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
#plt.savefig('./plots/magnitudeRhoCrit_'+s_vel+'randomrmaxrmin.pdf')
plt.show()

