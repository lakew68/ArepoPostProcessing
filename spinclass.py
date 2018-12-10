"""
Program to compute spin parameter 
spinparam is an object that calculates the spinparameter and other related quantities 

If program is run directly, pickled data will be created and saved.
"""
from __future__ import division
import matplotlib
#matplotlib.use('Agg')
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import readsubfHDF5
import snapHDF5 
import hdf5lib
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

class spinparam(object):
	def __init__(self, res, vel, snapnum):
		self.vel = vel
		self.res = res
		self.snapnum = int(snapnum)
		self.s_vel = vel.replace(".","")
		self.s_res = res.replace(".","")

		#File paths
		filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + self.res + '_' + self.vel  + "/output/"
		filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
		filename3 = filename + "snap_" + str(self.snapnum).zfill(3) #Used for hdf5lib, snapHDF5
		#Read header information	
		header = snapHDF5.snapshot_header(filename3)
		with hdf5lib.OpenFile(filename3 + ".hdf5") as fs:
			red = hdf5lib.GetAttr(fs, "Header", "Redshift")
			atime = hdf5lib.GetAttr(fs, "Header", "Time")
			boxSize = hdf5lib.GetAttr(fs, "Header", "BoxSize")
			boxSize *= atime / hubbleparam #convert from ckpc/h to kpc
			Omega0 = hdf5lib.GetAttr(fs, "Header", "Omega0")
			OmegaLambda = hdf5lib.GetAttr(fs, "Header", "OmegaLambda")
		
		#Read halo catalog
		cat = readsubfHDF5.subfind_catalog(filename2, self.snapnum)	
		#critical_density *= 1. / (Omega0 + OmegaLambda * atime * atime * atime) #redshift correction
		r200 = cat.Group_R_Crit200
		r200 *= atime / hubbleparam #convert from ckpc/h to kpc
		m200 = cat.Group_M_Crit200
		m200 *= 1. / hubbleparam #convert to 10^10 M_sun
		haloCMvel = cat.GroupVel
		haloCMvel *= 1. / atime #convert from km/s/a to km/s
		haloPos = cat.GroupPos
		haloPos *= atime / hubbleparam #convert from ckpc/h to kpc

		#Initialize arrays
		spinparamTotal = np.zeros(np.size(r200))
		spinparamGas = np.zeros(np.size(r200))
		spinparamDM = np.zeros(np.size(r200))
		gasfrac = np.zeros(np.size(r200))
		costheta = np.zeros(np.size(r200)) #misalignment angle
		v200 = np.zeros(np.size(r200))	
		numGas = np.zeros(np.size(r200))
		numDM = np.zeros(np.size(r200)) 

		#Read in particles
		massgas = snapHDF5.read_block(filename3, "MASS", parttype=0)
		massdm = snapHDF5.read_block(filename3, "MASS", parttype=1)
		posgas = snapHDF5.read_block(filename3, "POS ", parttype=0)
		posdm = snapHDF5.read_block(filename3, "POS ", parttype=1)
		velgas = snapHDF5.read_block(filename3, "VEL ", parttype=0)
		veldm = snapHDF5.read_block(filename3, "VEL ", parttype=1)
		#redefine position units from ckpc/h to kpc
		posgas *= atime / hubbleparam
		posdm *= atime / hubbleparam
		#redefine velocity units from kmsqrt(a)/s to km/s
		velgas *= np.sqrt(atime)
		veldm *= np.sqrt(atime)

		#boxSize hubble flow correction for halo CM velocity subtraction
		boxSizeVel = boxSize * hubbleparam * .1 * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)
		


		#load particle indices
		over300idx, indgas, inddm = np.load('particleindex_' + self.res + '_' + self.vel + '_' + str(self.snapnum) + '.npy')
		over300idx = over300idx.astype(int)
		over1 = []

		for i,j in enumerate(over300idx):
			#remove halo CM velocity
			tempvelgas = dx_wrap(velgas[indgas[i]] - haloCMvel[j],boxSizeVel)
			tempveldm = dx_wrap(veldm[inddm[i]] - haloCMvel[j],boxSizeVel)
			#redefine positions wrt COM
			tempposgas = dx_wrap(posgas[indgas[i]] - haloPos[j],boxSize)
			tempposdm = dx_wrap(posdm[inddm[i]] - haloPos[j],boxSize)
			numDM[j] = np.size(tempposdm)
			numGas[j] = np.size(tempposgas)
			#Calculating j200
			#j200 of all particles
			j200vecgas = np.sum(np.cross(tempposgas,tempvelgas)*massgas[indgas[i]][:, np.newaxis],axis=0)
			j200vecdm = np.sum(np.cross(tempposdm,tempveldm)*massdm[inddm[i]][:, np.newaxis],axis=0)
			if np.size(tempvelgas)!=0: #can be no gas particles!
				costheta[j] = np.dot(j200vecgas,j200vecdm)/np.linalg.norm(j200vecgas)/np.linalg.norm(j200vecdm)
			j200vec = j200vecgas + j200vecdm
			j200 = np.linalg.norm(j200vec)
			j200gas = np.linalg.norm(j200vecgas)
			j200dm = np.linalg.norm(j200vecdm)
			v200[j] = np.sqrt(GCONST*m200[j]/r200[j])
			
			#Bullock spin parameter
			totalmass = massgas[indgas[i]].sum(dtype='float64') + massdm[inddm[i]].sum(dtype='float64')
			spinparamTotal[j] = j200/np.sqrt(2)/v200[j]/r200[j]/totalmass
			if np.size(tempveldm)!=0: #tempveldm can be empty no dm particles!
				spinparamDM[j] = j200dm/np.sqrt(2)/v200[j]/r200[j]/massdm[inddm[i]].sum(dtype='float64')
			if np.size(tempvelgas)!=0: #tempvelgas can be empty no gas particles!
				spinparamGas[j] = j200gas/np.sqrt(2)/v200[j]/r200[j]/massgas[indgas[i]].sum(dtype='float64')
			gasfrac[j] = massgas[indgas[i]].sum(dtype='float64') / (massgas[indgas[i]].sum(dtype='float64') + massdm[inddm[i]].sum(dtype='float64'))

		#Reindex over300idx to account for SO halos with DM particles >300
		over300idx2 = over300idx[numDM[over300idx] > 300]

		#Plotting
		#Redfine in terms of over300idx2
		self.spinparamTotal = spinparamTotal[over300idx2]
		self.spinparamGas = spinparamGas[over300idx2]
		self.spinparamDM = spinparamDM[over300idx2]
		self.gasfrac = gasfrac[over300idx2]	
		self.m200 = m200[over300idx2]
		self.m200 *= 10**10  #Convert to solar mass.
		self.costheta = costheta[over300idx2]
		self.gasfracCosTheta = self.gasfrac[self.costheta!=0.]
		self.m2002 = self.m200[self.costheta!=0.]
		self.costheta = self.costheta[self.costheta!=0.] #take out the 0 gas components
		self.thetadeg = np.arccos(self.costheta)*180./np.pi
		


if __name__ == '__main__':
	#If this program is being run directly, create/save the pickled data files

	#load class instances for all runs
	"""	
	s112Mpc_Sig0_22 = spinparam('1.12Mpc','Sig0','22')
	s112Mpc_118kms_22 = spinparam('1.12Mpc','11.8kms','22')
	s112Mpc_Sig0_10 = spinparam('1.12Mpc', 'Sig0', '10')
	s112Mpc_118kms_10 = spinparam('1.12Mpc', '11.8kms', '10')
	"""
	#s14Mpc_Sig0_22 = spinparam('1.4Mpc','Sig0','22')
	#s14Mpc_118kms_22 = spinparam('1.4Mpc','11.8kms','22')
	#s14Mpc_Sig0_10 = spinparam('1.4Mpc', 'Sig0', '10')
	s14Mpc_118kms_10 = spinparam('1.4Mpc', '11.8kms', '10')


	#pickle files
	"""
	f = open('s112Mpc_Sig0_22.dat', 'w')
	pickle.dump(s112Mpc_Sig0_22,f)
	f.close()
	f = open('s112Mpc_118kms_22.dat', 'w')
	pickle.dump(s112Mpc_118kms_22,f)
	f.close()
	f = open('s112Mpc_Sig0_10.dat', 'w')
	pickle.dump(s112Mpc_Sig0_10,f)
	f.close()
	f = open('s112Mpc_118kms_10.dat', 'w')
	pickle.dump(s112Mpc_118kms_10,f)
	f.close()
	f = open('s14Mpc_Sig0_22.dat', 'w')
	pickle.dump(s14Mpc_Sig0_22,f)
	f.close()
	f = open('s14Mpc_118kms_22.dat','w')
	pickle.dump(s14Mpc_118kms_22,f)
	f.close()
	f = open('s14Mpc_Sig0_10.dat','w')
	pickle.dump(s14Mpc_Sig0_10,f)
	f.close()
	"""
	f = open('s14Mpc_118kms_10.dat','w')
	pickle.dump(s14Mpc_118kms_10,f)
	f.close()	

