"""
Program to calculate gas ellipsoid spin parameter with class structure

This version includes DM indices 

This version includes new shrinker data

only snapshots 10 and 22 are supported

This uses the exec command instead of hardcoding

This version creates spinclassellipsoid and pickle data files for plot
6/13/18: Added if name == main functionality, don't need to comment out code
pickled files will be created/saved if this program itself is run.

"""
import matplotlib
#matplotlib.use('Agg')
from sys import argv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as stats
import readsubfHDF5
import snapHDF5 as snap
import hdf5lib
try:
	import cPickle as pickle
except:
	import pickle

from headerInfo import *

#Depreciated Cristina data
#from ellipsoidFit_noRho_112Mpc_118kms import *
#from ellipsoidFit_noRho_112Mpc_Sig0 import *
#from ellipsoidFit_noRho_14Mpc_118kms import *
#from ellipsoidFit_noRho_14Mpc_Sig0 import *

from yeoushrinker112Mpc_Sig0_10 import *
from yeoushrinker112Mpc_118kms_10 import *
from yeoushrinker112Mpc_Sig0_22 import *
from yeoushrinker112Mpc_118kms_22 import *
from yeoushrinker14Mpc_Sig0_10 import *
from yeoushrinker14Mpc_118kms_10 import *
from yeoushrinker14Mpc_Sig0_22 import *
from yeoushrinker14Mpc_118kms_22 import *

from match112Mpc_Sig0 import *
from match112Mpc_118kms import *
from match14Mpc_Sig0 import *
from match14Mpc_118kms import *



def dx_wrap(dx,box):
	idx = dx > +box/2.0
	dx[idx] -= box
	idx = dx < -box/2.0
	dx[idx] += box 
	return dx
def dist2(dx,dy,dz,box):
	return dx_wrap(dx,box)**2 + dx_wrap(dy,box)**2 + dx_wrap(dz,box)**2
class spinparamellipsoid(object):
	def __init__(self, res, vel, snapnum):
		#self.res = res
		#self.vel = vel
		#self.snapnum = snapnum
		if res == "1.12Mpc":
			s_res = '112Mpc'
		elif res == "1.4Mpc":
			s_res = '14Mpc'
		if vel == "Sig0":
			s_vel = "Sig0"
		elif vel == "11.8kms":
			s_vel = '118kms'
		snapnum = int(snapnum) 

		filename = "/n/hernquistfs3/mvogelsberger/GlobularClusters/InterfaceWArepo_All_" + res + '_' + vel  + "/output/"
		filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
		########## CHANGED FILENAME3 TO GROUPORDERED IN GAS ONLY
		filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for hdf5lib, snapHDF5
		#### Not sure if this works with change but don't care about 2.8
		if res == '2.8Mpc':
			filename3 = filename + "snapdir_" + str(snapnum).zfill(3)+"/snap_" + str(snapnum).zfill(3)  

		#Units
		GRAVITY_cgs = 6.672e-8
		UnitLength_in_cm = 3.085678e21 # code length unit in cm/h
		UnitMass_in_g = 1.989e43       # code length unit in g/h
		UnitVelocity_in_cm_per_s = 1.0e5
		UnitTime_in_s= UnitLength_in_cm / UnitVelocity_in_cm_per_s
		UnitDensity_in_cgs= UnitMass_in_g/ np.power(UnitLength_in_cm,3)
		UnitPressure_in_cgs= UnitMass_in_g/ UnitLength_in_cm/ np.power(UnitTime_in_s,2)
		UnitEnergy_in_cgs= UnitMass_in_g * np.power(UnitLength_in_cm,2) / np.power(UnitTime_in_s,2)
		GCONST=GRAVITY_cgs/ np.power(UnitLength_in_cm,3) * UnitMass_in_g *  np.power(UnitTime_in_s,2)
		critical_density = 3.0*.1 * .1 / 8.0/np.pi/GCONST #.1 is for 1/Mpc to 1/kpc, also in units of h^2

		header = snap.snapshot_header(filename3)
		if res == "2.8Mpc":
			fs = hdf5lib.OpenFile(filename3 + ".0.hdf5")
		else:
			fs = hdf5lib.OpenFile(filename3 + ".hdf5")
		red = hdf5lib.GetAttr(fs, "Header", "Redshift")
		atime = hdf5lib.GetAttr(fs, "Header", "Time")
		boxSize = hdf5lib.GetAttr(fs, "Header", "BoxSize")
		boxSize *= atime #convert from ckpc/h to kpc/h
		Omega0 = hdf5lib.GetAttr(fs, "Header", "Omega0")
		OmegaLambda = hdf5lib.GetAttr(fs, "Header", "OmegaLambda")
		fs.close()
		cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
		Omega_a = Omega0/(Omega0 + OmegaLambda * atime * atime * atime)
		critical_density *= (Omega0/Omega_a)
		r200 = cat.Group_R_Crit200
		r200 *= atime #convert from ckpc/h to kpc/h
		m200 = cat.Group_M_Crit200
		haloCMvel = cat.GroupVel
		haloCMvel *= 1./atime #convert from km/s/a to km/s
		haloPos = cat.GroupPos
		haloPos *= atime #convert from ckpc/h to kpc/h

		#Read in particles
		#read in all simulation masses to calculate cosmic baryon fraction
		massgassim = snap.read_block(filename + "snap_" + str(snapnum).zfill(3), "MASS", parttype=0)
		massdmsim =  snap.read_block(filename + "snap_" + str(snapnum).zfill(3), "MASS", parttype=1)
		massgas = snap.read_block(filename3, "MASS", parttype=0)
		massdm = snap.read_block(filename3, "MASS", parttype=1)
		posgas = snap.read_block(filename3, "POS ", parttype=0)
		posdm = snap.read_block(filename3, "POS ", parttype=1)
		velgas = snap.read_block(filename3, "VEL ", parttype=0)
		veldm = snap.read_block(filename3, "VEL ", parttype=1)
		#redefine position units from ckpc/h to kpc/h
		posgas *= atime
		posdm *= atime
		#redefine velocity units from kmsqrt(a)/s to km/s
		velgas *= np.sqrt(atime)
		veldm *= np.sqrt(atime)

		fb = massgassim.sum(dtype="float64")/(massgassim.sum(dtype="float64")+massdmsim.sum(dtype="float64"))
		gaslimit = .4 # Set the limit for gas fraction in plots

		#boxSize hubble flow correction for halo CM velocity subtraction
		boxSizeVel = boxSize * .1 * UnitLength_in_cm/UnitVelocity_in_cm_per_s * np.sqrt(Omega0/atime/atime/atime + OmegaLambda)

		#load particle indices
		pGas= snap.read_block(filename3,"POS ", parttype=0)
		mGas= snap.read_block(filename3,"MASS", parttype=0)
		pDM= snap.read_block(filename3,"POS ",parttype=1)
		halo100_indices= np.where(cat.GroupLenType[:,0] >100)[0]		
		startAllGas = []
		endAllGas   = []
		for i in halo100_indices:
			startAllGas += [np.sum(cat.GroupLenType[:i,0])]
			endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]
		#Initialize arrays
		spinparam = np.zeros(np.size(halo100_indices))
		jsptotspinparam = np.zeros(np.size(halo100_indices))
		jspgasspinparam = np.zeros(np.size(halo100_indices))
		jspdmspinparam = np.zeros(np.size(halo100_indices))
		gasfrac = np.zeros(np.size(halo100_indices))
		costheta = np.zeros(np.size(halo100_indices)) #misalignment angle
		v200 = np.zeros(np.size(halo100_indices))
		velgasall = np.zeros(np.size(halo100_indices)) 
		veldmall = np.zeros(np.size(halo100_indices))
		virialratio = np.zeros(np.size(halo100_indices))
		numGas = np.zeros(np.size(halo100_indices))
		numDM = np.zeros(np.size(halo100_indices)) 

		j200gas= np.zeros(np.size(halo100_indices)) 
		j200dm = np.zeros(np.size(halo100_indices)) 
		j200 = np.zeros(np.size(halo100_indices)) 
		totmass = np.zeros(np.size(halo100_indices)) 
		gasmass = np.zeros(np.size(halo100_indices)) 
		DMmass = np.zeros(np.size(halo100_indices)) 
		rmax = np.zeros(np.size(halo100_indices)) 
		rmin = np.zeros(np.size(halo100_indices)) 
		j200gasNoNorm = np.zeros(np.size(halo100_indices)) 
		closestm200 = np.zeros(np.size(halo100_indices)) 
		#some radii are errors and  negative, will have a value of 1 to be excluded
		negradii = np.zeros(np.size(halo100_indices)) 


		#Indexing for global variable works because halos are ordered from largest to smallest so <100 particles are at the end and not counted.
		for i in halo100_indices:
			exec("cm = cm_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
			exec("rotation = rotation_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
			exec("radii = radii_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
			#some radii are errors and  negative, will have a value of 1 to be excluded
			if radii[0] < 0.:
				negradii[i]=1.
			else:
				maxrad = radii[2]
				maxrad *= atime #convert from ckpc to kpc
				exec("mDM=mDM_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
				exec("DMinEll=DMindices_%s_%s_%d[0][i]"%(s_res,s_vel,snapnum))
				exec("m200dm = M200dm_%s_%s[snapnum-10][i]"%(s_res,s_vel))	
				#Check if CM is buggy
				if np.sum(cm == np.array([0., 0., 0.]))==3:
					# it's probbaly an error; recompute com	
					totalGas = np.sum(mGas[startAllGas[i]: endAllGas[i]])
					cm = np.array([np.sum(pGas[startAllGas[i]: endAllGas[i], j]*mGas[startAllGas[i]: endAllGas[i]])/totalGas for j in range(3)])



				# Get positions of gas particles
				P = pGas[startAllGas[i]: endAllGas[i]]
				# Shift coordinate system to center on the center of the ellipsoid
				Precentered = dx_wrap(P - cm,boxSize/atime)
				# Rotate coordinated to the the axes point along x,y,z directions:
				Precentered = np.array([np.dot(pp, rotation.T) for pp in Precentered])
				# Figure out which particles are inside the ellipsoid
				inEll = (Precentered[:,0]**2./radii[0]**2. + Precentered[:,1]**2./radii[1]**2 + Precentered[:,2]**2./radii[2]**2)<=1.
				
				#remove halo CM velocity
				tempvelgas = dx_wrap(velgas[startAllGas[i]: endAllGas[i]][inEll] - haloCMvel[i],boxSizeVel)
				tempveldm = dx_wrap(veldm[DMinEll] - haloCMvel[i],boxSizeVel)
				#redefine positions wrt COM
				tempposgas = dx_wrap(posgas[startAllGas[i]: endAllGas[i]][inEll] - haloPos[i],boxSize)
				tempposdm = dx_wrap(posdm[DMinEll] - haloPos[i],boxSize)
				numDM[i] = np.size(tempposdm)
				numGas[i] = np.size(tempposgas)
				#Calculating j200
				#j200 of all particles
				j200vecgas = np.sum(np.cross(tempposgas,tempvelgas)*massgas[startAllGas[i]: endAllGas[i]][inEll][:, np.newaxis],axis=0)
				j200vecdm = np.sum(np.cross(tempposdm,tempveldm)*massdm[DMinEll][:, np.newaxis],axis=0)
				#if np.size(tempveldm)!=0: #can be no dm particles!
				#	costheta[i] = np.dot(j200vecgas,j200vecdm)/np.linalg.norm(j200vecgas)/np.linalg.norm(j200vecdm)
				j200vec = j200vecgas + j200vecdm
				j200[i] = np.linalg.norm(j200vec)
				j200dm[i] = np.linalg.norm(j200vecdm)

				j200gas[i] = np.linalg.norm(j200vecgas)
				j200gasNoNorm[i] = np.linalg.norm(j200vecgas)
				gasmass[i] = np.sum(massgas[startAllGas[i]: endAllGas[i]][inEll])
				totmass[i] = gasmass[i]+mDM	
				DMmass[i] = mDM
				rmax[i] = radii[2] 
				rmin[i] = radii[0]
				closestm200[i] = m200dm
				#using fudicial m200~6mgas
				#get r200 from analytic formula in Barkana,Loeb 01 review
				if gasmass[i] !=0.: #Some ellpsoids fit nothing
					m200fid = 6.*gasmass[i]
					omgz = .27*atime**(-3.)/(.27*atime**(-3.)+.73)
					dfact = omgz-1.
					delc = 18.*np.pi**2.+82*dfact-39.*dfact**2.
					r200fid = .784*(m200fid*100.)**(1./3.)*(.27/omgz*delc/18./np.pi**2)**(-1./3.)*10*atime
					v200fid = np.sqrt(GCONST*(m200fid)/r200fid)
					j200gas[i] *= 1./np.sqrt(2)/(gasmass[i])/v200fid/r200fid
					j200[i] *= 1./np.sqrt(2)/(totmass[i])/v200fid/r200fid
					if mDM !=0.:
						j200dm[i] *= 1./np.sqrt(2)/mDM/v200fid/r200fid
					gasfrac[i] = gasmass[i]/totmass[i]


		#Reindex to account for shrunken ellipsoids with gas particles >100
			
		goodidx, = np.where(np.logical_and(numGas>100,negradii==0.))

		self.j200gas = j200gas[goodidx]
		self.j200dm = j200dm[goodidx]
		self.j200 = j200[goodidx]
		self.j200gasNoNorm = j200gasNoNorm[goodidx]
		self.gasfrac = gasfrac[goodidx]	
		self.totmass = totmass[goodidx]
		self.totmass *= 10**10
		#costheta = costheta[goodidx]
		self.rmax = rmax[goodidx]
		self.rmin = rmin[goodidx]
		#thetadeg = np.arccos(costheta)*180./np.pi
		self.gasmass = gasmass[goodidx]
		self.closestm200 = closestm200[goodidx]
	
		#Reindex the Rmin, R200_DM params
		exec("self.rclosest = Rmin_%s_%s[snapnum-10][goodidx]"%(s_res,s_vel))
		exec("self.R200dm = R200dm_%s_%s[snapnum-10][goodidx]"%(s_res,s_vel))
		#Depreciated from single plot code
		#exec("Rmin_%s_%s[snapnum-10]=Rmin_%s_%s[snapnum-10][goodidx]"%(s_res,s_vel,s_res,s_vel))
		#exec("R200dm_%s_%s[snapnum-10]=R200dm_%s_%s[snapnum-10][goodidx]"%(s_res,s_vel,s_res,s_vel))


if __name__ == '__main__':
	#If this program is run directly, pickled filse will be created/saved
	#load class instances for all runs 
	
	s112Mpc_Sig0_22 = spinparamellipsoid('1.12Mpc','Sig0',22)
	s112Mpc_118kms_22 = spinparamellipsoid('1.12Mpc','11.8kms',22)
	s112Mpc_Sig0_10 = spinparamellipsoid('1.12Mpc','Sig0',10)
	s112Mpc_118kms_10 = spinparamellipsoid('1.12Mpc','11.8kms',10)

	s14Mpc_Sig0_22 = spinparamellipsoid('1.4Mpc','Sig0',22)
	s14Mpc_118kms_22 = spinparamellipsoid('1.4Mpc','11.8kms',22)
	s14Mpc_Sig0_10 = spinparamellipsoid('1.4Mpc','Sig0',10)
	s14Mpc_118kms_10 = spinparamellipsoid('1.4Mpc','11.8kms',10)

	#Pickle files
	f = open('s112Mpc_Sig0_22_ellipsoid.dat', 'w')
	pickle.dump(s112Mpc_Sig0_22,f)
	f.close()
	f = open('s112Mpc_118kms_22_ellipsoid.dat', 'w')
	pickle.dump(s112Mpc_118kms_22,f)
	f.close()
	f = open('s112Mpc_Sig0_10_ellipsoid.dat', 'w')
	pickle.dump(s112Mpc_Sig0_10,f)
	f.close()
	f = open('s112Mpc_118kms_10_ellipsoid.dat', 'w')
	pickle.dump(s112Mpc_118kms_10,f)
	f.close()
	f = open('s14Mpc_Sig0_22_ellipsoid.dat', 'w')
	pickle.dump(s14Mpc_Sig0_22,f)
	f.close()
	f = open('s14Mpc_118kms_22_ellipsoid.dat','w')
	pickle.dump(s14Mpc_118kms_22,f)
	f.close()
	f = open('s14Mpc_Sig0_10_ellipsoid.dat','w')
	pickle.dump(s14Mpc_Sig0_10,f)
	f.close()
	f = open('s14Mpc_118kms_10_ellipsoid.dat','w')
	pickle.dump(s14Mpc_118kms_10,f)
	f.close()
	



### Old plotting scripts for single plots. Refer to plotting.py for update
#Plotting

#Plot difference between closest DM halo m200 and fudicial m200 from
#saying that the gas ellipsoid came from a halo of mass ~ 6*mgas
"""
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$6M_{gas}$ $(M_{\\odot})$')
ax.set_ylabel('Closest $M_{200,DM}$ $(M_{\\odot})$')
ax.scatter(closestm200*10**10,6.*gasmass*10**10,color='black')
ax.plot(ax.get_xlim(),ax.get_ylim(),linestyle='--')
#plt.savefig('./finalplots/closestm200'+s_res+'_'+s_vel+'_'+str(snapnum)+'.pdf')
plt.show()
"""
### Plotting spinparam vs mass
"""
### gas
gaslimit = .4
fig, ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')
ax.set_xscale('log')
ax.set_yscale('log')
#im = ax.scatter(totmass,j200gas,c=gasfrac,s=35,cmap=cm,marker='o',norm=colors.LogNorm(vmin=gasfrac.min(),vmax=gasfrac.max()))
im = ax.scatter(s1.totmass,s1.j200gas,c=s1.gasfrac,s=35,cmap=cm,marker='o',vmin=s1.gasfrac.min(),vmax=s1.gasfrac.max())

#Plot only those with gasfrac > gaslimit, Rmin/R200dm > 1.
'''
exec("totmassprime = totmass[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
exec("j200gasprime = j200gas[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
exec("gasfracprime = gasfrac[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
'''
totmassprime = s1.totmass[(s1.gasfrac>gaslimit) & (s1.rclosest/s1.R200dm>1.)]
j200gasprime = s1.j200gas[(s1.gasfrac>gaslimit) & (s1.rclosest/s1.R200dm>1.)]
gasfracprime = s1.gasfrac[(s1.gasfrac>gaslimit) & (s1.rclosest/s1.R200dm>1.)]

#ax.scatter(totmassprime,j200gasprime,c=gasfracprime,s=500,cmap=cm,marker='*',norm=colors.LogNorm(vmin=gasfrac.min(),vmax=gasfrac.max()))
ax.scatter(totmassprime,j200gasprime,c=gasfracprime,s=500,cmap=cm,marker='*',vmin=s1.gasfrac.min(),vmax=s1.gasfrac.max())


cbar = fig.colorbar(im,ax=ax)
cbar.set_label("Gas fraction")
ax.set_xlabel('$M$ $(M_{\odot})$')
ax.set_ylabel('$\\lambda_{gas}$')
#plt.savefig('./finalplots/jgas'+s_res+'_'+s_vel+'_'+str(snapnum)+'.pdf')
plt.show()
plt.clf()
"""
"""
### DM 
fig, ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')
ax.set_xscale('log')
ax.set_yscale('log')
im = ax.scatter(totmass,j200dm,c=gasfrac,s=35,cmap=cm,marker='o',norm=colors.LogNorm(vmin=10**-2.,vmax=.5))
cbar = fig.colorbar(im,ax=ax)
ax.set_xlabel('$M$ $(M_{\odot})$')
ax.set_ylabel('$\\lambda_{DM}$')
#plt.savefig('./finalplots/jdm'+s_res+'_'+s_vel+'_'+str(snapnum)+'_maxrad.pdf')
plt.show()
plt.clf()
"""


###Plotting spinparam dist

"""
#j200gas = j200gas[j200gas<1.] #Culling >1 darkmatter
fig, ax = plt.subplots()
bins = np.linspace(0,.4,20)
s, loc, scale = stats.lognorm.fit(j200gas,floc=0)
#print s
#print scale
ax.hist(j200gas,bins = bins, normed = True,histtype='step')
xmin = j200gas.min()
xmax = j200gas.max()
x = np.linspace(xmin,xmax,1000)
pdf = stats.lognorm.pdf(x,s,loc=0,scale=scale)
ax.set_xlim([0,.4])
ax.plot(x,pdf,'k')
ax.set_title('$\\bar{\\lambda}=%.3f$, $\\sigma =%.3f$ '%(scale,s))
ax.set_xlabel('$\\lambda_{gas}$')
ax.set_ylabel('$P(\\lambda_{gas})$')
#plt.savefig('./finalplots/jgas'+s_res+'_'+s_vel+'_'+str(snapnum)+'_lognormal.pdf')
#plt.show()
"""

### Plot rmax/rmin vs spinparam
"""
fig, ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')
im = ax.scatter(j200gas,rmax/rmin,c=gasfrac,vmin=gasfrac.min(),vmax=gasfrac.max())
ax.set_xscale('log')
ax.set_xlim([10**-3,10**1])
ax.set_ylim([0,30])
ax.set_xlabel("$\\lambda_{gas}$")
ax.set_ylabel("$\\frac{R_{max}}{R_{min}}$")
cbar = fig.colorbar(im,ax=ax)
cbar.set_label("Gas fraction")
exec("rmaxprime = rmax[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
exec("rminprime = rmin[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
exec("j200gasprime = j200gas[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))
exec("gasfracprime = gasfrac[(gasfrac > gaslimit) & (Rmin_%s_%s[snapnum-10]/R200dm_%s_%s[snapnum-10]>1.)]"%(s_res, s_vel, s_res, s_vel))

ax.scatter(j200gasprime,rmaxprime/rminprime,c=gasfracprime,s=200,marker='*',vmin=gasfrac.min(),vmax=gasfrac.max())
#plt.savefig("./finalplots/spinparamVsRmaxRmin_" + s_res + "_" + s_vel + "_" + str(snapnum) + ".pdf")
plt.show()
"""
"""
#Plot histogram of rmax/rmin
fig, ax = plt.subplots()
bins = np.linspace(0,20,50)
ax.hist(rmax/rmin,bins = bins, normed = True,histtype='step')
plt.show()
"""
