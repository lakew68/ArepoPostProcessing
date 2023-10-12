"""
Program to make dictionary of gas, dm, and star IDs in each gas object at each redshift.
Returns: pickled file with dictionary containing gas IDs, dm IDs, and star IDs
Output is indexed by object in the format (snap number, object index in shrinker)

Added to Github 10/12/23
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
baryonfraction = .044 / .27 #OmegaB/Omega0
snapkey = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,151,152,153]

idDict = {}

for snapnum2 in snapkey:
    print('Snap: ', snapnum2)
    #Should be run with a snap number input
    res = '14Mpc'
    vel = 'Sig2'
    snapnum = int(snapnum2)
    s_vel = vel.replace(".","")
    s_res = res.replace(".","")

    #File paths
    filename = "D:/Star_Movie_768/"
    filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
    filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for snapHDF5

    print(filename3)
    
   
    with open(filename+'shrinker'+s_res+'_'+s_vel+'_'+str(snapnum)+'.dat','rb') as f:
        shrunken = pickle.load(f)
        
    with open(filename+'SIGOidx'+s_res+'_'+s_vel+'_'+str(snapnum)+'_fb0.6.dat','rb') as f:
        SIGOidx = pickle.load(f)
    
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
    pStar = snapHDF5.read_block(filename3,"POS ", parttype=4)
    mStar = snapHDF5.read_block(filename3,"MASS", parttype=4)
    pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
    idGas = snapHDF5.read_block(filename3,"ID  ", parttype=0)
    idStars = snapHDF5.read_block(filename3,"ID  ", parttype=4)
    idDM = snapHDF5.read_block(filename3,"ID  ", parttype=1)
    cat = readsubfHDF5.subfind_catalog(filename2, snapnum)

    halo100_indices= np.where((cat.GroupLenType[:,0]+cat.GroupLenType[:,4]) >100)[0]		
    startAllGas = []
    endAllGas   = []
    for i in halo100_indices:
        startAllGas += [np.sum(cat.GroupLenType[:i,0])]
        endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]

    haloPos = cat.GroupPos[halo100_indices]
    
    print(halo100_indices)
    
    for idx in halo100_indices:
        cm = shrunken['cm'][idx]
        rotation = shrunken['rotation'][idx]
        radii = shrunken['radii'][idx]
        gasindices = shrunken['gasindices'][idx]
        dmindices = shrunken['DMindices'][idx]
        starindices = shrunken['starindices'][idx]
        gasFrac = shrunken['gasFrac'][idx]
        
        isSIGO = idx in SIGOidx
        
        startGas = startAllGas[list(halo100_indices).index(idx)]
        endGas = endAllGas[list(halo100_indices).index(idx)]
        
        starIDs = []
        posStars = pStar[starindices]
        if len(starindices) > 0 and starindices[0] > 0:
            PrecenteredStar = dx_wrap(posStars - cm,boxSize)
            PrecenteredStar = np.array([np.dot(pp, rotation.T) for pp in PrecenteredStar])
            inEll = (PrecenteredStar[:,0]**2./radii[0]**2. + PrecenteredStar[:,1]**2./radii[1]**2 + PrecenteredStar[:,2]**2./radii[2]**2)<=1.
            starIDs = idStars[list(starindices[inEll])]
        else:
            starIDs = [-1]
        
        if gasindices[0] != -1:
            gasIDs = idGas[startGas:endGas][gasindices]
        else:
            gasIDs = [-1]
        
        dmIDs = []
        
        tempPosDM = pDM[dmindices]
        if len(dmindices) > 0 and dmindices[0] > 0:
            PrecenteredDM = dx_wrap(tempPosDM-cm,boxSize)
            PrecenteredDM = np.array([np.dot(pp, rotation.T) for pp in PrecenteredDM])
            DMinEll = (PrecenteredDM[:,0]**2./radii[0]**2. + PrecenteredDM[:,1]**2./radii[1]**2 + PrecenteredDM[:,2]**2./radii[2]**2)<=1.
            dmIDs = idDM[list(dmindices[DMinEll])]
        else:
            dmIDs = [-1]
           
        objectDict = {}
        objectDict['gasIDs'] = gasIDs
        objectDict['dmIDs'] = dmIDs
        objectDict['starIDs'] = starIDs
        objectDict['baryonFrac'] = gasFrac
        objectDict['isSIGO'] = isSIGO
        objectKey = (snapnum, idx)
        idDict[objectKey] = objectDict


print('Done')

with open(filename+"IDDictionary"+s_res+"_"+s_vel+"_fb0.6.dat",'wb') as f:
    pickle.dump(idDict, f)
