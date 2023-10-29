"""
Program to fit ellipsoid to FOF object, then shrinks the ellipsoid, in parallel. Use on cluster like Hoffman.
Returns: pickled file with dictionary containing ellipsoid information
Run with a command like `mpirun -np 16 python shrinker_with_mpi.py' where the number after np is the number of threads it will use.

10/26/23
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
import time
from mpi4py import MPI

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

# For mpi

comm = MPI.COMM_WORLD
numProcesses = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    print(numProcesses,rank)

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
snapkey = [6]
for snapnum2 in snapkey:
    if rank == 0: # The master is the only process that reads the file
        #Should be run with a snap number input
        startTime = time.clock()
    else:
        startTime = 0
    res = '14Mpc'
    vel = 'Sig2'
    snapnum = int(snapnum2)
    s_vel = vel.replace(".","")
    s_res = res.replace(".","")

    #File paths
    filename = "/u/project/snaoz/wlake/Output_Sig2_Molecular/"
    filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
    filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for snapHDF5
    print(filename3)

    #Read header information
    header = snapHDF5.snapshot_header(filename3)
    red = header.redshift
    atime = header.time
    boxSize = header.boxsize
    Omega0 = header.omega0
    OmegaLambda = header.omegaL
    massDMParticle = header.massarr[1] #all DM particles have same mass
    if rank == 0:
        print("DM mass: ", massDMParticle)
        print(header.massarr)

    #redshift evolution of critical_density
    critical_density *= Omega0 + atime**3 * OmegaLambda
    critical_density_gas = critical_density * baryonfraction

    if rank == 0:
        #load particle indices and catalogs
        pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
        mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
        pStar = snapHDF5.read_block(filename3,"POS ", parttype=4)
        mStar = snapHDF5.read_block(filename3,"MASS", parttype=4)
        pDM = snapHDF5.read_block(filename3,"POS ", parttype=1)
        IDGas = snapHDF5.read_block(filename3,"ID  ", parttype=0)
        IDDM = snapHDF5.read_block(filename3,"ID  ", parttype=1)
        IDStar = snapHDF5.read_block(filename3,"ID  ", parttype=4)
        cat = readsubfHDF5.subfind_catalog(filename2, snapnum)

        halo100_indices= np.where((cat.GroupLenType[:,0]+cat.GroupLenType[:,4]) >100)[0]		
        startAllGas = []
        endAllGas   = []
        for i in halo100_indices:
            startAllGas += [np.sum(cat.GroupLenType[:i,0])]
            endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]

        haloPos = cat.GroupPos[halo100_indices]


        overRadii = [-1] * len(halo100_indices)
        radii_ellipsoid = [-1]* len(halo100_indices)
        rotation_ellipsoid = [-1]* len(halo100_indices)
        cm_ellipsoid = [-1]* len(halo100_indices)
        mDM_ellipsoid = [-1]* len(halo100_indices)
        mGas_ellipsoid = [-1]* len(halo100_indices)
        mStar_ellipsoid = [-1]* len(halo100_indices)
        gasFrac = [-1]* len(halo100_indices)
        DMindices = [-1]* len(halo100_indices)
        gasindices = [-1]* len(halo100_indices)
        starindices = [-1]* len(halo100_indices)
        DMIDs = [-1]* len(halo100_indices)
        gasIDs = [-1]* len(halo100_indices)
        starIDs = [-1]* len(halo100_indices)
    else:
        halo100_indices = []
        
    halo100_indices = comm.bcast(halo100_indices, root=0)
    loopLen = int(len(halo100_indices)/numProcesses + .99999)
    if rank == 0:
        print('Number of data loops: ', loopLen)
     
    for parallelLoopIdx in range(loopLen):
        if rank == 0:
            dataArray = [0] * numProcesses
            for loopidx in range(numProcesses):
		if parallelLoopIdx*numProcesses+loopidx >= len(halo100_indices):
		    continue
                idx = halo100_indices[parallelLoopIdx*numProcesses+loopidx]
                cm = haloPos[idx] 
                startGas = startAllGas[idx]
                endGas = endAllGas[idx]
                if endGas > startGas:
                    if np.sum(cm == np.array([0., 0., 0.]))==3:
                        # it's probbaly an error; recompute com
                        totalGas = np.sum(mGas[startGas: endGas])
                        cm = np.array([np.sum(pGas[startGas: endGas, i]*mGas[startGas: endGas])/totalGas for i in range(3)])

                    #Reposition particles to take into account CM
                    P = pGas[startGas:endGas]
                    I = IDGas[startGas:endGas]
                    M = mGas[startGas:endGas]
                else:
                    print(idx, ' was only stars')
                    radii_ellipsoid[np.where(halo100_indices==idx)[0][0]] = [-1.,-1.,-1.]
                    rotation_ellipsoid[np.where(halo100_indices==idx)[0][0]] = [[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]] 
                    cm_ellipsoid[np.where(halo100_indices==idx)[0][0]] = [-1.,-1.,-1.]
                    DMindices[np.where(halo100_indices==idx)[0][0]] = [-1]
                    gasindices[np.where(halo100_indices==idx)[0][0]] = [-1]
                    starindices[np.where(halo100_indices==idx)[0][0]] = [-1]
                    DMIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                    gasIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                    starIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                    continue
                dataArray[loopidx] = (idx,cm,startGas,endGas,P,I,M)
        else:
            dataArray = [0] * numProcesses
        dataArray = comm.scatter(dataArray,root=0)
        if type(dataArray) == int:
            outputTuple = 0
        else:
            idx,cm,startGas,endGas,P,I,M = dataArray
            Precentered = dx_wrap(P - cm,boxSize)
            dists = np.sqrt(dist2(P[:,0]-cm[0],P[:,1]-cm[1],P[:,2]-cm[2],boxSize))
            maxAxis = np.max(dists)
            ratios, evecs = axis(Precentered,maxAxis,axes_out=True,quiet=True)

            #Shrink ellipsoid by increments of .5% of the maximum axis until density ratio of
            #lengths of axes of shrunken ellipse to the original ellipse is greater than the ratio of
            #the number of gas cells enclosed in the shrunken ellipose to the original ellipse	
            #or until 20% of the total number of gas cells are removed
            if ratios[0] > 0. and ratios[1] > 0.: #accounts for erros in fit
                evecs = np.array(evecs)
                Precentered = np.array([np.dot(pp,evecs.T) for pp in Precentered])
                tempAxis = maxAxis
                inEll = Precentered[:,0]**2/ratios[0]**2+Precentered[:,1]**2/ratios[1]**2+Precentered[:,2]**2 <= maxAxis**2
                numGasOrig = np.sum(inEll)	
                skipThis=False
                while True:		
                    inEll = Precentered[:,0]**2/ratios[0]**2+Precentered[:,1]**2/ratios[1]**2+Precentered[:,2]**2 <= tempAxis**2
                    numGasTemp = np.sum(inEll) 
                    if maxAxis == 0. or numGasOrig == 0.:
                        skipThis = True
                        break
                    if tempAxis/maxAxis > 1.*numGasTemp/numGasOrig or 1.*numGasTemp/numGasOrig <= .8:
                        break
                    tempAxis-= .005*maxAxis

                if not skipThis:
                    over =  np.sum(M[inEll])/(4.*np.pi/3.*ratios[0]*ratios[1]*tempAxis**3.)/critical_density_gas
                    gasMass = np.sum(M[inEll])
                    overRadius = over
                    radii = np.array([tempAxis*ratios[0], tempAxis*ratios[1], tempAxis])
                    rotation = np.array(evecs)
                    radius_ellipsoid = list(radii)	
                    rotation = [list(r) for r in rotation]
                    cm = list(cm)
                    gasindex = list(inEll)
                    gasID = I[inEll]
                else:
                    overRadius = -1.
                    radius_ellipsoid = [-1.,-1.,-1.]
                    rotation = [[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]
                    cm = [-1.,-1.,-1.]
                    gasMass = -1.
                    gasindex = [-1]
                    gasID = [-1]
                    tempAxis = -1
                    ratios = [-1,-1,-1]
                    evecs = [-1,-1,-1]
            else:
                print(idx, ' had no gas')
                overRadius = -1.
                radius_ellipsoid = [-1.,-1.,-1.]
                rotation = [[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]
                cm = [-1.,-1.,-1.]
                gasMass = -1.
                gasindex = [-1]
                gasID = [-1]
                tempAxis = -1
                ratios = [-1,-1,-1]
                evecs = [-1,-1,-1]
            outputTuple = (idx,overRadius,radius_ellipsoid,rotation,cm,gasMass,gasindex,gasID,tempAxis,ratios,evecs)
        
        outputData = comm.gather(outputTuple,root=0)
                
        if rank == 0:
            for i in range(len(outputData)):
                if type(outputData[i]) == type(0):
                    continue
                idx,overRadius,radius_ellipsoid,rotation,cm,gasMass,gasindex,gasID,tempAxis,ratios,evecs = outputData[i]
		print(idx)
                radii_ellipsoid[np.where(halo100_indices==idx)[0][0]] = radius_ellipsoid
                rotation_ellipsoid[np.where(halo100_indices==idx)[0][0]] = rotation
                cm_ellipsoid[np.where(halo100_indices==idx)[0][0]] = cm
                gasindices[np.where(halo100_indices==idx)[0][0]] = gasindex
                gasIDs[np.where(halo100_indices==idx)[0][0]] = gasID
                
                if cm[0] > -1:
                    #Calculate Star mass
                    if type(pStar) != type(0):
                        tempPosStar = dx_wrap(pStar-cm,boxSize)			
                        #Only look for stars within the sphere generated by the max ellipsoid axis
                        nearidx, = np.where(dist2(pStar[:,0]-cm[0],pStar[:,1]-cm[1],pStar[:,2]-cm[2],boxSize)<=tempAxis**2)
                        starindex = nearidx

                        tempPosStar = tempPosStar[nearidx]
                        tempPosStar = np.array([np.dot(pp,evecs.T) for pp in tempPosStar])
                        stellarmass = 0
                        if len(tempPosStar) == 0:
                            #SIGO has no Stars
                            mStar_ellipsoid[np.where(halo100_indices==idx)[0][0]] = [0.]
                            starIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                        else:
                            StarinEll = tempPosStar[:,0]**2/ratios[0]**2 + tempPosStar[:,1]**2/ratios[1]**2 + tempPosStar[:,2]**2 <= tempAxis**2 
                            stellarmass = np.sum(mStar[nearidx][StarinEll])
                            starIDs[np.where(halo100_indices==idx)[0][0]] = IDStar[nearidx][StarinEll]
                            mStar_ellipsoid [np.where(halo100_indices==idx)[0][0]] = stellarmass
                            
                    else:
                        #Simulation has no Stars
                        stellarmass = 0
                        mStar_ellipsoid[np.where(halo100_indices==idx)[0][0]] = 0.
                        starIDs[np.where(halo100_indices==idx)[0][0]] = [-1]

                    #Calculate DM mass
                    tempPosDM = dx_wrap(pDM-cm,boxSize)			
                    #Only look for dark matter within the sphere generated by the max ellipsoid axis
                    nearidx, = np.where(dist2(pDM[:,0]-cm[0],pDM[:,1]-cm[1],pDM[:,2]-cm[2],boxSize)<=tempAxis**2)
                    DMindices += [nearidx] 
                    tempPosDM = tempPosDM[nearidx]
                    tempPosDM = np.array([np.dot(pp,evecs.T) for pp in tempPosDM])
                    if len(tempPosDM) == 0:
                        #Edge case: SIGO has no DM
                        mDM_ellipsoid[np.where(halo100_indices==idx)[0][0]] =0.
                        print("Index", idx)
                        DMIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                        gasFrac[np.where(halo100_indices==idx)[0][0]] = 1.
                    else:
                        DMinEll = tempPosDM[:,0]**2/ratios[0]**2 + tempPosDM[:,1]**2/ratios[1]**2 + tempPosDM[:,2]**2 <= tempAxis**2 
                        mDM_ellipsoid[np.where(halo100_indices==idx)[0][0]] = 1.*np.sum(DMinEll)*massDMParticle
                        DMIDs[np.where(halo100_indices==idx)[0][0]] = IDDM[nearidx][DMinEll]
                        gasFrac[np.where(halo100_indices==idx)[0][0]] = (gasMass+1.*stellarmass)/(gasMass + stellarmass + 1.*np.sum(DMinEll)*massDMParticle)
                else:
                    # Couldn't locate center of mass
                    mStar_ellipsoid[np.where(halo100_indices==idx)[0][0]] = 0.
                    starIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                    DMIDs[np.where(halo100_indices==idx)[0][0]] = [-1]
                    gasFrac[np.where(halo100_indices==idx)[0][0]] = 1.
    if rank == 0:
        #Print information
        shrunken = {} #Initialize dict of results
        shrunken['radii'] = np.array(radii_ellipsoid)   #each ellipsoid axis value from min to max
        shrunken['rotation'] = np.array(rotation_ellipsoid) #rotation matrix to get into principal frame
        shrunken['cm'] = np.array(cm_ellipsoid) #center of mass
        shrunken['overRadii'] = np.array(overRadii) #density / rhocrit
        shrunken['mDM'] = np.array(mDM_ellipsoid) #DM mass in the ellipsoid
        shrunken['mGas'] = np.array(mGas_ellipsoid) #gas mass in the ellipsoid
        shrunken['gasFrac'] = np.array(gasFrac) #gas fraction of the ellipsoid
        shrunken['DMindices'] = np.array(DMindices) #DM indices in the ellipsoid
        shrunken['gasindices'] = np.array(gasindices) #gas indices in the ellipsoid
        shrunken['starindices'] = np.array(starindices) #star indices in the ellipsoid
        shrunken['DMIDs'] = np.array(DMIDs) #DM indices in the ellipsoid
        shrunken['gasIDs'] = np.array(gasIDs) #gas indices in the ellipsoid
        shrunken['starIDs'] = np.array(starIDs) #star indices in the ellipsoid
        shrunken['mStar'] = np.array(mStar_ellipsoid) #star mass in the ellipsoid

        with open(filename+'shrinker_mpiTest'+s_res+'_'+s_vel+'_'+str(snapnum)+'_'+str(numProcesses)+'.dat','wb') as f:
            pickle.dump(shrunken, f)

        print('Time for snap ', str(snapnum),': ', str((time.clock()-startTime)/60./60.), ' hours.')
if rank == 0:
    with open("parallelTime_"+str(numProcesses)+".txt", "w") as text_file:
        text_file.write("Processing time: %s hours" % str((time.clock()-startTime)/60./60.))

    print('Done')       
                
