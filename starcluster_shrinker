# This file is similar to shrinker, but fits only to the stellar component of objects from shrinker. 
# Gives a tight ellipsoid containing all stars in the star cluster, as well as associated gas. Useful for Kennicut-Schmidt relation.

import snapHDF5
import numpy as np
from scipy.spatial.distance import cdist
from annikaEllipsoid import *
import readsubfHDF5
import pickle
import time
import progressbar

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

widgets = [
        "Progress :",
        ' ', progressbar.Percentage(),
        ' ', progressbar.GranularBar(),
        ' ', progressbar.AdaptiveETA(),
        ] # For progress bar


res = '14Mpc'
vel = 'Sig0'
s_res = res
s_vel = vel
filename = "Documents/SF_Sig0/"
COMsInOrderMultiZ = []
starLocationsInOrderMultiZ = []
starIDsInOrderMultiZ = []
mStar_ellipsoid_multiZ = []
radii_ellipsoid_multiZ = []
rotation_ellipsoid_multiZ = []
cm_ellipsoid_multiZ = []
snapkey = [] #Fill this in with the snap numbers to use
for snapnum2 in snapkey:
    snapnum = snapnum2
    snapfile = filename+"snap_"+str(snapnum).zfill(3)+".hdf5"
    with open(filename+'shrinker'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
        shrunken = pickle.load(f)
    filename2 = filename +  "GasOnly_FOF" #Used for readsubfHDF5
    filename3 = filename2 + "/snap-groupordered_" + str(snapnum).zfill(3) #Used for snapHDF5
    cat = readsubfHDF5.subfind_catalog(filename2, snapnum)
    header = snapHDF5.snapshot_header(filename3)
    red = header.redshift
    atime = header.time
    boxSize = header.boxsize
    print(boxSize)
    #COMsInOrder = []
    pGas= snapHDF5.read_block(filename3,"POS ", parttype=0)
    mGas= snapHDF5.read_block(filename3,"MASS", parttype=0)
    pStar = snapHDF5.read_block(filename3,"POS ", parttype=4)
    mStar = snapHDF5.read_block(filename3,"MASS", parttype=4)
    IDGas = snapHDF5.read_block(filename3,"ID  ", parttype=0)
    IDStar = snapHDF5.read_block(filename3,"ID  ", parttype=4)
    halo100_indices= np.where((cat.GroupLenType[:,0]+cat.GroupLenType[:,4]) >100)[0]		
    startAllGas = []
    endAllGas   = []
    for i in halo100_indices:
        startAllGas += [np.sum(cat.GroupLenType[:i,0])]
        endAllGas   += [startAllGas[-1] + cat.GroupLenType[i,0]]
    
    radii_ellipsoid = []
    rotation_ellipsoid = []
    cm_ellipsoid = []
    mGas_ellipsoid = []
    mStar_ellipsoid = []
    gasIDs = []
    starindices = []
    starIDs = []
    print(halo100_indices)
    with progressbar.ProgressBar(max_value=len(halo100_indices), widgets=widgets) as bar:
        progressCounter = -1
        
        for i in halo100_indices:		
            progressCounter += 1
            if progressCounter % 20 == 0:
                bar.update(i)
            cm = shrunken['cm'][i]
            rotation = shrunken['rotation'][i]
            radii = shrunken['radii'][i]
            mStarSIGO = shrunken['mStar'][i]
            starIDsSIGO = shrunken['starIDs'][i]
            gasIDsSIGO = shrunken['gasIDs'][i]
            if len(starIDsSIGO) > 1:
                pSIGO = np.isin(IDStar, list(starIDsSIGO))
                locStars = pStar[pSIGO]
                COM = np.mean(locStars,axis=0)
                initialStarLocations = np.array(locStars)
                starLocations = []
                tempStarIDs = []
                if len(initialStarLocations) > 0:
                    
                    Precentered = np.array(dx_wrap(locStars - COM,boxSize))
                    dists = np.sqrt(dist2(Precentered[:,0],Precentered[:,1],Precentered[:,2],boxSize))
                    maxAxis = np.max(np.abs(dists))
                    
                    
                    maxSeparation = maxAxis
                    
                    
                    print(maxSeparation)
                    
                    inEll = (Precentered[:,0]**2./maxSeparation**2. + Precentered[:,1]**2./maxSeparation**2 + Precentered[:,2]**2./maxSeparation**2)<=1.
                    starLocations.append(list(locStars[inEll]))
                    tempStarIDs.append(list((IDStar[pSIGO])[inEll]))
                    xCoords = []
                    yCoords = []
                    COM = np.mean(locStars[inEll],axis=0)

                    P = locStars[inEll]
                    M = (mStar[pSIGO])[inEll]
                    if len(P) > 0:
                        cm = np.average(P,axis=0)#,weights = R
                        Precentered = dx_wrap(P - cm,boxSize)
                        dists = np.sqrt(dist2(P[:,0]-cm[0],P[:,1]-cm[1],P[:,2]-cm[2],boxSize))
                        maxAxis = np.max(np.abs(dists))
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
                            if maxAxis == 0. or numGasOrig == 0.:
                                skipThis = True
                                    
                            if not skipThis:
                                radii = np.array([tempAxis*ratios[0], tempAxis*ratios[1], tempAxis])
                                rotation = np.array(evecs)
                                radii_ellipsoid += [list(radii)]	
                                rotation_ellipsoid += [[list(r) for r in rotation]]
                                cm_ellipsoid += [list(cm)]	


                                #Calculate Star mass
                                tempPosStar = dx_wrap(locStars-cm,boxSize)			
                                #Only look for stars within the sphere generated by the max ellipsoid axis
                                nearidx, = np.where(dist2(locStars[:,0]-cm[0],locStars[:,1]-cm[1],locStars[:,2]-cm[2],boxSize)<=tempAxis**2)
                                starindices += [nearidx] 
                                starIDs += [(IDStar[pSIGO])[nearidx]]
                                tempPosStar = tempPosStar[nearidx]
                                tempPosStar = np.array([np.dot(pp,evecs.T) for pp in tempPosStar])
                                stellarmass = 0
                                if len(tempPosStar) == 0:
                                    #SIGO has no Stars
                                    mStar_ellipsoid += [0.]
                                else:
                                    StarinEll = tempPosStar[:,0]**2/ratios[0]**2 + tempPosStar[:,1]**2/ratios[1]**2 + tempPosStar[:,2]**2 <= tempAxis**2 
                                    stellarmass = np.sum(mStar[pSIGO][nearidx][StarinEll])
                                    mStar_ellipsoid += [stellarmass]

                                #Calculate Gas mass
                                pGasSIGO = np.isin(IDGas, list(gasIDsSIGO))
                                locGas = pGas[pGasSIGO]
                                tempPosGas = dx_wrap(locGas-cm,boxSize)			
                                #Only look for gas within the sphere generated by the max ellipsoid axis
                                nearidx, = np.where(dist2(locGas[:,0]-cm[0],locGas[:,1]-cm[1],locGas[:,2]-cm[2],boxSize)<=tempAxis**2)
                                gasIDs += [IDGas[pGasSIGO][nearidx]]
                                tempPosGas = tempPosGas[nearidx]
                                tempPosGas = np.array([np.dot(pp,evecs.T) for pp in tempPosGas])
                                gasmass = 0
                                if len(tempPosGas) == 0:
                                    #SIGO has no Gas
                                    mGas_ellipsoid += [0.]
                                else:
                                    GasinEll = tempPosGas[:,0]**2/ratios[0]**2 + tempPosGas[:,1]**2/ratios[1]**2 + tempPosGas[:,2]**2 <= tempAxis**2 
                                    gasmass = np.sum(mGas[pGasSIGO][nearidx][GasinEll])
                                    mGas_ellipsoid += [gasmass]
                                    #mStar_ellipsoid += [1.*np.sum(StarinEll)*massStarParticle]		


                            else:
                                #overRadii += [-1.]
                                radii_ellipsoid += [[-1.,-1.,-1.]]
                                rotation_ellipsoid +=[[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]] 
                                cm_ellipsoid += [[-1.,-1.,-1.]]
                                mGas_ellipsoid += [-1.]
                                mStar_ellipsoid += [-1.]
                                gasIDs += [[-1]]
                                starindices += [[-1]]
                                starIDs += [[-1]]

                        else:
                            print(i, ' no stars in initial ellipsoid')
                            #overRadii += [-1.]
                            radii_ellipsoid += [[-1.,-1.,-1.]]
                            rotation_ellipsoid +=[[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]] 
                            cm_ellipsoid += [[-1.,-1.,-1.]]
                            mGas_ellipsoid += [-1.]
                            mStar_ellipsoid += [-1.]
                            gasIDs += [[-1]]
                            starindices += [[-1]]
                            starIDs += [[-1]]
            else:
                radii_ellipsoid += [[-1.,-1.,-1.]]
                rotation_ellipsoid +=[[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]] 
                cm_ellipsoid += [[-1.,-1.,-1.]]
                mGas_ellipsoid += [-1.]
                mStar_ellipsoid += [-1.]
                gasIDs += [[-1]]
                starindices += [[-1]]
                starIDs += [[-1]]
            
    
        
    shrunken = {} #Initialize dict of results
    shrunken['radii'] = np.array(radii_ellipsoid)   #each ellipsoid axis value from min to max
    shrunken['rotation'] = np.array(rotation_ellipsoid) #rotation matrix to get into principal frame
    shrunken['cm'] = np.array(cm_ellipsoid) #center of mass
    shrunken['mGas'] = np.array(mGas_ellipsoid) #gas mass in the ellipsoid
    shrunken['starindices'] = np.array(starindices) #star indices in the ellipsoid
    shrunken['gasIDs'] = np.array(gasIDs) #gas indices in the ellipsoid
    shrunken['starIDs'] = np.array(starIDs) #star indices in the ellipsoid
    shrunken['mStar'] = np.array(mStar_ellipsoid) #star mass in the ellipsoid
    print(radii_ellipsoid)
    print(mGas_ellipsoid)

    with open(filename+'star_only_shrinker'+res+'_'+vel+'_'+str(snapnum)+'.dat','wb') as f:
        pickle.dump(shrunken, f)

