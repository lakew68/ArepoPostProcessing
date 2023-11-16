from __future__ import print_function, division
from sys import argv
import numpy as np
import readsubfHDF5
import snapHDF5 
try:
   import cPickle as pickle
except:
   import pickle
import networkx as nx
import matplotlib.pyplot as plt
import astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.units as u
def timeToZ(time):
    cosmo = FlatLambdaCDM(H0=71, Om0=0.27, Ob0=0.044)
    return z_at_value(cosmo.age,time*u.Myr)

def zToTime(z):
    return (cosmo.age(z)-cosmo.age(redshifts(135))).value*1000

def redshifts(snap):
    if snap < 151:
        return 30 - snap/10.
    else:
        return 165 - snap

plt.rcParams.update({'font.size': 36})
cosmo = FlatLambdaCDM(H0=71, Om0=0.27, Ob0=0.044)


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

print('Started')
DMGMGasSig0 = []
DMGMStarSig0 = []
snapkey = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,151,152,153]

#Should be run with a snap number input
res = '14Mpc'
s_res = res
vel = 'Sig0'
s_vel = vel

filename = "Documents/SF_Sig0/"
DMGidxs = [] #Contains all DMG indexes in snap order. 2-D array.
for snapnum in snapkey:
    with open(filename + 'DMGidx'+s_res+"_"+s_vel+"_"+str(snapnum)+".dat",'rb') as f:
        DMGidx = pickle.load(f)
        DMGidxs.append(DMGidx)

G = nx.read_gpickle(filename+'GasObjectHistoryGraph.dat')




snapnum = 0
fbaryonSIGO = 0.6

for snap in range(len(snapkey)):
    print(snap)
        
    snapnum = snapkey[snap]
    #File paths
    filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
    filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for snapHDF5

    #Read header information
    header = snapHDF5.snapshot_header(filename3)
    red = header.redshift
    atime = header.time
    boxSize = header.boxsize
    Omega0 = header.omega0
    OmegaLambda = header.omegaL
    massDMParticle = header.massarr[1] #all DM particles have same mass
       

    critical_density *= Omega0 + atime**3 * OmegaLambda
    critical_density_gas = critical_density * baryonfraction
    
    with open(filename + 'shrinker'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
        shrunken = pickle.load(f)
    DMGMGasAtSnap = 0
    DMGMStarAtSnap = 0
    for gasObject in G.nodes():
        snap, objectID = gasObject
        if snap == snapnum:
            gas = set(G.nodes()[gasObject]['gasIDs']) #- allStarIDs
            
            if objectID in DMGidxs[snapkey.index(snapnum)]:
                mDM = max(shrunken['mDM'][objectID] * 1e10 / 0.71,0)
                mGas = max(shrunken['mGas'][objectID] * 1e10 / 0.71,0)
                mStar = max(shrunken['mStar'][objectID] * 1e10 / 0.71,0)
                DMGMGasAtSnap += shrunken['mGas'][objectID] * 1e10/.71
                DMGMStarAtSnap += shrunken['mStar'][objectID] * 1e10/.71

    
    
    
    DMGMGasSig0.append(DMGMGasAtSnap * 1e10/.71)
    DMGMStarSig0.append(DMGMStarAtSnap * 1e10/.71)


print(DMGMGasSig0)
print(DMGMStarSig0)



filteredSIGOsWithStars = [[[83, 834], [84, 837], [85, 845], [86, 858], [87, 860], [88, 886], [89, 880], [90, 880], [91, 882], [92, 884], [93, 889], [94, 886], [95, 899], [96, 923], [97, 948], [98, 932], [99, 936], [100, 958], [101, 959], [102, 633], [102, 995], [103, 666], [103, 1009], [104, 683], [104, 1022], [105, 702], [105, 1010], [106, 730], [106, 1038], [107, 718], [107, 1034], [108, 725], [108, 1043], [109, 738], [109, 1077], [110, 735], [110, 1043], [111, 766], [111, 1090], [112, 749], [112, 1119], [113, 755], [113, 1142], [114, 763], [114, 1184], [115, 767], [115, 1189], [116, 800], [116, 1208], [117, 802], [117, 1245], [118, 812], [118, 1280], [119, 840], [119, 1323], [120, 825], [120, 1354], [121, 853], [121, 1422], [122, 858], [122, 1479], [123, 886], [123, 1477], [124, 902], [124, 1496], [125, 920], [125, 1549]], [[102, 576], [104, 624], [105, 679], [106, 647], [107, 686], [108, 694], [109, 737], [110, 785], [111, 774], [112, 769], [113, 789], [114, 835], [115, 872], [116, 891], [117, 900], [118, 912], [119, 930], [120, 935], [121, 969], [122, 987], [123, 997], [124, 1040], [125, 1059], [126, 1066], [127, 1082], [128, 1098], [129, 1106], [130, 1120], [131, 1126], [132, 1125], [133, 1141], [134, 1150], [135, 1158], [136, 1178], [137, 1180], [138, 1221], [139, 1229], [140, 1282], [141, 1302], [142, 1345], [143, 1422], [144, 1453], [145, 1317], [146, 1296], [147, 1286]], [[130, 1910], [130, 2770], [131, 1899], [131, 2757], [132, 1937], [132, 2793], [133, 1930], [133, 2758], [134, 1980], [134, 2790], [135, 1978], [135, 2806], [136, 1982], [136, 2840], [137, 1982], [137, 2814], [138, 2025], [138, 2838], [139, 2065], [139, 2822], [140, 2022], [140, 2814], [141, 2027], [141, 2806], [142, 2044], [142, 2744], [143, 2032], [143, 2663], [144, 2013], [144, 2684], [145, 2026], [145, 2628], [146, 1985], [146, 2606], [147, 2023], [147, 2571], [148, 1930], [148, 2457], [149, 1766], [149, 2244], [149, 7275], [150, 944], [151, 874], [152, 895]], [[132, 1195], [133, 1196], [134, 1203], [135, 1224], [136, 1217], [137, 1215], [138, 1207], [139, 1216], [140, 1248], [141, 1284], [142, 1314], [143, 1330], [144, 1371], [145, 1335], [146, 1326], [147, 1352], [148, 1353], [149, 1346], [150, 1347], [151, 1391], [153, 3108]], [[96, 1146], [99, 1193], [100, 1163], [101, 1177], [102, 1205], [103, 1235], [104, 1293], [105, 1366], [106, 1411], [107, 1465], [108, 1557], [109, 1661], [110, 1690], [111, 1781], [116, 2205], [117, 2247], [118, 2246], [119, 2356], [120, 2375], [121, 2414], [122, 2507], [123, 2481], [124, 2495], [125, 2548], [126, 2605], [127, 2647], [128, 2728], [129, 2754], [130, 2747], [131, 2677], [132, 2657], [133, 2514], [134, 2636], [135, 2708], [136, 2682], [137, 2699], [139, 2800], [140, 2853], [141, 2811], [142, 2862], [143, 2781], [144, 2854], [146, 3934], [147, 3959], [148, 3994], [149, 4010], [150, 4147], [151, 4931]], [[152, 2536], [153, 2244]], [[152, 5968]], [[153, 5848]], [[135, 1671], [136, 1753], [137, 1795], [138, 1867], [139, 1883], [140, 1956], [141, 1959], [142, 2020], [143, 2125], [144, 2087], [145, 2216], [146, 2205], [147, 2253], [148, 2293], [149, 2285], [150, 2356], [151, 2689], [152, 2724]]]
SIGOsWithStars = set([])
for row in filteredSIGOsWithStars:
    for SIGO in row:
        SIGOsWithStars.add((SIGO[0],SIGO[1]))

SIGOMGas = []
DMGMGas = []
SIGOMStar = []
DMGMStar = []
snapkey = range(154)
massBins = []

#Should be run with a snap number input
res = '14Mpc'
s_res = res
vel = 'Sig2'
s_vel = vel

filename = "D:/Star_Movie_768/"
DMGidxs = [] #Contains all DMG indexes in snap order. 2-D array.
for snapnum in snapkey:
    with open(filename + 'DMGidx'+s_res+"_"+s_vel+"_"+str(snapnum)+".dat",'rb') as f:
        DMGidx = pickle.load(f)
        DMGidxs.append(DMGidx)

G = nx.read_gpickle(filename+'GasObjectHistoryGraph_newest3.dat')




snapnum = 0
fbaryonSIGO = 0.6
for snapidx in range(len(snapkey)):
    print(snapidx)
        
    snapnum = snapkey[snapidx]
    #File paths
    filename = 'D:/Star_Movie_768/'
    filename2 = filename +  "DM_FOF" #Used for readsubfHDF5
    filename3 = filename + "snap_" + str(snapnum).zfill(3) #Used for snapHDF5

    #Read header information
    header = snapHDF5.snapshot_header(filename3)
    red = header.redshift
    atime = header.time
    boxSize = header.boxsize
    Omega0 = header.omega0
    OmegaLambda = header.omegaL
    massDMParticle = header.massarr[1] #all DM particles have same mass
       

    critical_density *= Omega0 + atime**3 * OmegaLambda
    critical_density_gas = critical_density * baryonfraction
    
    with open(filename + 'shrinker'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
        shrunken = pickle.load(f)
    
    SIGOMGasAtSnap = 0
    DMGMGasAtSnap = 0
    SIGOMStarAtSnap = 0
    DMGMStarAtSnap = 0
    for gasObject in G.nodes():
        snap, objectID = gasObject
        if snap == snapnum:
            gas = set(G.nodes()[gasObject]['gasIDs']) #- allStarIDs
            if G.nodes()[gasObject]['isSIGO'] and gasObject in SIGOsWithStars:
                mDM = max(shrunken['mDM'][objectID] * 1e10 / 0.71,0)
                mGas = max(shrunken['mGas'][objectID] * 1e10 / 0.71,0)
                mStar = max(shrunken['mStar'][objectID] * 1e10 / 0.71,0)
                SIGOMGasAtSnap += shrunken['mGas'][objectID] * 1e10/.71
                SIGOMStarAtSnap += shrunken['mStar'][objectID] * 1e10/.71
                
            if objectID in DMGidxs[snapidx]:
                mDM = max(shrunken['mDM'][objectID] * 1e10 / 0.71,0)
                mGas = max(shrunken['mGas'][objectID] * 1e10 / 0.71,0)
                mStar = max(shrunken['mStar'][objectID] * 1e10 / 0.71,0)
                
                DMGMGasAtSnap += shrunken['mGas'][objectID] * 1e10/.71
                DMGMStarAtSnap += shrunken['mStar'][objectID] * 1e10/.71
    
    
    SIGOMGas.append(SIGOMGasAtSnap * 1e10/.71)
    DMGMGas.append(DMGMGasAtSnap * 1e10/.71)
    SIGOMStar.append(SIGOMStarAtSnap * 1e10/.71)
    DMGMStar.append(DMGMStarAtSnap * 1e10/.71)

print(DMGMGas)

print(DMGMStar)



zList = [(redshifts(snap+1)) for snap in range(153)]
zList.insert(0, timeToZ(cosmo.age(30).value * 1000))
zListSig0 = [(redshifts(snap)) for snap in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,151,152,153]]
zListSig0.insert(0, timeToZ(cosmo.age(30).value * 1000))
plt.plot(zListSig0,np.array(DMGMStarSig0) / (np.array(DMGMGasSig0)+np.array(DMGMStarSig0)),linewidth=5,c='b')
plt.plot(zList,np.array(DMGMStar) / (np.array(DMGMGas)+np.array(DMGMStar)),linewidth=5,c='orange')
plt.plot(zList,np.array(SIGOMStar) / (np.array(SIGOMGas)+np.array(SIGOMStar)),linewidth=5,c='r')

plt.yscale('log')
plt.xlabel('z')
plt.ylabel('Star Formation Efficiency')
plt.text(28,.125,'DM GHOSts',color='orange')
plt.text(28,.4,'Classical Halos',color='b')
plt.text(22.5,.1,'SIGOs',color='r')
plt.xlim(12,30)
plt.ylim(3e-2,1)
plt.gca().invert_xaxis()
plt.gcf().set_size_inches([16,12])
plt.gcf().savefig('newSFE.pdf',bbox_inches='tight')





