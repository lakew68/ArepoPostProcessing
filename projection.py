#Plot projections for SIGOs
#kind is element of ['TEMP', 'MACH', 'VEL', 'CS', 'RHOC', 'RHO']:

import matplotlib
matplotlib.use('agg')
import pylab
from gadget import *

from gadget_subfind import *
import calcGrid
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from pylab import axes, colorbar, gca
import readsubfHDF5
try:
	import cPickle as pickle
except:
	import pickle


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

element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}

toinch = 0.393700787

res = '14Mpc'
vel = '118kms'
snapnum = 5

with open('../shrinker'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
	shrunken = pickle.load(f)
with open('../match'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
	matched = pickle.load(f)
with open('../SIGOidx'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
	SIGOidx = pickle.load(f)
with open('../luminosity/GP_luminosity'+res+'_'+vel+'_'+str(snapnum)+'.dat','rb') as f:
        gp = pickle.load(f)

prefix = '../../../'
run = "14Mpc_118kms_Cooling_OldArepo"
snap = 5 
name = 'clump'
cat_118kms = readsubfHDF5.subfind_catalog("../../../14Mpc_118kms_Cooling_OldArepo/output/GasOnly_FOF",snap)

jobdir = 'paperplots/' # 'Plots_projections/'


def h100toSIGOidx(idx):
    try:
        return np.where(SIGOidx==idx)[0][0]
    except:
        print "No SIGO found"
        raise KeyError 

def plotProjection(kind, index, boxsize=1.):
        #index is which SIGO it is, ihalo refers to the numbering in halo100_indices
        ihalo = SIGOidx[index] 
        print ihalo

	path = prefix+run+'/output/GasOnly_FOF/'
	s = gadget_readsnap( snap, snapbase='snap-groupordered_',snappath=path, hdf5=True, loadonly=['pos','vel', 'mass', 'vol', 'rho', 'u'], loadonlytype=[0], forcesingleprec=False )
	cms = cat_118kms.GroupPos
	cms = cms/(s.hubbleparam / s.time)
	cvel = cat_118kms.GroupVel
	cvel = cvel / s.time


	pxsize = 6. 
	pysize = 6.

	psize = 2.
	offsetx = 1.1
	offsety = .4
	offset = .33
		
	fig = pylab.figure( figsize=(np.array([pxsize,pysize])*toinch), dpi=300)
	res = 256
	#res = 128
	#boxsize = .5 # 10kpc
	fact = 0.5      # projection length will be 2.0 * fact * boxsize
	
	iplot = 0
	ix = iplot % 4
	x = ix * (2. * psize + offsetx) / pxsize + offsetx/pysize
		
	y = offsety/pysize
	y = (2.*offsety) / pysize
	ax1 = axes( [x,y,2.*psize/pxsize,2.*psize/pysize], frameon=True )
	
	y = (2.*psize + 3.*offset) / pysize + 0.15 * psize / pysize
	cax = axes( [x,y,2.*psize/pxsize,psize/pysize/15.], frameon=False )


	s.pos = s.pos - cms[ihalo]
	s.vel = s.vel - cvel[ihalo]
	
	#convert to kpc
	igas, = np.where( (np.abs(s.pos[:,0]) < boxsize) & (np.abs(s.pos[:,1]) < boxsize) & (np.abs(s.pos[:,2]) < boxsize) )
	npart = len(igas)
	print "Gas density plot selected particles in box", len(igas)

	# this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
	# multiplying by then and then in cm^-2 with all the other factors (this holds also for the
	# other projection functions). The factor boxsize / res is the dl of the projection


	
	#ne = s.data['ne'][:]
	#metallicity  = 0 
	#XH = s.data['gmet'][:, 0]
	#yhelium = (1 - XH - metallicity) / (4. * XH);
	#mu = (1 + 4 * yhelium) / (1 + yhelium + ne)

        mu = 1.22 #neutral primordial gas

	s.data['pos'] = s.pos[igas]
	s.data['type'] = s.type[igas]
	#s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10 * MSUN / KPC**2.0 / mu / PROTONMASS * boxsize / res #put in 1/cm^2
	s.data['rho'] =  s.rho[igas].astype('float64') * 1.0e10  *MSUN/ KPC**2.0 * boxsize / res #put in g/cm^2

	
	s.data['vol'] = s.vol[igas]
	s.data['mass'] = s.mass[igas]
	#convert internal energy to temperature 
	u = 1.0e10 * s.data['u'][igas] #it's a velocity squared to be converted in cgs

        #turn into temp
        temp = GAMMA_MINUS1 /  BOLTZMANN * u * PROTONMASS * mu	

        if kind == "TEMP": 
            s.data['u'] = temp

	#turn into sound speed
	cs = np.sqrt(GAMMA * BOLTZMANN * temp / mu / PROTONMASS)
        if kind == "CS":
            s.data['u'] = cs / 1.0e5 #convert to km/s
        
	s.data['vel'] = s.vel[igas] * 1.0e5 #convert to cgs

	#Turn first v_x into |v|
	velmag = np.linalg.norm(s.vel,axis=1)
        if kind == "VEL":
            s.data['u'] = velmag / 1.0e5 #convert to km/s

	mach = velmag / cs
        if kind == "MACH":
            s.data['u'] = mach

        if kind == "RHOC":
            rhocrit = gp['rhocritSIGO'][index]
            s.data['rho'] = s.rho / (rhocrit * boxsize / res * KPC)

        if kind == "RHO":
            s.data['rho'] = s.rho /  mu / PROTONMASS # in 1/cm^2

        if kind == "OVER":
            critical_density = 3.0 * .1 * .1 / 8.0 / np.pi / GCONST #.1 is to convert 100/Mpc to 1/kpc, this is in units of h^2
            Omega0 = s.omega0
            OmegaLambda = s.omegalambda
            atime = s.time
            critical_density *= Omega0 + atime**3 * OmegaLambda
            critical_density_gas = critical_density * baryonfraction
            critical_density_gas *= hubbleparam**2 / atime**3 * 1.0e10  *MSUN/ KPC**2.0 *  boxsize / res  # in units of g/cm^2

            s.data['rho'] = s.rho / critical_density_gas  - 1.


	axes( ax1 )
	dextoshow = 6
	numthreads = 4

	#Plot mass weighted slice
        #temperature
        if kind == "TEMP":
	    s.plot_Aweightedslice( "u", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, minimum=1.0, newfig=False, cmap='inferno',numthreads=8,vrange=[100,10000])
        #mach
        if kind == "MACH":
            s.plot_Aweightedslice( "u", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]),proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='inferno',numthreads=8,vrange=[0.0001,20])

        if kind == "CS":
            s.plot_Aweightedslice( "u", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, minimum=1.0, newfig=False, cmap='inferno',numthreads=8,vrange=[1,50])

        if kind == "VEL":
            s.plot_Aweightedslice( "u", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, minimum=1.0, newfig=False, cmap='inferno',numthreads=8,vrange=[1,50])

        if kind == "RHOC":
	    s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='Spectral',numthreads=8,vrange=[1,40])
            #s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='Spectral',numthreads=8,minimum=.7)


        if kind == "RHO":
	    s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='Spectral',numthreads=8,vrange=[1e17,1e19])
            
            #Make velocity field
            denom = np.sqrt( (s.data['vel'][:,:2]**2).sum(axis=1) )
            vnorm = s.data['vel'] / denom[:,None]
            vmax = np.max(denom)
            vmin = np.min(denom)
            vnorm *= (denom[:, None] - vmin) / (vmax - vmin) 
            
            ngbs = s.get_Aslice('rho', box=[boxsize,boxsize], axes=[0,1], center=np.array([0.,0.,0.]), proj=False, proj_fact=fact, res=res )["neighbours"]
            
            ax1.quiver( s.pos[ngbs[:,:],0], s.pos[ngbs[:,:],1], vnorm[ngbs[:,: ],0], vnorm[ngbs[:, :],1], scale_units='inches', scale=5., pivot='middle', width=0.002, edgecolors=(''), color='lightgray' ) 
            #Make inset of SIGO 
            axins = zoomed_inset_axes(ax1, 2, loc=1)
            axes(axins)
            s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='Spectral',numthreads=8,vrange=[1e17,1e19])
            x1, x2, y1, y2 = -0.2, 0.2, -0.2, 0.2 
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax1, axins, loc1 = 2, loc2=4, fc="none", ec="0.5")
            plt.tick_params(axis='both', which='both',bottom=False, left=False, top=False, labelbottom=False, labelleft=False)
            axes(ax1)
            
        if kind == "OVER":
            s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, minimum=1.0, newfig=False, cmap='Spectral',numthreads=8,vrange=[1e-1,1e2])
            #Make inset of SIGO 
            axins = zoomed_inset_axes(ax1, 2, loc=1)
            axes(axins)
            s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True,axes=[0,1], box=[boxsize,boxsize], center=np.array([0.,0.,0.]), proj_fact=fact,logplot=True, rasterized=True, newfig=False, cmap='Spectral',numthreads=8,vrange=[1e-1,1e2])
            x1, x2, y1, y2 = -0.2, 0.2, -0.2, 0.2 
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            mark_inset(ax1, axins, loc1 = 2, loc2=4, fc="none", ec="0.5")
            plt.tick_params(axis='both', which='both',bottom=False, left=False, top=False, labelbottom=False, labelleft=False)
            axes(ax1)
            


        #colorbar( cax=cax, orientation='horizontal')
	colorbar( cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext()) 

        if kind == "TEMP":
    	    cax.set_title( '$Temperature\\rm{\\,[K]}}$', size=8 )
        if kind == "MACH":
	    cax.set_title( '$Mach$', size=8 )
        if kind == "VEL":
            cax.set_title( '$Vel\\rm{\\,[km/s]}}$',size=8)
        if kind == "CS":
            cax.set_title( '$cs\\rm{\\,[km/s]}}$',size=8)
        if kind == "RHOC":
	    cax.set_title( '$\\rho/\\rho_{\\rm crit}$', size=8 )
        if kind == "RHO":
	    cax.set_title( '$N\\rm{\\,[cm^{-2}]}$', size=8 )
        if kind == "OVER":
            cax.set_title( '$\\delta$', size=8)
	for label in cax.xaxis.get_ticklabels(): label.set_fontsize(8);
	
	#******* ******* ******* ******* ******* ******* ******* ******* ******* ******* *******
	for label in cax.xaxis.get_ticklabels(): label.set_fontsize(6);
	for label in ax1.xaxis.get_ticklabels(): label.set_fontsize(6);
	for label in ax1.yaxis.get_ticklabels(): label.set_fontsize(6);

        majorLocator = MultipleLocator(1.0)
        minorLocator = MultipleLocator(0.5)
        ax1.xaxis.set_major_locator(majorLocator)
	ax1.yaxis.set_major_locator(majorLocator)
	ax1.xaxis.set_minor_locator(minorLocator)
	ax1.yaxis.set_minor_locator(minorLocator)
	ax1.set_xlabel( "$\\rm{x\\,[kpc]}$", size=7 )
	ax1.set_ylabel( "$\\rm{y\\,[kpc]}$", size=7 )
	ax1.xaxis.labelpad = -0.25
	ax1.yaxis.labelpad = -1

        if kind == "TEMP":
            fig.savefig( jobdir+'Temp_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )
        if kind == "MACH":
            fig.savefig( jobdir+'Mach_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )
        if kind == "VEL":
            fig.savefig( jobdir+'velnorm_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )
        if kind == "CS":
            fig.savefig( jobdir+'cs_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )

        if kind == "RHOC":
	    fig.savefig( jobdir+'Rhocrit_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )
        if kind == "RHO":
	    fig.savefig( jobdir+'Rho_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )
        if kind == "OVER":
            fig.savefig( jobdir+'Over_%s_%s_Cooling.pdf' % (name, str(ihalo)), transparent=True, dpi=300 )

      

        
#for i in ['TEMP', 'MACH', 'VEL', 'CS', 'RHOC', 'RHO']:
#for i in ['RHO','TEMP']:
for i in ['MACH']:
    #for j in gp['rhoGTrhocritANDSIGO']:
    for j in [330, 692, 711]:
        plotProjection(i,h100toSIGOidx(j), boxsize=3.)
        #plotProjection(i,j) 
    #plotProjection(i,18)
    #plotProjection(i,40)
#plotProjection('RHOC',40)
'''
plotProjection('RHO',18)
plotProjection('RHO',40)
plotProjection('RHO', h100toSIGOidx(711))
plotProjection('RHO', h100toSIGOidx(330))
'''
#print('This is the SIGOidx: ' + str(h100toSIGOidx(8)))
#plotProjection('RHO', h100toSIGOidx(8))

