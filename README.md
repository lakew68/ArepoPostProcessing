# ArepoPostProcessing

Set of codes to postprocess Arepo simulations and Supersonically Induced Gas Objects (SIGOs) (see Naoz+14, Popa+16, Chiou+18, Chiou+19, Chiou+21, Lake+21, Lake+23a, Lake+23b, Williams+23)

- annikaEllipsoid.py contains routines for ellipsoid fitting of gas cells.
- DMVisualization.py generates gas density and DM density projections with the SIGO outlined, for presentation and for filtering.
- evolutiontracker.py produces a digraph representing the evolution of all objects in the simulation across snapshots.
- luminosityrhochrit.py contains routines for calculation of luminosity of SIGOs
- makeiddictionary.py produces a dictionary containing particle IDs for every object in the simulation across snapshots.
- makeSIGOidx.py finds all SIGOs from Gas-Primary (GP) objects
- match.py finds the closest GP to the Dark Matter Primary/Gas Secondary (DM/G) objects
- particleindicesgeneral.py finds all particles associated with DM/Gs.
- projection.py produces gas projection plots. (Mostly deprecated with the move we're making to Py3, use DMVisualization.py or yt)
- shrinker.py fits a tightly fitted ellipsoid to GPs.
- shrinker_with_mpi.py is an alternative version of shrinker that parallelizes the gas fitting. Use with mpi4py on a cluster.
- shrinker_with_copied_memory.py parallelizes finding star and dm particles in the ellipsoids, but uses a lot of memory: be careful with it!
- spinclass.py contains routines to calculate the spin parameter of DM/Gs.
- spinclassellipsoid.py contains routines to calculate the spin parameter of GPs.
- starcluster_shrinker.py fits a tightly fitted ellipsoid to the stellar component of GPs.

# Workflow

1.) Run particleindicesgeneral.py

2.) Run match.py

3.) Run shrinker.py, shrinker_with_copied_memory.py, or shrinker_with_mpi.py

4.) Run makeSIGOidx.py

5.) To visually confirm important SIGOs and remove NSCs, run DMVisualization.py.

6.) (Optional: run spinclass/spinclassellipsoid if working with spin parameter)

7.) (Optional: run luminosityrhochrit.py if working with semianalytic luminosity model)

8.) (Optional: run makeiddictionary.py if tracking the evolution of objects)

9.) (Optional: run evolutiontracker.py if tracking the evolution of objects)

10.) (Optional: run starcluster_shrinker.py for Kennicutt-Schmidt relation)

# TODO: 
- correct gasindices in shrinker.py


