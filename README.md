# ArepoPostProcessing

Set of codes to postprocess Arepo simulations and Supersonically Induced Gas Objects (SIGOs) (see Naoz+14, Popa+16, Chiou+18, Chiou+19)

- annikaEllipsoid.py contains routines for ellipsoid fitting of gas cells.
- luminosityrhochrit.py contains routines for calculation of luminosity of SIGOs
- makeSIGOidx.py finds all SIGOs from Gas-Primary (GP) objects
- match.py finds the closest GP to the Dark Matter Primary/Gas Secondary (DM/G) objects
- particleindicesgeneral.py finds all particles associated with DM/Gs.
- shrinker.py fits a tightly fitted ellipsoid to GPs.
- spinclass.py contains routines to calculate the spin parameter of DM/Gs.
- spinclassellipsoid.py contains routines to calculate the spin parameter of GPs.


# TODO: 
- correct gasindices in shrinker.py

- update spinclassellipsoid.py

- make object-oriented. class for group, subclass DM/G, GP, subclass SIGO
