---
title: 'hectorui: A web-based interactive scenario builder and visualization application for the Hector climate model'
tags:
  - Python
  - climate model
  - EBM
authors:
  - name: Benjamin Schmiedel
    orcid: 0000-0003-2411-3176
    affiliation: "1, 2"
affiliations:
 - name: Institute of environmental physics, University of Heidelberg, Germany
   index: 1
 - name: University of Gothenburg, Germany
   index: 2
date: 16 December 2020
bibliography: paper.bib
---

# Summary

Energy balance models (EBMs) provide one of the most fundamental approximations of the Earth's climate state, 
and despite their simplicity, they can give valuable information on the climate state and climate model behaviour. 
The development of EBMs goes back to the late 1960s, where [@Budyko1968] and [@Sellers1969] were able to reveal
fundamental processes of Earth's climate system. Sadly, these early descriptions of EBM are digitized in modified form only,
which is why the initial scope of `ZOEE` was their digitization, to ensure that their valuable work is not lost.
The parameterizations of these EBMs are valid to a limited extent only. Therefore, `ZOEE` includes updated versions 
as well as extensions of the models' physical description, e.g. including radiative forcing terms. Not only does the update
reduce the inaccuracies the models show when compared to modern climate data, but also allows the use of modern techniques
to simulate the evolution of the climate state.
The foundation is the use of a numerical scheme to solve the EBMs differential equation in the temporal domain and by that
explore its representation of earth's climate. The simple formulation of the model allows fast running simulations that can
be preformed on any computer. 

Beneath the interest to update the model descriptions and possibly provide teaching material, 
one of the mayor research purposes is the optimization of the parameterizations used.
An optimization algorithm is introduced based on the gradient desccent method. It gradually optimizes a set of parameters
such that the deviation of simulated climate indicators (e.g. Zonal mean temperature) to a given target is minimized. 
Figure 1 show an example where the optimization algorithm is applied to a one-dimensional energy balance model. The 
global mean temperature (GMT) and zonal mean temperature (ZMT) gradually approach the target data, 
which is a simulation run from the general circulation model HadCM3 and the ERA-20CM reanalysis product.

![Application example of the ZOEE optimization scheme where a simple one dimensional model is optimized to target data 
from a general circulation model](figures/figure1.png)

With this method included in `ZOEE`, it is pursued to investigate the limits of the simplified parameterizations of 
climate system processes as used in EBMs. 
It may provide new applications of EBMs and the EBM's point of view on the climate system that can be comprehended
through the few energetic terms may enable one to actually keep track of the evolution towards an optimal parameterization, 
which is largely inaccessible in complex climate models.

# Statement of need

The need of `ZOEE` is threefold. First of all, it provides a digitization of the early descriptions of EBMs by [@Budyko1968] 
and [@Sellers1969], which ensures that the knowledge gained from these studies is carried along. This aspect should be 
recognised especially in the teaching of climate physics, as the concept of EBMs is often one of the first to learn, 
and the increasingly use of online material neglects the material not digitally available. 

Further, `ZOEE` allows to easily modify the EBM configurations and provides already updated versions of the 
Budyko- and Sellers-EBMs. While complex general circulation models come with the large drawback of high computation times 
and inconclusive interpretations, simulations with the EBMs in `ZOEE` are fast-running and the findings can be 
attributed to specific terms of the EBMs formulation. 

At last, the implemented optimization scheme targets the scientific use of `ZOEE` and the presented EBMs.
It is included in `ZOEE`, but is decoupled from the EBMs in the package and should be applicable to any climate model, 
although it primarily targets low complexity models. The idealized parameterizations of EBMs are the mayor point of criticism, 
as many processes are approximated in single terms. However, the evolution of the 
global climate state is represented quite accurately in EBMs and the optimization scheme proved to modify the set of 
parameters in a way that the EBM becomes even more accurate (in terms of deviation from much more complex target data).
By exploring the optimal set of parameters of an EBM when optimized to various specific climate states opens the possibility
to provide a first order approximation of how the underlying processes might behave in different climate states. 

# Acknowledgments

I acknowledge the continuous support from Kira Rehfeld, University of Heidelberg, and many informative discussions with 
with members from the STACY-group. 
Further, the funding through the Emmy-Noether-Programm and the PalMod Programm is highly appreciated. At last, a big thanks
goes to Ingo Bethke, University Bergen, for the opportunity to cooperate through an internship during which mayor 
steps were made in the implementation. 
# References
