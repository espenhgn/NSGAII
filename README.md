# NSGAII

Collection of scripts running a NSGA-II multi-objective evolutionary fitting
algorithm on a simplified HH model neuron, suitable for network use.

Contains code from Armin Bahl, see https://projects.g-node.org/emoo/

Files:
fit_HH_to_HH.py - main example file
fit_HH_to_HH.sh - example job submission script
eamoo/__init__.py - implementaition of NSGA-II as python class EAMoo
feature.py - defines class for extracting features of data
soma.hoc - NEURON morphology file
LFPyCellTemplate.hoc - cell template file