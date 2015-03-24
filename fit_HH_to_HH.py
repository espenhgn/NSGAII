#!/usr/bin/env python
'''
test script fitting a HH model to model HH data using EAMoo class

Usage:
python fit_HH_to_HH.py

or:
mpirun -np 8 python fit_HH_to_HH.py

'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from time import time
import LFPy
from feature import GetFeatures
from eamoo import EAMoo
from mpi4py import MPI


#set up some MPI stuff
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


#####################
# FUNCTIONS         #
#####################

def cellsim_w_HH(stim_I=0.2,
                 gnabar_hh=0.12,
                 gkbar_hh=0.036,
                 gl_hh=0.0003,
                 el_hh=-54.3):
    #LFPy.TemplateCell params
    cell_params = dict(
        morphology = 'soma.hoc',
        templatefile = 'LFPyCellTemplate.hoc',
        templatename = 'LFPyCellTemplate',
        passive = False,
        nsegs_method = None,
        tstartms = 0,
        tstopms = 1000,
        extracellular = False,
    )
    
    #LFPy.StimIntElectrode params
    stim_params = {
        'idx' : 0,
        'pptype' : 'IClamp',
        'delay' : 100.,
        'dur' : 800.,
        'amp' : stim_I,
    }
    
    
    #Why templatecell - Hay2011 is one, see LFPy examples using it.
    cell = LFPy.TemplateCell(**cell_params)
    
    #throw on some HH channels
    for sec in cell.allseclist:
        sec.insert('hh')
        #These are important, basically the DOFs were attempting to fit
        for seg in sec:
            seg.hh.gnabar = gnabar_hh
            seg.hh.gkbar = gkbar_hh
            seg.hh.gl = gl_hh
            seg.hh.el = el_hh
    
    #set stimulus
    stim = LFPy.StimIntElectrode(cell, **stim_params)
    
    #run the sim
    cell.simulate()
    
    #return data
    return cell.tvec, cell.somav


def get_dataset(**kwargs):
    '''return some model data'''
    somavs = []
    for amp in stim_amp:
        tvec, somav = cellsim_w_HH(stim_I=amp, **kwargs)
        somavs.append(somav)
    
    return tvec, np.array(somavs)   


def func_to_optimize(parameters):
    '''function to be minimized'''
    #get the data
    tvec, somavs = get_dataset(**parameters)

    allFeatures = GetFeatures(tvec, somavs)
    
    #container for output
    stuff = []

    #compute errors in each featurespace
    diff_feat0 = (features.feature0(**feature_params_0) -
                  allFeatures.feature0(**feature_params_0))
    diff_feat0 *= diff_feat0
    error_feat0 = np.sqrt(diff_feat0).sum()
    
    diff_feat1 = (features.feature1(**feature_params_1) -
                  allFeatures.feature1(**feature_params_1))
    diff_feat1 *= diff_feat1
    error_feat1 = np.sqrt(diff_feat1).sum()

    diff_feat5 = (features.feature5(**feature_params_5) -
                  allFeatures.feature5(**feature_params_5))
    diff_feat5 *= diff_feat5
    error_feat5 = np.sqrt(diff_feat5).sum()
    
    #prepare output, which is just a list of numbers for each feature.
    stuff.append(error_feat0)
    stuff.append(error_feat1)
    stuff.append(error_feat5)
    
    return stuff


def checkfullpopulation(pop, gen):
    '''Give access to all members in population, save outcome of each at
    every generation to file'''
    print "this is generation %i talking" % gen
    unnormed_population = pop.getpopulation_unnormed()
        
    np.save(os.path.join(filedest, 'output_gen%.3i.npy' % gen),
                         unnormed_population)


################
# MAIN
################

plt.close('all')

#set random seed
np.random.seed(12345)


#set up file dest
filedest = 'savefolder'
if RANK == 0:
    if os.path.isdir(filedest):
        for f in glob.glob(os.path.join(filedest, '*')):
            os.system('rm %s' % f)
    else:
        os.mkdir(filedest)

COMM.Barrier()

#stimulus current amplitudes
stim_amp = [0.2, -0.05]


#Get the features from "data" we'll fit against
tvec, somavs = get_dataset()


#object returning features of soma traces
features = GetFeatures(tvec, somavs)


#define some parameters for feature extraction
feature_params_0 = dict(
    xedges = np.arange(-100, 55, 5),
    yedges = np.arange(-10, 25, 1),
    threshold=1,
    smooth=True)

feature_params_1 = dict(
    rows=[-1],
    inds=np.r_[range(800, 1600), range(7200, 8000)]    
)

feature_params_5 = dict()


#Set range variables of all degrees of freedom, must be used by
#func_to_optimize() or functions used by it.
variables = [   ['gnabar_hh', 0.,  0.2],
                ['gkbar_hh',  0.,  0.2],
                ['gl_hh',     0.,  0.002],
                ['el_hh',   -70.,-40.]]

#timer
tic = time()


#sync MPI ranks
COMM.Barrier()


#Initialize EAMOO class, set size of population, capacity, variables,
#number of features.
#if running with MPI, set size to COMM.Get_Size() * factor - 1, and capacity
#to twice this value
my_eamoo = EAMoo(size=31,
                 capacity=62,
                 variables=variables,
                 obj=3,
                 infos=0,)
#these variables can probably change the rate of convergence
my_eamoo.setup(eta_m_0 = 2, eta_c_0 = 10, p_m = 0.5, finishgen = 10, d_eta_m=10)

#point the fitter to some functions
my_eamoo.get_objectives_error = func_to_optimize
my_eamoo.checkfullpopulation = checkfullpopulation


if RANK == 0:
    print "Starting Optimization ..."


#Set the number of evolutions, and start the fitting procedure
my_eamoo.evolution(30)


#sync MPI ranks
COMM.Barrier()


#print some stats
toc = time()
if RANK == 0:
    print 'runtime %.3f' % (toc - tic)
    

#####################
# PLOTTING          #
#####################

if RANK == 0:
    
    #plot the data fitted against
    fig = features.plot_traces()
    fig.savefig(os.path.join(filedest, 'data.pdf'))

    fig = features.plot_feature0(**feature_params_0)
    fig.savefig(os.path.join(filedest, 'data_feature0.pdf'))

    fig = features.plot_feature1(**feature_params_1)
    fig.savefig(os.path.join(filedest, 'data_feature1.pdf'))

    fig = features.plot_feature5(**feature_params_5)
    fig.savefig(os.path.join(filedest, 'data_feature5.pdf'))
    
    
    #load all results from fitting variables
    output = []
    files = glob.glob(os.path.join(filedest, 'output_gen*.npy'))
    for fil in files:
        output.append(np.load(fil))

    #3D array storing everything of shape
    #(num_generations, population_size, [var 0, ..., var n, feat 0, ..., feat m, breedingpop, ??, num_child])
    output = np.array(output)
    
    
    #pareto optimal parameter combinations
    breedingpop = output[:, :, -3] == 0
    
    #the range of tested generations
    generations = np.arange(output.shape[0])
    
    
    #plot the tested variables 
    output_var = output[:, :, :len(variables)]
    varnames = [s[0] for s in variables]
    fig, axes = plt.subplots(int(np.ceil(np.sqrt(len(varnames)))),
                             int(np.ceil(len(varnames) / np.sqrt(len(varnames)))))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    axes = axes.flatten()
    if len(axes) > len(varnames):
        for ax in axes[len(varnames):]:
            fig.delaxes(ax)
    for i in range(len(varnames)):
        axes[i].plot(generations, output_var[:, :, i],
                     '.', color='gray')
        axes[i].plot(generations[np.where(breedingpop)[0]],
                     output_var[:, :, i][breedingpop], 'k.')
        axes[i].set_title(varnames[i])
        axes[i].set_xlabel('gen #')
        axes[i].set_ylabel('value')
            
    fig.savefig(os.path.join(filedest, 'variables.pdf'))

    
    #plot the feature scores
    output_feat = output[:, :, my_eamoo.objpos:my_eamoo.objpos + my_eamoo.obj]
    
    fig, axes = plt.subplots(int(np.ceil(np.sqrt(my_eamoo.obj))),
                             int(np.ceil(my_eamoo.obj / np.sqrt(my_eamoo.obj))))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    axes = axes.flatten()
    if len(axes) > my_eamoo.obj:
        for ax in axes[my_eamoo.obj:]:
            fig.delaxes(ax)
    for i in range(my_eamoo.obj):
        axes[i].semilogy(generations, output_feat[:, :, i],
                         '.', color='gray')
        axes[i].semilogy(generations[np.where(breedingpop)[0]],
                         output_feat[:, :, i][breedingpop], 'k.')
        axes[i].set_title('feature %i' % i)
        axes[i].set_xlabel('gen #')
        axes[i].set_ylabel('value')
    fig.savefig(os.path.join(filedest, 'featurescores.pdf'))



    #trace plot of each breeding optimal child on last gen
    for i in np.arange(my_eamoo.capacity)[breedingpop[-1]]:
        
        kwargs = dict(zip(varnames, output_var[-1, i, :]))
        
        tvec, somavs = get_dataset(**kwargs)
        
        #get the features
        features = GetFeatures(tvec, somavs)
        
        #plot soma traces
        fig = features.plot_traces()
        fig.savefig(os.path.join(filedest, 'traces_gen%i_var%i.pdf' % (len(generations), i)), )

    






