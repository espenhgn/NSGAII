"""
This module forms a population for an evolutionary algorithm. 
The population is a list of parameters. To the list the user must provide a list with
the objectives. 
This can then by sorted eather by nodominated sorting (deb 2004) or in the case of 
a single objective by normal sorting..

(C) Armin Bahl 16.01.2009, UCL, London, UK
modified on ACCN 2011 Course Bedlewo, Poland 09.08.2011 
"""

import numpy as np
from time import time
try:
    from mpi4py import MPI
    mpi4py_loaded = True
except:
    mpi4py_loaded = False
    
class EAMoo:
     
    def __init__(self, size, capacity, variables, obj, infos=0):
        
        # standard population parameters
        self.size = size                    # size of population, must be even!
        self.capacity = capacity      # when new children are born, we have this amount of individuals
        
        self.variables = variables
        self.obj = obj                # number of objectives
        self.infos = infos
        
        self.para = len(self.variables)
                
        self.eta_m_0 = 20               # mutation operator (bigger means less mutation width)
        self.eta_c_0 = 10               # cross over operator
        
        self.finishgen = -1           # no adaption of mutation and crossover parameters
        self.d_eta_m = 0              # Rate of Change per Generation of Mutation Parameter 
        self.d_eta_c = 0              # Rate of Change per Generation of Crossover Parameter 

        self.p_m = 0.1/len(variables) # Mutation probability  
        
        self.no_properties = np.ones(3)*(-1.0)
        self.no_objectives = np.ones(self.obj+self.infos)*(-1)
        
        self.objpos = self.para
        self.infopos = self.objpos + self.obj
        self.rankpos = self.infopos + self.infos
        self.distpos = self.rankpos + 1
        self.fitnesspos = self.distpos + 1
        
        if mpi4py_loaded == True:
            self.comm = MPI.COMM_WORLD
            self.master_mode = self.comm.rank == 0
            self.mpi = self.comm.size > 1
            
            print self.comm.size
            
        else:
            self.master_mode = True
            self.mpi = False
                
        self.checkfullpopulation = None
        self.checkpopulation = None
        
    def setup(self, eta_m_0, eta_c_0, p_m, finishgen=-1, d_eta_m=0, d_eta_c=0):
        self.eta_m_0 = eta_m_0
        self.eta_c_0 = eta_c_0
        self.p_m = p_m
        self.finishgen = finishgen
        self.d_eta_m = d_eta_m
        self.d_eta_c = d_eta_c
        
    def normit(self, p):
        p_norm = np.zeros(len(p), dtype=float)
        
        for i in range(len(p)):
            p_norm[i] = (p[i]-self.variables[i][1])/(self.variables[i][2] - self.variables[i][1])
                   
        return p_norm


    def unnormit(self, p_norm):
        p = np.zeros(len(p_norm), dtype=float)
    
        for i in range(len(p_norm)):
            p[i] = p_norm[i]*(self.variables[i][2] - self.variables[i][1]) + self.variables[i][1]
                   
        return p
    
    def getpopulation_unnormed(self):
        unnormed_population = []
        for individual in self.population:
            individual_unnormed = individual.copy()
            individual_unnormed[:self.para] = self.unnormit(individual[:self.para])
            unnormed_population.append(individual_unnormed)

        return np.array(unnormed_population)

    def print_individual(self, i):
        print "##############################################################################"
        
        p = self.unnormit(self.population[i][:self.para])

        print "Parameters:"
        
        for j in range(self.para):
            print self.variables[j][0], "=", p[j]
        
        print "Objectives:"
        print self.population[i][self.objpos:self.objpos+self.obj]
        print "##############################################################################"

    def initpopulation(self):
        
        init_parameters = np.random.rand(self.size, self.para)
        init_properties = np.ones((self.size, self.obj+self.infos+3))*(-1.0)

        self.population = np.c_[init_parameters, init_properties]       
    
    def evolution(self, generations):
        if(self.master_mode == True):
            self.eta_c = self.eta_c_0
            self.eta_m = self.eta_m_0
        
            self.initpopulation()
            self.evaluate()
            
            if(len(self.population) < 2):
                print "Population died out...!"
                
                # tell the slaves (if any) to terminate
                if self.mpi == True:
                    for i in range(1, self.comm.size):
                        self.comm.send(None, dest=i)
                return
            
            self.assign_fitness()
            
            if(self.checkpopulation != None):
                self.checkpopulation(self, 0)
                    
            for gen in range(generations):
            #for gen in range(1, generations):
                
                # Change the Crossover and Mutation Parameters
                if gen > self.finishgen and self.finishgen != -1:
                    self.eta_c += self.d_eta_c
                    self.eta_m += self.d_eta_m
                
                #print self.eta_c,self.eta_m
                                
                self.selection()
                self.crossover()
                self.mutation()
                
                self.evaluate()

                self.assign_fitness()
                
                if self.checkfullpopulation != None:
                    self.checkfullpopulation(self, gen)
                    
                self.new_generation()
                
                if(self.checkpopulation != None):
                    self.checkpopulation(self, gen)
        
            # tell the slaves (if any) to terminate
            if self.mpi == True:
                for i in range(1, self.comm.size):
                    self.comm.send(None, dest=i)

        else:      
            self.evaluate_slave()
        
                
                
    def selection(self):
        """
        In this step the mating pool is formed by selection
        The population is shuffelded and then each individal is compared with the next and only
        the better will be tranfered into the mating pool
        then the population is shuffelded again and the same happens again
        """
        
        # the population has the size N now
        # and all fitnesses are assigned!
        
        mating_pool = []
        
        for k in [0,1]:
            population_permutation = self.population[np.random.permutation(len(self.population))]
            # -1 because in the cases off odd population size!
            for i in np.arange(0, len(self.population)-1, 2):
                fitness1 = population_permutation[i][-1]
                fitness2 = population_permutation[i+1][-1]
                
                if(fitness1 < fitness2):
                    mating_pool.append(population_permutation[i])
                else:
                    mating_pool.append(population_permutation[i+1])
        
        # now we have a mating pool
        
        # this is our new population
        self.population = np.array(mating_pool)         
            
            
    def crossover(self):
        
        children = []
        
        while(len(children) + len(self.population) < self.capacity):
            
            # choose two random parents
            p = int(np.random.random()*len(self.population))
            q = int(np.random.random()*len(self.population))
            
            parent1 = self.population[p][:self.para]
            parent2 = self.population[q][:self.para]
            
            parameters1 = np.empty(self.para)
            parameters2 = np.empty(self.para)
                
            # determine the crossover parameters
            for i in range(self.para):
                u_i = np.random.random()
            
                if u_i <= 0.5:
                    beta_q_i = pow(2.*u_i, 1./(self.eta_c+1))
                else:
                    beta_q_i = pow(1./(2*(1-u_i)), 1./(self.eta_c+1))
            
                parameters1[i]  = 0.5*((1+beta_q_i)*parent1[i] + (1-beta_q_i)*parent2[i])
                parameters2[i]  = 0.5*((1-beta_q_i)*parent1[i] + (1+beta_q_i)*parent2[i])
            
                # did we leave the boundary?
                if(parameters1[i] > 1):
                    parameters1[i] = 1
                
                if(parameters1[i] < 0):
                    parameters1[i] = 0
                
                if(parameters2[i] > 1):
                    parameters2[i] = 1
                
                if(parameters2[i] < 0):
                    parameters2[i] = 0
                
            offspring1 = np.r_[parameters1, self.no_objectives, self.no_properties]
            offspring2 = np.r_[parameters2, self.no_objectives, self.no_properties]

            children.append(offspring1)
            children.append(offspring2)               

        children = np.array(children)
        self.population = np.r_[self.population, children]

         
    def mutation(self):
        
        # polynomial mutation (Deb, 124)
        for k in range(len(self.population)):
            
            individual = self.population[k]
            
            # if this individual is a parent
            if individual[self.fitnesspos] != -1:
                continue
        
            for i in range(self.para):
            
                # each gene only mutates with a certain probability
                m = np.random.random()
                
                if(m < self.p_m):
                    r_i = np.random.random()
                
                    if r_i < 0.5:
                        delta_i = pow(2*r_i, 1./(self.eta_m+1)) - 1
                    else:
                        delta_i = 1-pow(2*(1-r_i), 1./(self.eta_m+1))
                        
                    individual[i] += delta_i
                    
                    # did we leave the boundary?
                    if(individual[i] > 1):
                        individual[i] = 1
                    
                    if(individual[i] < 0):
                        individual[i] = 0
            
            individual[self.para:] = np.r_[self.no_objectives, self.no_properties]

    def evaluate(self):
        
        new_population = []
        
        # is the master alone?
        if(self.mpi == False):

            for individual in self.population:
                
                # only evaluate those that are really new!
                if individual[self.fitnesspos] == -1:
                    
                    parameters = individual[:self.para]
                    parameters_unnormed = self.unnormit(parameters)
                    
                    # make a dictionary with the unormed parameters and send them to the evaluation function
                    dict_parameters_normed = dict({})
                    for i in range(len(self.variables)):
                        dict_parameters_normed[self.variables[i][0]] = parameters_unnormed[i]
                    objectives_error = self.get_objectives_error(dict_parameters_normed)
                        
                    #objectives_error = self.get_objectives_error(self.unnormit(parameters))
                    
                    if(objectives_error != None):
                        new_population.append(np.r_[parameters, objectives_error, self.no_properties])
                else:
                    new_population.append(individual)
        else:
            # distribute the individuals among the slaves
            i = 0
            for individual in self.population:
                if individual[self.fitnesspos] == -1:
                    parameters = individual[:self.para]
                
                    dest = i%(self.comm.size-1) + 1
                    
                    print 'send to any dest: %i' % dest
                    
                    self.comm.send(parameters, dest=dest)
                    i += 1
                else:
                    new_population.append(individual)
                    
            # Receive the results from the slaves
            for i in range(i):
                result = self.comm.recv(source=MPI.ANY_SOURCE)
                
                print 'receive from any dest: %i' % (int(i)+1)
                
                if result != None:
                    new_population.append(np.r_[result[0], result[1], self.no_properties])
        
        self.population = np.array(new_population)
    
    def evaluate_slave(self):
        
        # We wait for parameters
        # we do not see the whole population!
        
        while(True):
            parameters = self.comm.recv(source=0) # wait....
            
            # Does the master want the slave to shutdown?
            if(parameters == None):
                # Slave finishing...
                break
            parameters_unnormed = self.unnormit(parameters)
                    
            # make a dictionary with the unormed parameters and send them to the evaluation function
            dict_parameters_normed = dict({})
            for i in range(len(self.variables)):
                dict_parameters_normed[self.variables[i][0]] = parameters_unnormed[i]
            objectives_error = self.get_objectives_error(dict_parameters_normed)
            
            #objectives_error = self.get_objectives_error(self.unnormit(parameters))
            
            if(objectives_error == None):
                self.comm.send(None, dest=0)
            else: 
                self.comm.send([parameters, objectives_error], dest=0)
    
    def assign_fitness(self):           
        """
        are we in a multiobjective regime, then the selection of the best individual is not trival
        and must be based on dominance, thus we determine all non dominated fronts and only use the best
        to transfer into the new generation
        """
        if(self.obj > 1):
            self.assign_rank()

            new_population = np.array([])
            
            maxrank = self.population[:,self.rankpos].max()

            for rank in range(0, int(maxrank)+1):
                
                new_front = self.population[np.where(self.population[:,self.rankpos] == rank)]
                
                new_sorted_front = self.crowding_distance_sort(new_front)
                
                if(len(new_population) == 0):
                    new_population = new_sorted_front
                else:
                    new_population = np.r_[new_population, new_sorted_front]
                
            self.population = new_population
                         
        else:
            # simple sort the objective value
            ind = np.argsort(self.population[:,self.objpos])
            self.population = self.population[ind]
        
        # now set the fitness, indiviauls are sorted, thus fitnes is easy to set
        fitness = range(0, len(self.population[:,0]))
        self.population[:,-1] = fitness   
                    
    
    def new_generation(self):
        # the worst are at the end, let them die, if there are too many
        if(len(self.population) > self.size):
            self.population = self.population[:self.size]
         
    def dominates(self, p, q):
        
        objectives_error1 = self.population[p][self.objpos:self.objpos+self.obj]
        objectives_error2 = self.population[q][self.objpos:self.objpos+self.obj]
        
        diff12 = objectives_error1 - objectives_error2
        
        # is individdum equal or better then individdum two?
        # and at least in one objective better
        # then it dominates individuum2
        # if not it does not dominate two (which does not mean that 2 may not dominate 1)
        return ( ((diff12<= 0).all()) and ((diff12 < 0).any()) )

    
    def assign_rank(self):
            
        F = dict()

        P = self.population
        
        S = dict()
        n = dict()
        F[0] = []
        
        # determine how many solutions are dominated or dominate
        for p in range(len(P)):
            
            S[p] = []       # this is the list of solutions dominated by p
            n[p] = 0        # how many solutions are dominating p
            
            for q in range(len(P)):
                
                if self.dominates(p, q):
                    S[p].append(q)      # add q to the list of solutions dominated by p
                elif self.dominates(q, p):
                    n[p] += 1           # q dominates p, thus increase number of solutions that dominate p
                
            
            if n[p] == 0:       # no other solution dominates p
                
                # this is the rank column
                P[p][self.rankpos] = 0
                
                F[0].append(p)  # add p to the list of the first front
            
        # find the other non dominated fronts
        i = 0
        while len(F[i]) > 0:
            Q = []              # this will be the next front
            
            # take the elements from the last front
            for p in F[i]:
                
                # and take the elements that are dominated by p
                for q in S[p]:
                    # decrease domination number of all elements that are dominated by p
                    n[q] -= 1
                    # if the new domination number is zero, than we have found the next front       
                    if n[q] == 0:
                        
                        P[q][self.rankpos] = i + 1
                        Q.append(q)
            
            i += 1
            F[i] = Q    # this is the next front
    
    
    def crowding_distance_sort(self, front):
        
        sorted_front = front.copy()
        
        l = len(sorted_front[:,0])
        
        sorted_front[:,self.distpos] = np.zeros_like(sorted_front[:,0])
        
        for m in range(self.obj):
            ind = np.argsort(sorted_front[:,self.objpos + m])
            sorted_front = sorted_front[ind]

            # definitely keep the borders
            sorted_front[0, self.distpos] += 1000000000000000
            sorted_front[-1, self.distpos] += 1000000000000000

            fm_min = sorted_front[0, self.objpos + m]
            fm_max = sorted_front[-1, self.objpos + m]
            
            for i in range(1, l - 1):
                sorted_front[i, self.distpos] += (sorted_front[i+1, self.objpos + m] - sorted_front[i-1, self.objpos + m])/(fm_max - fm_min)

        ind = np.argsort(sorted_front[:,self.distpos])
        sorted_front = sorted_front[ind]
        sorted_front = sorted_front[-1 - np.arange(len(sorted_front))]
                                                         
        return sorted_front