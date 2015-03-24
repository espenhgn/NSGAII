#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

class GetFeatures(object):
    '''Class to extract features from step-current soma traces'''
    def __init__(self, tvec, traces):
        '''Initialize object. Input tvec is global time vector,
        traces an n x tvec.size array containing the somatic traces'''
       
        self.tvec = tvec
        self.traces = traces
        
        #getting spiketrains, sounds useful....
        APtrains = []
        for x in self.traces:
            APtrains.append(self.return_spiketrain(v = x, v_t = -20))
        
        self.APtrains = np.array(APtrains)
    
    def return_spiketrain(self, v, v_t = -20):
        '''Takes voltage trace v, and some optional voltage threshold v_t,
        returning spike train array of same length as v.'''
        AP_train = np.zeros(v.shape)
        [u] = np.where(v > v_t)
    
        if len(u) > 0:
            #splitting u in intervals if there are more than 1 AP
            w = {}
            j = 0
            i_n = 0
            i = 0
    
            for i in xrange(1,len(u)):
                if u[i]!=u[i-1]+1:
                    w[j] = u[i_n:i]
                    i_n = i
                    j += 1
            w[j] = u[i_n:i+1]
    
            for j in xrange(len(w.keys())):
                [n] = np.where(v[w[j]] == v[w[j]].max())
                AP_train[w[j][n]] = 1.
    
        return AP_train


    def feature0(self,
                 xedges = np.arange(-100, 55, 5),
                 yedges = np.arange(-10, 25, 1),
                 threshold=2,
                 smooth=True):
        '''Return the phase plane trajectory 2D-histogram of the spikes'''
        diff_traces = np.zeros(self.traces.shape)
        for i, trace in enumerate(self.traces):
            diff_traces[i, ] = np.gradient(trace)
        
        
        hist2 = np.zeros((yedges.size-1, xedges.size-1))
        
        for i, diff_trace in enumerate(diff_traces):
            inds = abs(diff_trace) >= threshold
            if np.any(inds):
                hist2 += np.histogram2d(diff_trace[inds],
                                    self.traces[i, inds],
                                    bins=[yedges, xedges])[0]
        
        if smooth:
            y = np.ones((3,3))
            y *= ss.gaussian(3,1)[0]
            y[1,1] = 1
            y /= y.sum()
            hist2 = ss.convolve2d(hist2, y, 'same')
        
        return hist2

    def feature1(self,
                 rows=[-1],
                 inds=np.r_[range(800, 1600), range(7200, 8000)]):
        '''Return the sum of potentials from the negative step currents'''
        return self.traces[rows, inds]
     
    #def feature2(self):
    #    '''Return the trace of the rebound burst'''
    #    [inds] = np.where(self.tvec >= 1000)
    #    return self.traces[-1, inds]

    #def feature3(self):
    #    '''return the cumsum of trace with 70 pA input current'''
    #    return self.traces[6, ].cumsum()
    
    #def feature4(self, xedges = np.linspace(-100, 30, 27),
    #             yedges = np.linspace(-200, 300, 21), threshold=10):
    #    '''Return the phase plane trajectory 2D-histogram of the spikes'''
    #    
    #    trace = self.traces[7, 1E4:-1]
    #    
    #    
    #    diff_trace = np.diff(trace)*10
    #    
    #    
    #    #hist2 = np.zeros((yedges.size-1, xedges.size-1))
    #    hist2 = np.histogram2d(diff_trace, trace[1:], bins=[yedges, xedges])[0]
    #    
    #    return hist2
        
    def feature5(self, **kwargs):
        '''return the cumsum of the APtrains'''
        cumsums = []
        for x in self.APtrains:
            cumsums.append(x.cumsum())
        
        return np.array(cumsums)
    
    #def feature6(self):
    #    '''return the rebound burst from the last dataset'''
    #    return self.traces[7, 1E4:-1]
    
    def feature7(self):
        '''return the value of the resting potential, i.e mean potential of all
        traces before stimulus onset'''
        inds = np.arange(0, 1000)
        return self.traces[:, inds].mean()
        
    
    def plot_feature0(self, new_fig=True, **kwargs):
        data = self.feature0(**kwargs)
        
        if new_fig:
            plt.figure()
        else:
            plt.gca()
        plt.imshow(data, interpolation='nearest', origin='bottom', )
        plt.axis('tight')
        plt.ylabel(r'$dV/dt$ (-)')
        plt.xlabel(r'$V_\mathrm{soma}$ (-)')
        plt.colorbar()
        plt.title('Feature 0')

        return plt.gcf()

    
    def plot_feature1(self, new_fig=True, **kwargs):
        if new_fig:
            plt.figure()
        else:
            plt.gca()
        plt.plot(self.feature1(**kwargs), label='sag&rebound')
        plt.axis('tight')
        plt.legend(loc='best')
        plt.xlabel('time (ms)')
        plt.ylabel('Potential (mV)')
        plt.title('Feature 1')

        return plt.gcf()
        

    def plot_feature5(self, new_fig=True, **kwargs):
        data = self.feature5(**kwargs)
        if new_fig:
            plt.figure() 
        else:
            plt.gca()
        plt.imshow(data, interpolation='nearest')
        plt.axis('tight')
        plt.colorbar()
        plt.ylabel('Trial ()')
        plt.xlabel('Timestep ()')
        plt.title('Feature 5')
        
        return plt.gcf()
        

    def plot_traces(self, ymin=None, ymax=None, new_fig=True):
        if new_fig:
            fig = plt.figure()
        else:
            fig = plt.gcf()
        
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0, hspace=None)
        
        nrows = self.traces.shape[0]
        #get global min and maxima of traces to set axis correctly
        i = 1
        for x in self.traces:
            ax = fig.add_subplot(nrows, 1, i,  frameon=False, xticks=[], yticks=[], clip_on=False)
            ax.plot(self.tvec, x)
            if ymin != None and ymax != None:
                plt.ylim([ymin, ymax])
            else:
                plt.ylim([self.traces.min(), self.traces.max()])
            plt.xlim([self.tvec[0], self.tvec[-1]])
            
            if i == 1:
                ax.plot([self.tvec[-1]-500, self.tvec[-1]-400],
                    [self.traces.min(), self.traces.min()],
                    lw=2, color='k', clip_on=False)
                ax.text(self.tvec[-1]-500, self.traces.min()-30, '100 ms')
                
                ax.plot([self.tvec[-1], self.tvec[-1]], [-50, 0],
                    lw=2, color='k', clip_on=False)
                ax.text(self.tvec[-1]+5, -25, '50 mV')                
            i += 1
        
        for o in fig.findobj():
            o.set_clip_on(False)
            
        return plt.gcf()