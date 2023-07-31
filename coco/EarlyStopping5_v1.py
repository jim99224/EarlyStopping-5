import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.pyplot import cm
import copy
import matplotlib.colors as mc
import colorsys
from plot_3D import plot_3D
from plot_2D import plot_2D

def defaultINF():
    return np.Inf

def defaultLIST():
    return []

def defaultTRUE():
    return True

def defaultTHRESHOLD():
    return 0.07
# def lighten_color(color, amount=1.5):
#     """
#     Lightens the given color by multiplying (1-luminosity) by the given amount.
#     Input can be matplotlib color string, hex string, or RGB tuple.

#     Examples:
#     >> lighten_color('g', 0.3)
#     >> lighten_color('#F034A3', 0.6)
#     >> lighten_color((.3,.55,.1), 0.5)
#     """
#     try:
#         c = mc.cnames[color]
#     except:
#         c = color
#     c = colorsys.rgb_to_hls(*mc.to_rgb(c))
#     return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
def lighten_color(color, amount=-0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
def default3D():
    return [[], []]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, subclass, warmup=0, deno=300, verbose=False, alpha=0.9, sub_beta = None, ignore=[], path='', trace_func=print, max_iter=None, fixed_threshold=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        #================================
        self.patience = patience
        self.class_average_loss = []
        self.warmup = warmup
        self.deno = deno
        self.path = path
        self.subclass = subclass
        self.ignore = ignore
        self.counter = 0
        self.early_stop = False
        self.alpha = alpha
        self.max_iter = max_iter
        self.iter_loss = []
        self.iteration = 0
        self.best_score = None
        self.previous_convergence_state = True

        self.sub_beta = dict()
        if sub_beta!=None:
            if isinstance(sub_beta, (collections.defaultdict, dict)):
                for key, val in sub_beta:
                    self.sub_beta[key] = val
            else:
                for sub in subclass:
                    self.sub_beta[sub] = sub_beta
        else:
            for sub in subclass:
                self.sub_beta[sub] = 0.07

        self.subclass_loss = collections.defaultdict(defaultLIST)
        self.best_subclass_loss = collections.defaultdict(defaultINF)
        self.averaged_subclass_loss = collections.defaultdict(defaultLIST)
        self.subclass_threshold = dict()
        self.subclass_converge = dict()

        for sub in subclass:
            self.subclass_converge[sub] = True

        self.class_average_loss = []
        #===========visualize============
        self.fig = plt.figure(0, figsize=(12.8, 9.6))
        self.ax = plt.subplot(111)

        self.subclass_color = dict()
        self.subclass_not_converged_color = dict()
        n = len(subclass)
        # n = 0
        # for sub in subclass:
        #     if sub in ignore:
        #         n+=1
        #     else:
        #         n+=2
        # n = len(subclass)+len(subclass-ignore)
        self.color_3D = dict()
        color = iter(cm.rainbow(np.linspace(0, 1, n)))
        for sub in subclass:
            c = next(color)
            self.subclass_color[sub] = c
            tmp = copy.deepcopy(c)
            tmp = [int(i*255//1) for i in tmp]
            self.color_3D[sub] = 'rgb('+str(tmp[:-1])[1:-1]+')'
            # print('rgb'+str(tuple(c*255//1)[:-1]))
            # self.color_3D
            # print(self.color_3D[sub])
            if sub not in ignore:
                # c = next(color)
                c = lighten_color(c)
                self.subclass_not_converged_color[sub] = lighten_color(c)
                # print(list(c))
                c = list(c)
                c = [int(i*255//1) for i in c]
                self.color_3D[sub+'_not_converged'] = 'rgb('+str(c)[1:-1]+')'
                # print(self.color_3D[sub+'_not_converged'])
        self.color_3D['class_average_loss'] = 'rgb(0,0,255)'
                # self.color_3D[sub+'_not_converged'] = 'rgb('+str((list(c)*255//1)[1:-2]+')')
        #================================
        self.verbose = verbose
        self.trace_func = trace_func

        self.plot_3D = dict()
        self.plot_3D['class_average_loss'] = [[], []]
        self.x_axis = dict()
        self.x_axis['class_average_loss'] = 0
        for idx, sub in enumerate(subclass):
            self.plot_3D[sub] = [[], []]
            self.x_axis[sub] = (idx+1)*1
        for idx, sub in enumerate(subclass):
            if sub not in ignore:
                self.plot_3D[sub+'_not_converged'] = [[], []]
                self.x_axis[sub+'_not_converged'] = (idx+1)*1
        self.fixed_threshold = fixed_threshold
        # self.iter_arr = [[], []]
        # print(self.subclass)

    def __call__(self, subclass_loss, iter_loss, model):

        #================================================
        self.iteration+=1
        self.previous_subclass_convergence_state = copy.deepcopy(self.subclass_converge)

        for key, val in subclass_loss.items():
            self.subclass_loss[key].extend(val)
            if len(self.subclass_loss[key])>self.deno:
                self.subclass_loss[key] = self.subclass_loss[key][-self.deno:]

        start = True
        for sub in self.subclass:
            if sub=='iter_loss': continue
            if len(self.subclass_loss[sub])<self.deno:
                start = False
        
        self.iter_loss.append(iter_loss)

        if start:
            class_average_loss = 0.0
            for key, val in self.subclass_loss.items():
                cur_subclass_loss = np.mean(self.subclass_loss[key][-self.deno:])
                self.best_subclass_loss[key] = min(self.best_subclass_loss[key], cur_subclass_loss)
                self.averaged_subclass_loss[key].append(cur_subclass_loss)
                if key=='iter_loss': continue
                class_average_loss+=cur_subclass_loss
            self.class_average_loss.append(class_average_loss/(len(self.subclass_loss.keys())-1))
        else:
            return
        #================================================
        score = self.class_average_loss[-1]

        if not self.previous_convergence_state:
            self.previous_convergence_state = True
            if not self.AllConverged():
                self.previous_convergence_state = False
        else:
            if len(self.class_average_loss)==1:
                self.best_score = score
                if self.fixed_threshold:
                    threshold = 0.0
                    for sub in self.subclass:
                        if sub in self.ignore or sub=='iter_loss': continue
                        threshold = max(threshold, self.averaged_subclass_loss[sub][-1]*self.sub_beta[sub])
                    for sub in self.subclass:
                        if sub in self.ignore or sub=='iter_loss': continue
                        self.subclass_threshold[sub] = threshold
                else:
                    for sub in self.subclass:
                        if sub in self.ignore or sub=='iter_loss': continue
                        self.subclass_threshold[sub] = self.averaged_subclass_loss[sub][-1]*self.sub_beta[sub]

            elif not self.AllConverged():
                self.counter = 0
                self.previous_convergence_state = False
            elif score > self.best_score*self.alpha:
                if self.iteration>self.warmup:
                    self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        #==============================================
        plt.figure(0)
        if len(self.class_average_loss)>1:
            for sub in self.subclass:
                # if not self.subclass_converge[sub] or not self.previous_subclass_convergence_state[sub]:
                #     self.ax.plot(np.array([self.iteration-1, self.iteration]), np.array([self.averaged_subclass_loss[sub][-2], self.averaged_subclass_loss[sub][-1]]), color = self.subclass_not_converged_color[sub], label='not_converged')
                # else:
                #     self.ax.plot(np.array([self.iteration-1, self.iteration]), np.array([self.averaged_subclass_loss[sub][-2], self.averaged_subclass_loss[sub][-1]]), color = self.subclass_color[sub], label=sub)
                
                if not self.subclass_converge[sub]:
                    self.plot_3D[sub+'_not_converged'][0].append(self.iteration)
                    self.plot_3D[sub+'_not_converged'][1].append(self.averaged_subclass_loss[sub][-1])
                else:
                    self.plot_3D[sub][0].append(self.iteration)
                    self.plot_3D[sub][1].append(self.averaged_subclass_loss[sub][-1])


            # l3, = self.ax.plot(np.array([self.iteration-1, self.iteration]), np.array([self.class_average_loss[-2], self.class_average_loss[-1]]), color = 'b', label='subclass_averaged_loss', linestyle=':')
            

            self.plot_3D['class_average_loss'][0].append(self.iteration)
            self.plot_3D['class_average_loss'][1].append(self.class_average_loss[-1])

        #==============================================
        if self.max_iter and self.iteration>=self.max_iter:
            self.early_stop = True
        else:
            self.early_stop = False
        if (self.early_stop == True or self.iteration%500==0) and len(self.class_average_loss)>1:
        #if (self.max_iter and self.iteration>=self.max_iter) or self.iteration%2000==0:
            # handles, labels = plt.gca().get_legend_handles_labels()
            # by_label = dict(zip(labels, handles))
            # plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.savefig(self.path, bbox_inches="tight")
            # if self.early_stop==True:
            #     plt.close('all')
            plot_3D(self.plot_3D, self.x_axis, self.color_3D, self.path)
            plot_2D(self.plot_3D, self.x_axis, self.color_3D, self.path)
            # np.save('class_average_loss.npy', np.array(self.iter_arr))

    def AllConverged(self):
        all_converged = True
        for sub in self.subclass:
            if sub in self.ignore or sub=='iter_loss':
                self.subclass_converge[sub] = True
                continue
            if self.averaged_subclass_loss[sub][-1]>self.best_subclass_loss[sub]+self.subclass_threshold[sub]:
                self.subclass_converge[sub] = False
                all_converged = False
            else:
                self.subclass_converge[sub] = True
        return all_converged
    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path+'checkpoint.pt')
    #     self.val_loss_min = val_loss
