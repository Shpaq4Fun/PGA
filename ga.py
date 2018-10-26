import random, time, math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors, patheffects
from matplotlib.gridspec import GridSpec


class GA:
    def __init__(self, fitfun, dnaLength=None, mutationRate=0.01, populationSize=100,
                 userdata=None, elitecount=1, plotting='slow', ltc='stall',
                 lcon=None, ucon=None, prl=False, learningRate=1.2, epochs=1, customPlot=[],
                 progIncrement=1, stochastic=False, limMode='random'):
        self.mutationRate = mutationRate  # Mutation rate
        self.population = []  # Population of DNA objects
        self.matingPool = []  # Proportional mating pool for roulette selection
        self.generation = 0   # Current generation
        self.generationInEpoch = 0
        self.popsize = populationSize  # Size of population
        self.elitecount = elitecount  # Amount of best objects surviving directly from one generation to next
        self.userdata = userdata  # User data for fitness function
        self.plotprocdata = [[], [], [], [], [], [], [], [], []]  # Historical record of entire process, required for plotting
        self.plothandles = [0]*9  # Handles of figure and axis of process drawing 7
        self.besthistory = []
        self.speed = plotting
        self.ltc = ltc
        self.lastbest = 0
        self.learningRate = learningRate
        self.lcon = lcon
        self.ucon = ucon
        self.parallel = prl
        self.customPlot = customPlot
        self.progIncrement = progIncrement
        self.stochastic=stochastic
        self.SAtemp = 0.5
        self.limMode = limMode
        # self.accProb = 1
        if self.customPlot is None:
            self.customPlot = []
        # Pass fitness function to DNA class definition
        # setattr(DNA, 'fitfun', classmethod(fitfun))

        if self.lcon is None or len(self.lcon) is not dnaLength:
            self.lcon = np.array([-1]*dnaLength)
        if self.ucon is None or len(self.ucon) is not dnaLength:
            self.ucon = np.array([1]*dnaLength)

        # Create population
        for i in range(self.popsize):
            self.population.append(DNA(dnaLength, fitfun, self.learningRate,
                                       self.lcon, self.ucon, self.userdata, limMode))

        for i in range(self.popsize):
            self.population[i].fitfun(self.population[i], self.userdata)
        self.sortFitness()

        # Add population data to historical record
        self.updateHistory()

        # Draw first generation
        self.plotfun(self)

        start = time.time()
        for i in range(epochs):
            self.mainLoop()
            if i < epochs-1:
                self.progressify()
            # if self.termination():
            #     break

        end = time.time()
        self.timeElapsed = end - start
        print("Time elapsed: ", round(self.timeElapsed, 3), " seconds.")

    def plotBarGenes(self, ax):
        genePool = np.zeros((self.popsize,len(self.population[0].genes)))
        for i in range(self.popsize):
            # for j in range(len(self.population[0].genes)):
            genePool[i,:]=self.population[i].genes
        # genePool = (genePool - np.amin(genePool, axis=0))
        # genePool = genePool / np.amax(genePool, axis=0)
        genePool = (genePool - self.lcon)
        genePool = genePool / (self.ucon - self.lcon)
        ax.pcolor(genePool, cmap='gray')

    def plotGraphGenes(self, ax):
        ax.clear()
        genePool = np.zeros((self.popsize, len(self.population[0].genes)))
        for i in range(self.popsize):
            # for j in range(len(self.population[0].genes)):
            genePool[i, :] = self.population[i].genes

        if len(set(self.lcon)) <= 1 and len(set(self.ucon)) <= 1:
            ax.set_ylim(self.lcon[0], self.ucon[0])
        else:
            genePool = (genePool - self.lcon)
            genePool = genePool / (self.ucon - self.lcon)
            ax.set_ylim(0, 1)
        ax.plot(range(len(self.population[0].genes)), genePool.transpose(), c='white', alpha=0.15, lw=1)
        # ax.plot(range(len(self.population[0].genes)), genePool[-1,:], c='lime', alpha=1, lw=1)
        ax.plot(range(len(self.population[0].genes)), genePool[-1,:], c='lime', alpha=1, lw=1)
        ax.grid(True)
    # def plotGraphGenes(self, ax):
    #     ax.clear()
    #     genePool = np.zeros((self.popsize, len(self.population[0].genes)))
    #     for i in range(self.popsize):
    #         # for j in range(len(self.population[0].genes)):
    #         genePool[i, :] = self.population[i].genes
    #
    #     if len(set(self.lcon)) <= 1 and len(set(self.ucon)) <= 1:
    #         ax.set_ylim(self.lcon[0], self.ucon[0])
    #     else:
    #         genePool = (genePool - self.lcon)
    #         genePool = genePool / (self.ucon - self.lcon)
    #         ax.set_ylim(0, 1)
    #     ax.plot(range(len(self.population[0].genes)), genePool.transpose(), c='white', alpha=0.15, lw=1)
    #     # ax.plot(range(len(self.population[0].genes)), genePool[-1,:], c='lime', alpha=1, lw=1)
    #     ax.plot(range(len(self.population[0].genes)), genePool[-1,:], c='lime', alpha=1, lw=1)
    #     ax.grid(True)

    def plotfun(self, ifprog=False):
        def sigmoid(x, scale, yoffset):
            # y = []
            t = (1 / (1 + np.exp(-x)))
            t = (t - (1 / (1 + math.exp(-x[0])))) * (1 + (2 * math.exp(-x[-1])))
            y = scale * t + yoffset
            # for i in x:
            #     t = (1 / (1 + math.exp(-i)))
            #     t = (t - (1 / (1 + math.exp(-x[0])))) * (1 + (2 * math.exp(-x[-1])))
            #     y.append(scale * t + yoffset)
            return y

        norms = colors.Normalize(vmin=0, vmax=max(self.plotprocdata[1]))

        if self.generation is 0:
            # plt.ion()
            self.plothandles[0] = plt.figure(1)
            gs = GridSpec(len(self.customPlot) + 1, 4)
            ax = self.plothandles[0].add_subplot(gs[0, :-1])
            # ax = plt.subplot(len(self.customPlot)+1, 2, 1)
            # self.plothandles[2] = plt.text(x=self.generation / 2, y=self.besthistory[-1] / 5, alpha=0.25,
            #                                s=[len(self.population[0].genes), round(self.besthistory[-1], 4)],
            #                                fontsize=self.plothandles[0].get_figwidth() * 8, color='white',
            #                                verticalalignment='center', horizontalalignment='center', weight='bold')
            if self.speed is 'fast':

                # self.plothandles[1] = plt.scatter(self.plotprocdata[0], self.plotprocdata[1], alpha=1, marker='s', s=25,
                #                                   c=cm.jet(norms(self.plotprocdata[1]), bytes=True) / 255)

                self.plothandles[5] = plt.plot(self.plotprocdata[0], np.transpose(self.plotprocdata[5]), alpha=0.5, c='white', lw=0.5)
                self.plothandles[4] = plt.plot(self.plotprocdata[0], self.plotprocdata[6], alpha=1, c='white', lw=0.7)
                self.plothandles[1] = plt.plot(self.plotprocdata[0], self.plotprocdata[1],c='red')
            elif self.speed is 'slow':
                # verts = list(zip([-0.1, 0.1, 0.1, -0.1], [-1., -1., 1., 1.]))
                self.plothandles[1] = plt.scatter(self.plotprocdata[0], self.plotprocdata[1], alpha=1, marker='_', s=25,
                                                  c=cm.jet(norms(self.plotprocdata[1]), bytes=True) / 255)

            ax.set_title("Fitness: " + str(round(self.besthistory[-1], 4)) + ", Learning rate: " + str(round(self.learningRate, 3)), color='white')

            ax2 = self.plothandles[0].add_subplot(gs[0, 3])
            self.plotGraphGenes(ax2)
            ax3 = self.plothandles[0].add_subplot(gs[1, 3])
            self.plotBarGenes(ax3)

            if len(self.customPlot) is not 0:
                for i in range(len(self.customPlot)):
                    ax = self.plothandles[0].add_subplot(gs[i+1, :-1])
                    self.plothandles.append(ax)
                    cp = self.customPlot[i]
                    cp(self, self.plothandles[-1])

            self.figFormat(self.plothandles[0])

            plt.pause(0.0001)
        else:
            # self.plothandles[0].axes[0]
            # plt.subplot(len(self.customPlot) + 1, 2, 1)
            # self.plothandles[2].set_text(str(len(self.population[0].genes)) + ", " + str(round(self.besthistory[-1], 4)))
            # self.plothandles[2].set_y(self.besthistory[-1] / 5)
            # self.plothandles[2].set_x(self.generation / 2)
            # self.plothandles[2].set_fontsize(self.plothandles[0].get_figwidth() * 8)
            # self.plothandles[1] = plt.scatter(self.plotprocdata[0], self.plotprocdata[1], alpha=1, marker='_', s=25,
            #                                   c=cm.jet(norms(self.plotprocdata[1]), bytes=True) / 255)

            self.plothandles[0].axes[0].set_xlim(- 0.5,
                                                 max(self.plotprocdata[0]) + 0.5)
            self.plothandles[0].axes[0].set_ylim(0.9*min(min(self.plotprocdata[1]),min(self.plotprocdata[6])), 1.02*max(self.plotprocdata[1]))
            self.plothandles[0].axes[0].set_title("Fitness: " + str(round(self.besthistory[-1], 4)) + ", LR: " +
                                str(round(self.learningRate, 3)) + ", SAtemp: " + str(round(self.SAtemp, 3)), color='white')

            if self.speed is 'slow':
                self.plothandles[1].set_offsets(np.c_[self.plotprocdata[0], self.plotprocdata[1]])
                self.plothandles[1].set_color(cm.jet(norms(self.plotprocdata[1]), bytes=True) / 255)
                t = np.linspace(self.generation - 1, self.generation, 14)
                # z = []
                # for i in range(len(self.plotprocdata[6])):
                #
                #     z1 = sigmoid(np.linspace(-6, 6, 14), (self.plotprocdata[8][i] - self.plotprocdata[7][i]),
                #                  self.plotprocdata[7][i])
                #     z2 = sigmoid(np.linspace(-6, 6, 14), (self.plotprocdata[8][i] - self.plotprocdata[6][i]),
                #                  self.plotprocdata[6][i])
                #     z.append(z1)
                #     z.append(z2)
                # if len(z) is not 0:
                #     self.plothandles[0].axes[0].plot(t, np.transpose(z), c='yellow', alpha=0.3, lw=0.5)
                y = []
                for i in range(len(self.plotprocdata[3])):
                    y.append(sigmoid(np.linspace(-6, 6, 14), (self.plotprocdata[5][i] - self.plotprocdata[3][i]),
                                self.plotprocdata[3][i]))
                if len(y) is not 0:
                    self.plothandles[0].axes[0].plot(t, np.transpose(y), c='red', alpha=0.8, lw=0.5, label='Constructive mutation')
                if ifprog is True:
                    self.plothandles[0].axes[0].plot([self.generation - 0.5, self.generation - 0.5],
                                                     [self.besthistory[0], self.besthistory[-2]], c='lime', lw=2)

            elif self.speed is 'fast':
                self.plothandles[1][0].set_xdata(self.plotprocdata[0])
                self.plothandles[1][0].set_ydata(self.plotprocdata[1])
                self.plothandles[4][0].set_xdata(self.plotprocdata[0])
                self.plothandles[4][0].set_ydata(self.plotprocdata[6])
                # self.plothandles[5][0].set_xdata(np.tile(np.transpose(np.expand_dims(np.array(self.plotprocdata[0]), axis=1)), (9, 1)))
                # self.plothandles[5][0].set_xdata(self.plotprocdata[0])
                [self.plothandles[5][i].set_xdata(self.plotprocdata[0]) for i in range(9)]
                # self.plothandles[5][0].set_ydata(self.plotprocdata[5])
                [self.plothandles[5][i].set_ydata(np.transpose(self.plotprocdata[5][i, :])) for i in range(9)]
                if ifprog is True:
                    self.plothandles[0].axes[0].plot([self.generation - 0.5, self.generation - 0.5],
                                                     [min(self.plotprocdata[6]), self.besthistory[-2]], c='lime', lw=2)

            self.plotGraphGenes(self.plothandles[0].axes[1])
            self.plotBarGenes(self.plothandles[0].axes[2])

            if ifprog is True:
                self.plothandles[0].axes[0].plot([self.generation - 0.5, self.generation - 0.5], [self.besthistory[0], self.besthistory[-2]], c='green', lw=2)

            if len(self.customPlot) is not 0:
                for i in range(len(self.customPlot)):
                    cp = self.customPlot[i]
                    cp(self, self.plothandles[0].axes[i+3])

        # t1 = time.time()
        # self.plothandles[0].canvas.draw()

        plt.pause(0.0001)

    def figFormat(self, fig=plt.gcf()):
        def onResize(event):
            plt.tight_layout()

        fig.set_facecolor((0.1, 0.1, 0.1))
        fig.canvas.toolbar.pack_forget()
        for ax in fig.axes:
            ax.grid(True)
            ax.set_facecolor('none')
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.tick_params(axis="both", which="major", bottom=False, top=False, labelbottom=True, left=False,
                           right=False, labelleft=True, colors='white', grid_alpha=0.2)

        # fig.tight_layout()
        cid = fig.canvas.mpl_connect('resize_event', onResize)
        cid2 = fig.canvas.mpl_connect('draw_event', onResize)

    def progressify(self):
        self.lcon = np.concatenate((self.lcon, [self.lcon[0]] * self.progIncrement))
        self.ucon = np.concatenate((self.ucon, [self.ucon[0]] * self.progIncrement))
        for i in range(self.popsize-self.elitecount, self.popsize):
            self.population[i].genes = np.concatenate((self.population[i].genes, ([0] * self.progIncrement)))
            self.population[i].lcon = self.lcon
            self.population[i].ucon = self.ucon
            self.population[i].fitfun(self.population[i], self.userdata)
        for i in range(self.popsize-self.elitecount):
            self.population[i].genes = np.concatenate((self.population[i].genes,[random.uniform(self.lcon[0],self.ucon[0])]*self.progIncrement))
            self.population[i].lcon = self.lcon
            self.population[i].ucon = self.ucon
            self.population[i].fitfun(self.population[i], self.userdata)
        self.generation += 1
        self.sortFitness()
        self.updateHistory()
        self.plotfun(ifprog=True)
        best = self.population[-1]
        self.generationInEpoch = 0
        print("Progressing at generation: ", self.generation, ", Fitness: ", round(best.fitness, 2))

    def mainLoop(self):
        timing = []
        while True:
        # for i in range(10):
            # Genetic operators
            t1 = time.time()
            self.naturalSelection()
            self.createNewGeneration()
            self.sortFitness()
            self.simulatedAnnealing()
            self.updateHistory()
            self.plotfun(ifprog=False)
            t2 = time.time()
            # Periodically print status info
            timing.append(t2-t1)
            if divmod(self.generation, 10)[1] is 0:
                best = self.population[-1]
                print("Generation: ", self.generation, ", Fitness: ", round(best.fitness, 4),
                      ", Iter time: ", np.mean(timing), "s.")
                timing = []
                # print(self.population[-1].genes)
            if self.localTermination():
                self.lastbest = self.besthistory[-1]
                break

    def simulatedAnnealing(self):
        best = self.population[-1]
        # new = best
        new = DNA(len(best.genes), best.fitfun, self.learningRate,
                    self.lcon, self.ucon, self.userdata, self.limMode)
        modRange = self.ucon - self.lcon
        diff = np.random.uniform(-0.001, 0.001, len(new.genes)) * modRange
        new.genes = best.genes + diff
        new.checkConstraints()
        new.fitfun(new, new.userdata)
        # print("BF: ", str(best.fitness), ", NF: " + str(new.fitness))
        if new.fitness > best.fitness:
            self.population[0].genes = new.genes
            print("SA: better")
        else:
            self.SAtemp *= (1-0.015)
            # self.accProb = np.exp(((new.fitness - best.fitness)/best.fitness)/self.SAtemp)
            # print("SAprob: ", self.accProb)
            if random.random() < self.SAtemp:
                self.population[0].genes = new.genes
                print("SA: worse")

    def localTermination(self):
        if self.ltc is 'inf':
            return False
        if self.ltc is 'stall':
            k = abs(self.besthistory[-1] - self.besthistory[max(0, self.generation-15)])
            if k < 0.00001 and self.generation > 15:
                return True
            else:
                return False
        elif self.ltc is 'quant':
            a = [o.fitness for o in self.population]
            k = np.percentile(a, 5)
            if k >= 0.995*self.besthistory[-1] and self.generation > 15:
                return True
            else:
                return False
        elif self.ltc is 'qs':
            a = [o.fitness for o in self.population]
            k = np.percentile(a, 5)
            thr = 0.99 * self.besthistory[-1]
            # print(k, thr)
            if k >= thr:
                kk = abs(self.besthistory[-1] - self.besthistory[max(0, self.generation - 15)])
                # print(kk)
                if kk < 0.001*self.besthistory[-1] and self.generationInEpoch > 20:
                    return True
            else:
                # print(k)
                return False
        elif self.ltc is 'cross':
            k = abs(self.besthistory[-1] - self.besthistory[max(0, self.generation - 15)])
            if k < 0.05 and self.lastbest < self.besthistory[-1] and self.generation > 15:
                return True
            else:
                return False

    def updateHistory(self):
        if self.speed is 'fast':
            # fitness:
            self.plotprocdata[0].extend([self.generation] * 1)
            self.plotprocdata[1].append(self.population[-1].fitness)
            # percentiles:
            a = [o.fitness for o in self.population]
            if self.generation is 0:
                self.plotprocdata[5] = np.expand_dims(np.percentile(a, np.array(list(range(10,100,10)))),axis=1)
            else:
                self.plotprocdata[5] = np.concatenate((self.plotprocdata[5],
                    np.expand_dims(np.percentile(a, np.array(list(range(10,100,10)))),axis=1)),axis=1)
            # self.plotprocdata[3].extend([self.generation] * self.popsize)
            # percentile 5
            self.plotprocdata[6].append(np.percentile(a, 5))
        elif self.speed is 'slow':
            # self.plotprocdata[0][:] = []
            # self.plotprocdata[1][:] = []
            # self.plotprocdata[2][:] = []
            self.plotprocdata[3][:] = []
            # self.plotprocdata[4][:] = []
            self.plotprocdata[5][:] = []
            self.plotprocdata[6][:] = []
            self.plotprocdata[7][:] = []
            self.plotprocdata[8][:] = []
            # self.plotprocdata[9][:] = []
            # self.plotprocdata[10][:] = []
            # Create and extend indices for generation plotting
            self.plotprocdata[0].extend([self.generation]*self.popsize)
            # Extend fitness values for generation plotting

            for i in range(self.popsize):
                self.plotprocdata[1].append(self.population[i].fitness)
                if self.population[i].inherited and self.population[i].fitness > self.population[i].mutatedFrom and self.population[i].fitness > self.population[i].inheritedFrom:
                    self.plotprocdata[6].append(self.population[i].inheritedFrom)
                    self.plotprocdata[7].append(self.population[i].mutatedFrom)
                    self.plotprocdata[8].append(self.population[i].fitness)
                    # self.plotprocdata[9].append(self.generation - 1)
                    # self.plotprocdata[10].append(self.generation)
                    self.population[i].inherited = False
                if self.population[i].mutated and self.population[i].fitness > self.population[i].mutatedFrom:
                    # self.plotprocdata[2].append(self.generation-1)
                    self.plotprocdata[3].append(self.population[i].mutatedFrom)
                    # self.plotprocdata[4].append(self.generation)
                    self.plotprocdata[5].append(self.population[i].fitness)
                    self.population[i].mutated = False
        else:
            return

    def sortFitness(self):
        # Sort population by objects' fitness
        self.population = sorted(self.population, key=lambda x: x.fitness)
        self.besthistory.append(self.population[-1].fitness)

    def naturalSelection(self):  # Roulette selection
        # Find maximum fitness value
        maxFitness = self.population[-1].fitness
        # Create mating pool in a Roulette selection way (proportional selection)
        self.matingPool = []
        for i in range(self.popsize):
            fitnessNormalized = (self.population[i].fitness / maxFitness)
            n = int(fitnessNormalized*100)
            self.matingPool.extend([self.population[i]]*n)

    def createNewGeneration(self):  # Heuristic crossover and mutation
        # Mate selected partners into the offspring, mutate obtained child
        # and insert into new generation by overwriting previous population

        # ratio = self.learningRate  # constant
        self.learningRate = 0.8 + 0.7*np.exp(-0.0035*self.generation)  # simulated annealing

        for i in range(self.popsize-self.elitecount):
            partner1 = random.choice(self.matingPool)
            partner2 = random.choice(self.matingPool)
            child = partner1.crossover(partner2, self.learningRate, self.mutationRate)
            self.population[i] = child
        if self.stochastic:
            for i in range(self.popsize - self.elitecount, self.popsize):
                self.population[i].fitfun(self.population[i], self.userdata)
        self.generation += 1
        self.generationInEpoch += 1


class DNA:
    def __init__(self, num, fitfun, learningRate, lcon, ucon, userdata, limMode):
        self.genes = []  # List of numeric values of genes of the individual
        self.fitness = 0  # Fitness value of the individual
        self.fitfun = fitfun  # Handle to fitness function
        self.mutated = False
        self.inherited = False
        self.mutatedFrom = 0
        self.inheritedFrom = 0
        self.userdata = userdata
        self.learningRate = learningRate
        self.lcon = lcon
        self.ucon = ucon
        self.limMode = limMode
        self.genes = np.random.uniform(self.lcon, self.ucon, num)
        # self.fitfun(self, self.userdata)

    # def mutate(self, mutationRate):
    #     # Possibly mutate one gene with the probability of mutationRate
    #     if random.random() < mutationRate:
    #         i = random.randint(0, len(self.genes)-1)
    #         modRange = self.ucon[i] - self.lcon[i]
    #         # self.genes[i] = self.genes[i] + np.random.uniform(-0.5, 0.5, 1) * modRange
    #         self.genes[i] = random.uniform(self.lcon[i], self.ucon[i])
    #         self.mutated = True
    #         self.checkConstraints()

    def mutate(self, mutationRate):
        # Possibly mutate every gene with the probability of mutationRate
        for i in range(len(self.genes)):
            if random.random() < mutationRate:
                self.genes[i] = random.uniform(self.lcon[i], self.ucon[i])
                self.mutated = True
                self.checkConstraints()
                # break

    def checkConstraints(self):
        if self.limMode is 'limit':
            for i in range(len(self.genes)):
                if self.genes[i] <= self.lcon[i]:
                    self.genes[i] = self.lcon[i]
                if self.genes[i] >= self.ucon[i]:
                    self.genes[i] = self.ucon[i]
        elif self.limMode is 'random':
            for i in range(len(self.genes)):
                if self.genes[i] <= self.lcon[i] or self.genes[i] >= self.ucon[i]:
                    self.genes[i] = random.uniform(self.lcon[i], self.ucon[i])
        elif self.limMode is 'none':
            pass

    def crossover(self, partner, ratio=1.5, mutationRate=1.1):
        # Create object for a child
        child = DNA(len(self.genes), self.fitfun, ratio,
                                       self.lcon, self.ucon, self.userdata, self.limMode)
        # Heuristic crossover - creation of actual child's genes
        if self.fitness > partner.fitness:  # I am better than partner
            # for i in range(len(self.genes)):
            child.genes=partner.genes+(random.uniform(-0.1, 0.1)+ratio)*(self.genes-partner.genes)
            child.mutatedFrom = self.fitness
            child.inheritedFrom = partner.fitness
        else:  # Partner is better than me
            # for i in range(len(self.genes)):
            child.genes = self.genes + (random.uniform(-0.1, 0.1) + ratio) * (partner.genes - self.genes)
            child.mutatedFrom = partner.fitness
            child.inheritedFrom = self.fitness
        child.checkConstraints()
        child.mutate(mutationRate)
        child.fitfun(child, child.userdata)
        if child.fitness > self.fitness and child.fitness > partner.fitness:
            child.inherited = True
        return child