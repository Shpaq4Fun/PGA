from ga import *
from scipy import io, signal, stats
import numpy as np
from matplotlib import pyplot as plt
import datetime
from matplotlib import ticker

# def figFormat(fig=plt.gcf()):
#     def onResize(event):
#         plt.tight_layout()
#
#     fig.set_facecolor((0.1, 0.1, 0.1))
#     # fig.canvas.toolbar.pack_forget()
#     for ax in fig.axes:
#         ax.set_facecolor('none')
#         ax.autoscale(enable=True, axis='x', tight=True)
#         ax.grid(True)
#         ax.tick_params(axis="both", which="major", bottom=False, top=False, labelbottom=True, left=False,
#                        right=False, labelleft=True, colors='white', grid_alpha=0.2)
#
#     fig.tight_layout(pad=5)
#     cid = fig.canvas.mpl_connect('resize_event', onResize)

def fitfun1(self, targ=None):
    a = self.genes
    # b = np.flipud(a[1:])
    a = np.concatenate((np.flipud(a[1:]), a))

    y = signal.lfilter(a, 1, targ[0])
    self.fitness = 3 + stats.kurtosis(y)

def fitfun2(self, targ):
    x=np.copy(targ[9])
    a = self.genes
    # rnd = np.random.normal(0, 1, (targ[4], targ[0] - targ[7] - 1))
    rnd = targ[8][:, :targ[0]-targ[7]-1]
    for i in range(1, targ[0]-targ[7]):
        # a1 = ((a[1] / (targ[1][i]+5)) + a[2]) * (a[0] + x[:, i - 1]) * targ[2]
        a1 = (a[0] + (((a[1] / (targ[1][i] + 10)) + a[2]) * x[:, i - 1])) * targ[2]
        a2 = np.sqrt((a[3] * ((targ[1][i] + 10) ** a[4]) + a[5]) + a[6] * x[:, i - 1])
        a3 = (a2 * np.sqrt(targ[2]))
        # print(rnd.shape)
        rn = rnd[:,i-1]
        a3 = a3 * rn
        x[:, i] = x[:, i - 1] + a1 + a3

    # for i in range(1, targ[0]-targ[7]):
    #     a1 = (a[0] * (((a[1] * (targ[1][i] + 10)) ** a[2]) - a[3] * x[:, i - 1])) * targ[2]
    #     a2 = a[4]*np.sqrt(a[1] * ((targ[1][i] + 10) ** a[2])* x[:, i - 1]) * np.sqrt(targ[2]) * rnd[:,i-1]
    #     x[:, i] = x[:, i - 1] + a1 + a2

    ind = np.argwhere(np.isnan(np.mean(x, axis=1)))
    if ind.any():
        # x[ind,:] = np.ones([1, targ[0] - targ[7]])* -100
        self.genes = []
        self.genes = np.random.uniform(self.lcon, self.ucon, len(a))
        self.fitness = np.finfo(float).eps
        return

    # sf = 1/(np.sum(np.abs(np.matlib.repmat(targ[3][0:-targ[7]], targ[4], 1) - x))/targ[4])

    # Median + MAE
    rp = np.matlib.repmat(targ[3][0:-targ[7]], targ[4], 1)
    scale = (targ[0] - targ[7])
    mat = np.median(rp - x, axis=0)
    mat2 = (rp - x)
    sm = 19*np.sum(np.abs(mat)) + (np.sum(np.abs(mat2)) / targ[4])
    sf = 1 / (sm / scale)

    # MAE
    # rp = np.matlib.repmat(targ[3][0:-targ[7]], targ[4], 1)
    # scale = (targ[4] * (targ[0] - targ[7]))
    # mat = (rp - x) / targ[0]
    # sm = np.sum(np.abs(mat))
    # sf = 1 / (sm / scale)

    # MISE
    # rp = np.matlib.repmat(targ[3][0:-targ[7]], targ[4], 1)
    # scale = (targ[4]*(targ[0]-targ[7]))
    # mat = (rp - x)/(np.sum(np.abs(np.diff(rp)))*((targ[0]-targ[7])/(targ[0]-targ[7]-1)))
    # sm = np.sum(np.abs(mat))
    # sf = 1/(sm/scale)

    if np.isnan(sf):# or len(ind) is targ[4]:
        print("NaN")
        self.genes = []
        self.genes = np.random.uniform(self.lcon, self.ucon, len(a))
        # self.fitfun(self, self.userdata)
        self.fitness = np.finfo(float).eps
    else:
        self.fitness = sf

def fitfun3(self, targ=None):
    x = np.array(targ[0])
    if len(x) is 0:
        self.fitness = np.finfo(float).eps
        return
    a = self.genes
    res = []
    for j in range(len(x)):
        y = 0
        for i in range(len(a)):
            y += a[i] * np.power(x[j], i)
        res.append(y)
    b = [abs(res[i] - targ[1][i]) for i in range(len(x))]
    self.fitness = 1/(np.mean(b))
    if np.isnan(self.fitness):
        print('NaN')
        self.fitness = 0

def customPlot1(self, ax):
    # print(len(self.besthistory))
    if self.generation > 1:
        if self.besthistory[-1] == self.besthistory[-2]:
            return
    a = self.population[-1].genes
    a = np.concatenate((np.flipud(a[1:]), a))
    # w, h = signal.freqz([float(i) / sum(np.abs(self.population[-1].genes)) for i in self.population[-1].genes], 1)
    w, h = signal.freqz(a/sum(abs(a)), 1)
    if self.generation is 0:
        self.plothandles[3] = ax.plot(0.5 * self.userdata[1] * w / np.pi, 20 * np.log10(abs(h)))
        ax.set_title('Designed filter', color='white', fontweight="bold")
        ax.set_ylabel('Amplitude [dB]', color='white')
        ax.set_xlabel('Frequency [Hz]', color='white')
        ax.set_xlim(0, 0.5 * self.userdata[1] * w[-1] / np.pi)
    else:
        self.plothandles[3][0].set_ydata(20 * np.log10(abs(h)))
        # self.plothandles[0].axes[self.nAx].set_ylim(np.nanmin(20 * np.log10(abs(h)))*1.05, 0)
        # self.plothandles[0].axes[self.nAx].set_title('Designed filter of length ' + str(len(self.population[0].genes)*2-1))
        ax.set_ylim(np.nanmin(20 * np.log10(abs(h))) * 1.05, 0)
        ax.set_title('Designed filter of length ' + str(len(self.population[0].genes) * 2 - 1))


def customPlot2(self, ax):
    # ax.clear()
    targ = self.userdata
    offset = targ[11]
    #     offset = targ[7]
    mplot = 50
    x = np.concatenate(
        (np.expand_dims(np.array([targ[3][1-divmod(targ[0], offset)[1]]] * mplot), axis=1),
         np.zeros((mplot, offset - 1))),
        axis=1)
    a = np.array(self.population[-1].genes)
    print(list(self.population[-1].genes))
    # rnd = np.random.normal(0, 1, (mplot, targ[0]-targ[7]-1))

    rnd = targ[8][:mplot, :targ[0]-divmod(targ[0], offset)[1]-1]
    # breaks = np.array([1, 6456-6197, 6714-6197, 6973-6197, targ[0]-1])
    breaks = np.array([1, targ[0]-targ[7], targ[0]-1])

    for i in range(1, offset):
        if any(i == breaks):
            x[:, i] = np.array([targ[3][i]] * mplot)
            continue
        # a1 = ((a[1] / (targ[1][-1 - offset + i] + 5)) + a[2]) * (a[0] + x[:, i - 1]) * targ[2]
        a1 = (a[0] + ((a[1] / (targ[1][-1 - offset + i]+10) + a[2]) * x[:, i - 1])) * targ[2]
        a2 = np.sqrt((a[3] * ((targ[1][-1 - offset + i]+10) ** a[4]) + a[5]) + a[6] * x[:, i - 1])
        rn = rnd[:, i-1]
        a3 = (a2 * np.sqrt(targ[2]))
        a3 = a3 * rn
        x[:, i] = x[:, i - 1] + a1 + a3

    # for i in range(1, offset):
    #     a1 = (a[0]*(((a[1]*(targ[1][-1-offset+i] + 10)) ** a[2]) - a[3] * x[:, i - 1])) * targ[2]
    #     a2 = a[4]*np.sqrt(a[1] * ((targ[1][-1-offset+i]+10)**a[2])* x[:,i-1]) * np.sqrt(targ[2]) * rnd[:,i-1]
    #     x[:, i] = x[:, i - 1] + a1 + a2

    ind = np.argwhere(np.isnan(np.mean(x, axis=1)))
    if ind.any():
        x[ind, :] = np.ones([1, offset])* -100#np.nanmin(x)
    estimate = np.median(x, axis=0)
    p5 = np.transpose(np.percentile(x, [5, 95], 0))
    p25 = np.transpose(np.percentile(x, [25, 75], 0))

    overs = np.zeros((len(breaks)-1, 2))
    for k in range(1, len(breaks)):
        ranges = range(breaks[k-1],breaks[k]-1)
        p5over = np.sum(targ[3][ranges] > p5[ranges, 1]) + np.sum(targ[3][ranges] < p5[ranges, 0])
        p25over = np.sum(targ[3][ranges] > p25[ranges, 1]) + np.sum(targ[3][ranges] < p25[ranges, 0])
        p5over = 100*p5over / (len(ranges))
        p25over = 100*(len(ranges)-p25over) / (len(ranges))
        overs[k - 1, 0] = p5over
        overs[k - 1, 1] = p25over

    if divmod(self.generation, 20)[1] is 0:
        self.userdata[8] = np.random.normal(0, 1, (targ[4], targ[0] - 1))

    def f(x):
        return datetime.datetime.fromordinal(np.int(x))
    f2 = np.vectorize(f)
    # breaks = breaks + 6197
    # xx = f2(targ[10][int(np.mean([breaks[0:1]]))])
    def onclick(event):
        input("Paused...")
    if self.generation is 0:
        # t=num2date(targ[10][-1 - offset:-1])
        self.plothandles[0].canvas.mpl_connect('button_press_event', onclick)
        t = f2(targ[10][-1 - offset:-1])
        self.plothandles[3] = ax.plot(t, np.transpose(x), c='white', alpha=0.2, lw=0.5)
        self.plothandles[4] = ax.plot(t, p25, c='lime', alpha=0.8, lw=0.5)
        self.plothandles[5] = ax.plot(t, p5, c='red', alpha=0.8, lw=1)
        self.plothandles[6] = ax.plot(t, np.transpose(estimate), c='blue', alpha=0.8, lw=1)

        ax.plot(f2(targ[10]), targ[3], c='white', alpha=1, lw=1)
        self.plothandles[7] = ax.plot([t[(targ[0]-targ[7])]]*2, [np.nanmin([np.nanmin(targ[3]), np.nanmin(x)]), np.nanmax([np.nanmax(targ[3]), np.nanmax(x)])], c='white', lw=1, alpha=1)
        ax.set_title(round(1/self.population[-1].fitness, 2), color='white') #, round(p5over, 1), round(p25over, 1), round(p5over1, 1), round(p25over1, 1)], color='white')
        self.plothandles[8] = [0,0,0,0]
        # aa=(str(overs[k - 1, 0]) + '\n' + str(overs[k - 1, 1]))
        for k in range(1, len(breaks)):
            xx=f(targ[10][int(breaks[k-1]+(breaks[k]-breaks[k-1])/2)])
            self.plothandles[8][k-1] = plt.text((xx),0,(str(round(overs[k-1,0],2))+'\n'+str(round(overs[k - 1, 1],2))),
                                                size=15, color='white', horizontalalignment='center')

    else:
        [self.plothandles[3][i].set_ydata(np.transpose(x[i, :])) for i in range(mplot)]
        self.plothandles[4][0].set_ydata(np.transpose(np.percentile(x, 25, 0)))
        self.plothandles[4][1].set_ydata(np.transpose(np.percentile(x, 75, 0)))
        self.plothandles[5][0].set_ydata(np.transpose(np.percentile(x, 5, 0)))
        self.plothandles[5][1].set_ydata(np.transpose(np.percentile(x, 95, 0)))
        self.plothandles[6][0].set_ydata(np.transpose(estimate))
        self.plothandles[7][0].set_ydata(np.transpose([np.nanmin([np.nanmin(targ[3]), np.nanmin(x)]), np.nanmax([np.nanmax(targ[3]), np.nanmax(x)])]))
        ax.set_title(round(1/self.population[-1].fitness, 2)) #, round(p5over, 1), round(p25over, 1), round(p5over1, 1), round(p25over1, 1)])
        ax.set_ylim(np.nanmin([np.nanmin(targ[3]), np.nanmin(x)]), np.nanmax([np.nanmax(targ[3]), np.nanmax(x)]))
        ll=ax.get_ylim()
        for k in range(1, len(breaks)):
            yy=ll[0]+0.8*(ll[1]-ll[0])
            self.plothandles[8][k - 1].set_y(yy)
            self.plothandles[8][k - 1].set_text(str(round(overs[k - 1, 0],2)) + '\n' + str(round(overs[k - 1, 1])))
        # self.plothandles[0].axes[3].set_xlim(num2date(targ[10][-1 - offset]), num2date(targ[10][-1]))
        # self.plothandles[0].axes[3].set_xlim(0, len(targ[1]) * targ[2])

def customPlot3(self, ax):
    def onclick(event):
        # ix, iy = event.xdata, event.ydata
        self.userdata[0].append(event.xdata)
        self.userdata[1].append(event.ydata)
        self.plothandles[0].canvas.mpl_disconnect(self.userdata[2][0])
        self.userdata[2] = []

    if self.generation is 0:
        self.userdata[2].append(self.plothandles[0].canvas.mpl_connect('button_press_event', onclick))
        self.plothandles[0].axes[3].set_ylim(-1, 1)
        self.plothandles[0].axes[3].set_xlim(-1, 1)
    if len(self.userdata[0]) is 0:
        # self.plothandles[0].canvas.mpl_disconnect(self.userdata[2][0])
        # print(len(self.userdata[2]))
        return
    a = self.population[-1].genes
    x = np.linspace(-1, 1, 30)
    y = np.zeros_like(x)
    for i in range(len(a)):
        y += a[i] * np.power(x, i)

    # if self.generation is 0:
        # print(len(self.userdata[2]))
        # self.userdata[2].append(self.plothandles[0].canvas.mpl_connect('button_press_event', onclick))
        # print(len(self.userdata[2]))
        # self.plothandles[3] = ax.scatter(0, 0, alpha=1, marker='.', s=25, c='white')
        # self.plothandles[4] = ax.plot([0,0], [0,0], c='white', alpha=1, lw=1)
    # else:
    if len(self.userdata[2]) is 0:
        self.userdata[2].append(self.plothandles[0].canvas.mpl_connect('button_press_event', onclick))
    if len(self.userdata[0]) is not 0 and self.plothandles[3] is 0:
        self.plothandles[3] = ax.scatter(self.userdata[0], self.userdata[1], alpha=1, marker='.', s=25, c='white')
        self.plothandles[4] = ax.plot(x, y, c='white', alpha=1, lw=1)
    else:
        # print(self.userdata[0], self.userdata[1])
        self.plothandles[3].set_offsets(np.c_[self.userdata[0], self.userdata[1]])
        self.plothandles[4][0].set_ydata(y)
        # self.plothandles[0].axes[3].set_ylim(-1, 1)
        # self.plothandles[0].axes[3].set_xlim(-1, 1)
        ax.set_title(a, color='white')


def data1():  # lozysko_b
    ff = io.loadmat('data/lozysko_b.mat')
    feed = ff.get('data')
    feed = np.array(feed)
    feed = np.transpose(feed).squeeze()
    fs = 19200
    nfft = 256
    dl = 4
    lcon = [-1] * dl
    ucon = [1] * dl
    mr = 0.004
    psize = 100
    ec = 4
    lr = 1.1

    epochs = 15
    cp = [customPlot1]
    pi = 1
    feed = [feed,fs]
    fitfun = fitfun1
    st = False
    limmode = 'random'
    ltc = 'quant'
    return feed, fs, nfft, np.array(lcon), np.array(ucon), mr, psize, ec, lr, dl, epochs, cp, pi, \
           fitfun, st, limmode, ltc

def data2():
    # feed = np.genfromtxt("data/CADUSD.txt", unpack=True, usecols=1)
    # feed = np.genfromtxt("data/USDPLNFIX.txt", unpack=True, usecols=1)
    ff = io.loadmat('data/USDCAD7k.mat')
    feed = ff.get('feed')
    feed = np.array(feed)
    feed = np.transpose(feed).squeeze()
    feed = feed[6197:]
    n = len(feed)
    m = 100
    h = 1.0 / 251
    tdt = ff.get('t')
    tdt = np.array(tdt[6197:]).transpose().squeeze()-366
     #+ datetime.timedelta(days=tdt % 1)

    t = np.arange(n)
    t = np.transpose(np.multiply(t, h))
    feed = np.log(feed)

    # cadusd
    # lcon = [0] * 7
    # ucon = [1] * 7
    # theta = 6.8359 * 10**-4
    # mu = 0.2254
    # lcon = [np.finfo(float).eps, 0, 0, np.finfo(float).eps, -2, 0, np.finfo(float).eps]
    # ucon = [2*mu, 3*theta, np.finfo(float).eps, 0.06, 2, np.finfo(float).eps, 0.06]
    # lcon = [0, 0, -4, 0, 0]
    # ucon = [3 * mu, 3 * mu, 0, 3 * mu, 3 * mu]

    # lcon = [0.055254, 1.22*(10**-12), 0, 0.5*h, 2.22*(10**-12), 2.22*(10**-12), (h-5*h**2)]
    # ucon = [0.2254, h, 10, 10*h, h, h, h+5*h**2]

    lcon = [0, 0, 0, 0, -6, 0, 0]
    ucon = [6] * 7

    # usdpln
    # start = [0.0324, 2.22*(10**-12), 0.0015, 2.22*(10**-12), h, 2.22*(10**-12), h]
    # lcon = [0.0424, 1.22 * (10 ** -12), 1.22 * (10 ** -12), 1.22 * (10 ** -12), h - h ** 2, 1.22 * (10 ** -12), h - h ** 2]
    # ucon = [0.0424, h, 1, h, 2*h + h ** 2, h, 2*h + h ** 2]
    # lcon = [0.0324, 0, 0.0029, 0.0012, 0.0040, 0.0012, 0.0040];
    # ucon = [0.0524, 0.0023, 0.0121, 0.0023, 0.0064, 0.0024, 0.0065];

    offset = 7101-6973  # ostatni rok
    poffset = n-1
    rnd = np.random.normal(0, 1, (m, n - 1))
    x = np.concatenate((np.expand_dims(np.array([feed[0]] * m), axis=1), np.zeros((m, n - 1 - offset))), axis=1)
    userdata = [n, t, h, feed, m, lcon, ucon, offset, rnd, x, tdt, poffset]
    mr = 0.05
    psize = 80
    ec = 3
    lr = 1.5
    dl = len(lcon)
    epochs = 1
    cp = [customPlot2]
    pi = 1
    fitfun = fitfun2
    st = True
    limmode = 'random'
    ltc = 'inf'
    return userdata, np.array(lcon), np.array(ucon), mr, psize, ec, lr, dl, epochs, cp, pi, fitfun, st, limmode, ltc

def data3():  # regresja
    dl = 2
    lcon = [-10] * dl
    ucon = [10] * dl
    mr = 0.08
    psize = 30
    ec = 3
    lr = 0.8
    feed = [[], [], []]
    epochs = 1
    cp = [customPlot3]
    pi = 1
    fitfun = fitfun3
    st = True
    limmode = 'none'
    ltc = 'inf'
    return feed, np.array(lcon), np.array(ucon), mr, psize, ec, lr, dl, epochs, cp, pi, fitfun, st, limmode, ltc

feed, fs, nfft, lcon, ucon, mr, psize, ec, lr, dl, epochs, cp, pi, fitfun, st, limmode, ltc = data1()
# feed, lcon, ucon, mr, psize, ec, lr, dl, epochs, cp, pi, fitfun, st, limmode, ltc = data2()
# feed, lcon, ucon, mr, psize, ec, lr, dl, epochs, cp, pi, fitfun, st, limmode, ltc = data3()

instance = GA(fitfun, dnaLength=dl, mutationRate=mr, populationSize=psize,
              userdata=feed, elitecount=ec, plotting='fast', ltc=ltc,
              lcon=lcon, ucon=ucon, learningRate=lr, epochs=epochs, customPlot=cp,
              prl=False, progIncrement=pi, stochastic=st, limMode=limmode)
print(instance.population[-1].genes)
# 
# N = feed.size
# time = np.arange(N) / float(fs)
# a = instance.population[-1].genes
# # aa = instance.population[-1].genes
# # aa.extend([0, 0])
#
# list(reversed(a[1:])).extend(a)
# # list(reversed(aa[1:])).extend(aa)
#
# w, h = signal.freqz(a, 1)
# # ww, hh = signal.freqz(aa, 1)
#
# filtered = signal.lfilter(a, 1, feed)
# fig = plt.figure(2)
# # plt.subplot(3, 1, 1)
# # plt.subplot(3, 1, 2)
# # plt.subplot(3, 1, 3)
#
# plt.subplot(3, 1, 1)
# plt.plot(time, feed, lw=0.5)
# # plt.grid()
# plt.title('Original signal', color='white', fontweight="bold")
# plt.ylabel('Amplitude [m/s^2]', color='white')
# plt.xlabel('Time [s]', color='white')
#
# plt.subplot(3, 1, 2)
# plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)))
# # plt.grid()
# plt.title('Designed filter', color='white', fontweight="bold")
# plt.ylabel('Amplitude [dB]', color='white')
# plt.xlabel('Frequency [Hz]', color='white')
#
# plt.subplot(3, 1, 3)
# # plt.plot(0.5 * fs * ww / np.pi, 20 * np.log10(abs(hh)))
# plt.plot(time, filtered, lw=0.5)
# # plt.grid()
# plt.title('Filtered signal', color='white', fontweight="bold")
# plt.title('Filtered signal, kurtosis = ' + repr(round(3+stats.kurtosis(filtered), 2)), color='white', fontweight="bold")
# plt.ylabel('Amplitude [m/s^2]', color='white')
# plt.xlabel('Time [s]', color='white')
#
# figFormat(fig)
plt.show()
