import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import hist
from numba import njit

def SetAnalysisWindow(dataset,PhiM):
    if dataset == "X300A15":
        return (PhiM > 7.0) & (PhiM < 23.0)
    if dataset == "X300A30":
        return (PhiM > 15.0) & (PhiM < 45.0)
    if dataset == "X300A75":
        return (PhiM > 62.0) & (PhiM < 88.0)
    else:
        return (PhiM > 7.0) & (PhiM < 88.0)
    
def ExtractParams(dataset):
    return [dataset["mass_asym"],dataset["DeltaR1"],dataset["DeltaR2"],dataset["DeltaEta"],dataset["mass"]]


@njit(parallel=True, fastmath=True)
def IterateCuts(cuts,dataset):
    X_wCuts = []
    for cut in cuts:
        define_cut = (dataset[0] < cut[0])  & (dataset[1] < cut[1]) & (dataset[2] < cut[2]) & (dataset[3] < cut[3])
        cut_X = dataset[4][define_cut]
        X_wCuts.append(cut_X)

    return X_wCuts

def CalcSvsB(signal,background):
    SvsB = []
    for i in range(len(background)):
        h1 = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()
        h1.fill(mass=signal[i])

        h2 = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()
        h2.fill(mass=background[i])

        s, edges = h1.to_numpy()
        b,edges = h2.to_numpy()
        s = 0.01*s
        denom=np.sqrt(s+b)
        ratio = np.divide(s,  denom, out=np.zeros_like(s, dtype=float), where=denom!=0)
        sig = np.sum(ratio)
        SvsB.append(sig)
    return SvsB

def GetCut(cuts,cut_var):
    lst = []
    for cut in cuts:
        if cut_var == "MA":
            lst.append(cut[0])
        #if cut_var == "#DeltaR_1":
        if cut_var == r'$\Delta R_1$':
            lst.append(cut[1])
        #if cut_var == "#DeltaR_2":
        if cut_var == r'$\Delta R_2$':
            lst.append(cut[2])
        #if cut_var == "#Delta#eta":
        if cut_var == r'$\Delta\eta$':
            lst.append(cut[3])
    return lst

#Used in setting variables to RGB values
def normalize(lst):
    min_val, max_val = min(lst), max(lst)
    den = max_val - min_val
    if den == 0:
        return [0.0 for _ in lst]
    return [(x - min_val) / den for x in lst]


def PlotCutVsSum(cut,res,mass,cuts):

    if(cut == "MA"):
        colors = [(r, g, b) for r, g, b in zip(np.ones(len(cuts)) - normalize(GetCut(cuts,r'$\Delta R_1$')),np.ones(len(cuts)) -  normalize(GetCut(cuts,r'$\Delta R_2$')),np.ones(len(cuts)) -  normalize(GetCut(cuts,r'$\Delta\eta$')))]
    if(cut == r'$\Delta R_1$'):
        colors = [(r, g, b) for r, g, b in zip(np.ones(len(cuts)) - normalize(GetCut(cuts,"MA")),np.ones(len(cuts)) -  normalize(GetCut(cuts,r'$\Delta R_2$')),np.ones(len(cuts)) -  normalize(GetCut(cuts,r'$\Delta\eta$')))]
    if(cut == r'$\Delta R_2$'):
        colors = [(r, g, b) for r, g, b in zip(np.ones(len(cuts)) - normalize(GetCut(cuts,"MA")), np.ones(len(cuts)) - normalize(GetCut(cuts,r'$\Delta R_1$')), np.ones(len(cuts)) - normalize(GetCut(cuts,r'$\Delta\eta$')))]
    if(cut == r'$\Delta\eta$'):
        colors = [(r, g, b) for r, g, b in zip(np.ones(len(cuts)) - normalize(GetCut(cuts,"MA")), np.ones(len(cuts)) - normalize(GetCut(cuts,r'$\Delta R_1$')), np.ones(len(cuts)) - normalize(GetCut(cuts,r'$\Delta R_2$')))]

    # Create a scatter plot using z_vals for color mapping
    plt.scatter(GetCut(cuts,cut), res, c=colors, s=50)

    # Add color bar
    #plt.colorbar(label='Z Value (Color)')

    # Set axis labels and title
    title = "Cut vs Sum [" + cut + "]"
    plt.xlabel(cut)
    plt.ylabel(r'Sum of $S/\sqrt{S+B}$')
    #plt.ylabel("Sum of S/root(S+B)")
    plt.title(title)
    fname = cut + "_" + mass + ".png"
    plt.savefig(fname, dpi=300)
    plt.clf()
