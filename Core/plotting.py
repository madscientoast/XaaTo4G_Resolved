import awkward as ak
import numpy as np
import hist
import matplotlib.pyplot as plt
import uproot as up

def ConvertToROOT(masses,bins,range):
    low,high = range
    h = hist.Hist.new.Reg(bins, low, high, name="mass", label="Mass [GeV]").Double()
    h.fill(masses)

    # Plot it with matplotlib
    fig, ax = plt.subplots()
    h.plot(ax=ax)
    ax.set_title("4-Mass")
    ax.set_ylabel("Events")

    # Now take MPL to convert to ROOT
    values = h.values()         
    variances = h.variances()   
    edges = h.axes[0].edges    
    centers = h.axes[0].centers

    # Special bin handling that ROOT likes
    data  = np.zeros(len(values) + 2, dtype=np.float64)
    errs2 = np.zeros_like(data)
    data[1:-1]  = values
    errs2[1:-1] = variances

    # Required to convert MPL to ROOT
    fTsumw  = float(values.sum())                     # total sum of weights
    fTsumw2 = float(variances.sum())                  # total sum of w^2
    fTsumwx = float(np.sum(values * centers))         # sum w*x
    fTsumwx2= float(np.sum(values * centers**2))      # sum w*x^2
    fEntries = float(h.sum(flow=True))                # number of fills (with under/overflow)

    xaxis = up.writing.identify.to_TAxis(
        fName="xaxis",
        fTitle=h.axes[0].label,   # "Mass [GeV]"
        fNbins=len(values),
        fXmin=float(edges[0]),
        fXmax=float(edges[-1]),
        fXbins=np.asarray(edges, dtype=np.float64),
    )


    root_h = up.writing.identify.to_TH1x(
        fName=None,
        fTitle="mass spectrum",
        data=data,
        fEntries=fEntries,
        fTsumw=fTsumw,
        fTsumw2=fTsumw2,
        fTsumwx=fTsumwx,
        fTsumwx2=fTsumwx2,
        fSumw2=errs2,
        fXaxis=xaxis)
    return root_h