import awkward as ak
import numpy as np
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from Pairing import PairDeltaR, unique_pairing_perms
import hist
import matplotlib.pyplot as plt
import uproot as up
from scipy.stats import crystalball
from scipy.optimize import curve_fit

NanoAODSchema.warn_missing_crossrefs=False

# Load File(s) #
fname = "../EHS_simulation/X300A75/result.root"
#fname = "/project01/ndcms/rsnuggs/DataFiles/merged_output.root"
access_log = []
events = NanoEventsFactory.from_root(
    {fname : "Events"}, 
    schemaclass=NanoAODSchema,
    metadata={"dataset": "X300A75"},
    mode="virtual",
    access_log=access_log
).events()


def cb_model(x, beta, m, loc, scale, norm):
    return norm * crystalball.pdf(x, beta, m, loc=loc, scale=scale)

# Define What to actually do here #
class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode="virtual"):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata['dataset']
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4 & events.HLT.TriplePhoton_35_35_5_CaloIdLV2_R9IdVL] #Grab events with at least 4 photons that pass trigger

        #Apply PhotonID cut
        isEB = abs(all_photons.eta) < 1.4
        isEE = abs(all_photons.eta) >= 1.4

        MVA_cut = ((isEB & (all_photons.mvaID >= 0.0)) |      # barrel threshold
                (isEE & (all_photons.mvaID >= 0.0)))       # endcap threshold
        photons_wMVA = all_photons[MVA_cut]

        # Filter any events that have too few photons after MVA cut #
        photons_presel = photons_wMVA[ak.num(photons_wMVA) >= 4]

        # Now take the 4 leading photons from every event #
        photons = photons_presel[:,:4]
       
        # Evaluate all possible DeltaR and find best combination index for each event
        best_idx = PairDeltaR(photons)
        
        # Now to pick best combo
        combos = np.array(unique_pairing_perms(4))
        perm_idx = ak.Array(combos)[best_idx]
        photons = photons[perm_idx]

        # Now put together parent particles
        Phi0 = photons[:,0] + photons[:,1]
        Phi1 = photons[:,2] + photons[:,3]
        X = Phi0 + Phi1

        masses = ak.to_numpy(X.mass)

        counts, bins = np.histogram(masses, bins=100,range=(0,1000),density=True)
        centers = 0.5*(bins[1:] + bins[:-1])
        p0 = [1.5, 3.0, np.mean(masses), np.std(masses), 1.0]  # initial guess

        popt, pcov = curve_fit(cb_model, centers, counts,p0)

        h = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()
        model_hist = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()
        pull_hist = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()

        # Fill the histogram
        h.fill(masses)
        
        bin_widths = np.diff(h.axes[0].edges)   # array of bin widths
        norm_factor = (h.values() * bin_widths).sum()
        h /= norm_factor
        xx = np.linspace(0, 1000.0, 500)
        model_hist.fill(cb_model(xx,*popt))
        
        # Work with arrays, not Hist objects directly
        data_vals   = h.values()
        model_vals  = model_hist.values()
        errors      = np.sqrt(data_vals)    # safer than sqrt(h.values())

        # Avoid division by zero
        mask = errors > 0
        pulls = np.zeros_like(data_vals)
        pulls[mask] = (data_vals[mask] - model_vals[mask]) / errors[mask]
        pull_hist.fill(pulls)
        

        # --- Plot
        fig = plt.figure(figsize=(6.0, 5.5))
        gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[2,1])
        ax_top, ax_bot = gs.subplots(sharex=True)
        # Top: data density and fitted PDF
        hep.style.use("CMS")
        h.plot(ax=ax_top, histtype="step", label="Data (density)")
        xx = np.linspace(0.0, 1000.0, 600)
        xxx = np.linspace(0.0,1000.0,100)
        ax_top.plot(xx, cb_model(xx, *popt), lw=1.6, label="CrystalBall fit")
        #ax_top.legend(frameon=False)
        ax_top.set_xlabel("")
        ax_top.set_ylabel("Normalization")
        hep.cms.label(ax=ax_top, data=False, fontsize=11)

        # Bottom: pulls
        #pull_hist.plot(ax=ax_bot, histtype="step")
        ax_bot.semilogy(xxx,pulls)
        ax_bot.axhline(0.0, lw=1)
        ax_bot.set_ylabel("Pull")
        ax_bot.set_xlabel("Mass [GeV]")
        ax_bot.set_ylim([-1.0, 1.0])
        fig.savefig("test.png", dpi=150)
        plt.close(fig)

        pass
    
    def postprocess(self, accumulator):
        pass

# Do things here #
p = MyProcessor("virtual")
out = p.process(events)