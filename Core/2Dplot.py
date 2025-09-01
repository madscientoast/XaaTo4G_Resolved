import awkward as ak
import numpy as np
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from Pairing import PairDeltaR, unique_pairing_perms
import hist
import matplotlib.pyplot as plt
import uproot as up
from plotting import ConvertToROOT

NanoAODSchema.warn_missing_crossrefs=False

# Load File(s) #
#fname = "../EHS_simulation/X300A15/result.root"
fname = "/project01/ndcms/rsnuggs/DataFiles/merged_output.root"
access_log = []
events = NanoEventsFactory.from_root(
    {fname : "Events"}, 
    schemaclass=NanoAODSchema,
    metadata={"dataset": "X300A15"},
    mode="virtual",
    access_log=access_log
).events()


# Define What to actually do here #
class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode="virtual"):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata['dataset']
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4] #Grab events with at least 4 photons

        #Apply PhotonID cut
        isEB = abs(all_photons.eta) < 1.4
        isEE = abs(all_photons.eta) >= 1.4

        MVA_cut = ((isEB & (all_photons.mvaID >= 0.0)) |      # barrel threshold
                (isEE & (all_photons.mvaID >= 0.0)))       # endcap threshold
        photons_wMVA = all_photons[MVA_cut]

        #apply electron veto 
        #eveto = all_photons.electronVeto[MVA_cut]
        #photons_wMVA = photons_wMVA[eveto]

        # Filter any events that have too few photons after MVA cut #
        photons_presel = photons_wMVA[ak.num(photons_wMVA) >= 4]

        # Now take the 4 leading photons from every event #
        photons = photons_presel[:,:4]
        #photons = all_photons[:,:4]
       
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

        Xmasses = ak.to_numpy(X.mass)
        Phimasses = ak.to_numpy((Phi0.mass + Phi1.mass)/2)
        
        h = hist.Hist.new.Reg(100, 0, 1000, name="Xmass", label="X Mass [GeV]").Reg(100,0,1000, name="Phimass", label="Phi Mass [GeV]").Double()

        # Fill the histogram
        h.fill(Xmasses,Phimasses)

        fig, axs = plt.subplots()
        hep.style.use("CMS")
        hep.cms.label(fontsize=10)
        pj = h.plot(ax=axs)
        pj.cbar.ax.tick_params(labelsize=8)             # tick labels
        pj.cbar.set_label("Entries", fontsize=9)        # colorbar label
        #cbar = fig.colorbar(pj[1],ax=axs)
        #fig.colorbar(fontsize=10)
        fig.savefig("2DTest.png", dpi=150)
        plt.close(fig)

        pass
    
    def postprocess(self, accumulator):
        pass

# Do things here #
p = MyProcessor("virtual")
out = p.process(events)