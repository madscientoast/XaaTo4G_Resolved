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
#fname = "../EHS_simulation/X300A30/result.root"

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

        '''
        #Apply PhotonID cut
        isEB = abs(all_photons.eta) < 1.4
        isEE = abs(all_photons.eta) >= 1.4

        MVA_cut = ((isEB & (all_photons.mvaID >= 0.0)) |      # barrel threshold
                (isEE & (all_photons.mvaID >= 0.0)))       # endcap threshold
        photons_wMVA = all_photons[MVA_cut]

        # Filter any events that have too few photons after MVA cut #
        photons_presel = photons_wMVA[ak.num(photons_wMVA) >= 4]

        # Now take the 4 leading photons from every event #
        photons = photons_presel[:,:4]'''
        photons = all_photons[:,:4]
       
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

        h = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()

        # Fill the histogram
        h.fill(mass=masses)

        # Now do for trigger 
        trig = events.HLT.TriplePhoton_35_35_5_CaloIdLV2_R9IdVL[ak.num(events.Photon.pt) >= 4]
        all_photons = all_photons[trig]
        photons = all_photons[:,:4]
       
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
        h_trig = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()

        # Fill the histogram
        h_trig.fill(mass=masses)

        # Now for PhotonID
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4] #Grab events with at least 4 photons

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

        h_MVA = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()

        # Fill the histogram
        h_MVA.fill(mass=masses)

        # Now for all selections
        # Now for PhotonID
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4] #Grab events with at least 4 photons
        all_photons = all_photons[trig]

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

        h_presel = hist.Hist.new.Reg(100, 0, 1000, name="mass", label="Mass [GeV]").Double()

        # Fill the histogram
        h_presel.fill(mass=masses)

        
        # Compare No presel vs. Trigger
        fig, ax = plt.subplots()
        hep.style.use("CMS")
        hep.cms.label("Preliminary", lumi=1.1, year=2017)
        h.plot(ax=ax,label='No selections', color='blue')
        h_trig.plot(ax=ax,label='With Trigger', color='red')
        ax.set_title("4-Mass")
        ax.set_ylabel("Events")
        fig.legend()
        fig.savefig("TriggerVsNone.png", dpi=150)
        plt.close(fig)

        # Compare No presel vs. PhotonID
        fig, ax = plt.subplots()
        hep.style.use("CMS")
        hep.cms.label("Preliminary", lumi=1.1, year=2017)
        h.plot(ax=ax,label='No selections', color='blue')
        h_MVA.plot(ax=ax,label='With PhotonID', color='red')
        ax.set_title("4-Mass")
        ax.set_ylabel("Events")
        fig.legend()
        fig.savefig("PhotonIDVsNone.png", dpi=150)
        plt.close(fig)

        # Compare No presel vs. PhotonID
        fig, ax = plt.subplots()
        hep.style.use("CMS")
        hep.cms.label("Preliminary",data=True,lumi=1.1, year=2018)
        h.plot(ax=ax,label='No selections', color='blue')
        h_trig.plot(ax=ax,label='With Trigger', color='red')
        h_presel.plot(ax=ax,label='With Trigger and PhotonID', color='orange')
        #ax.set_title("4-Mass")
        ax.set_ylabel("Events")
        plt.legend()
        fig.savefig("CompareAll.png", dpi=150)
        plt.close(fig)

        pass
    
    def postprocess(self, accumulator):
        pass

# Do things here #
p = MyProcessor("virtual")
out = p.process(events)