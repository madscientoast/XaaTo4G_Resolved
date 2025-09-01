import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from Pairing import PairDeltaR, unique_pairing_perms
#from CutFunctions import SetAnalysisWindow, GetCut, PlotCutVsSum, normalize,IterateCuts,ExtractParams
from CutFunctions import *
import hist
import matplotlib.pyplot as plt


NanoAODSchema.warn_missing_crossrefs=False

# Define File(s) #
fileset = {
    "Background": {
        "files": {
            "/project01/ndcms/rsnuggs/DataFiles/merged_output.root" : "Events"
            #"/project01/ndcms/rsnuggs/DataFiles/EGamma_Run2018D-UL2018_MiniAODv2_NanoAODv9-v3_1.root" : "Events"
        },
        "metadata": {
            "is_mc": False
        }
    },
    "X300A15": {
        "files": {
            "../EHS_simulation/X300A15/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
    "X300A30": {
        "files": {
            "../EHS_simulation/X300A30/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
    "X300A75": {
        "files": {
            "../EHS_simulation/X300A75/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
}

# Define Cuts
ma_list  = np.linspace(0.1, 1.0, 10, dtype=np.float64)
dr_list  = np.linspace(1.0, 3.9, 30, dtype=np.float64)
eta_list = dr_list.copy()
ma, dr1, dr2, eta = np.meshgrid(ma_list, dr_list, dr_list, eta_list, indexing="ij")
grid = np.stack([ma.ravel(), dr1.ravel(), dr2.ravel(), eta.ravel()], axis=1)
cuts = np.ascontiguousarray(grid, dtype=np.float64)

# Define What to actually do here #
class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode="virtual"):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata['dataset']
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4] #Grab events with at least 4 photons
    
        #Apply PhotonID cut
        endcap_MVA_filter = (all_photons.eta >= 1.4) & (all_photons.mvaID >= 0.14)
        barrel_MVA_filter = (all_photons.eta < 1.4) & (all_photons.mvaID >= 0.27)
        MVA_cut = ak.any([endcap_MVA_filter, barrel_MVA_filter], axis=0)
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
        avgPhi = (Phi0.mass + Phi1.mass)/2
        X = Phi0 + Phi1

        # Apply mass window to all particles #
        MassWindow = SetAnalysisWindow(dataset,avgPhi)
        X = X[MassWindow]
        Phi0 = Phi0[MassWindow]
        Phi1 = Phi1[MassWindow]
        photons = photons[MassWindow]

        # Implement Cut(s)
        mass_asym = abs(Phi0.mass - Phi1.mass)/(Phi0.mass + Phi1.mass)
        DeltaR1 = photons[:,0].deltaR(photons[:,1])
        DeltaR2 = photons[:,2].deltaR(photons[:,3])
        DeltaEta = abs(Phi0.eta - Phi1.eta)

        masses = ak.to_numpy(X.mass)

        return {
            dataset: {
                "entries": len(events),
                "mass": masses,
                "mass_asym": ak.to_numpy(mass_asym),
                "DeltaR1": ak.to_numpy(DeltaR1),
                "DeltaR2": ak.to_numpy(DeltaR2),
                "DeltaEta": ak.to_numpy(DeltaEta)
            }
        }
    
    def postprocess(self, accumulator):
        pass

# Do things here #
iterative_run = processor.Runner(
    executor = processor.FuturesExecutor(workers=4, compression=None),
    chunksize=100000000,
    schema=NanoAODSchema,
    savemetrics=True,
)

print("Processing Files...")
out, metrics = iterative_run(
    fileset,
    processor_instance=MyProcessor("virtual")
)
print("Processing Done!")

print("Starting Cuts")
X_SigX300A15 = IterateCuts(cuts,np.ma.getdata(ExtractParams(out["X300A15"])))
print("X300A15 Done!")
X_SigX300A30 = IterateCuts(cuts,np.ma.getdata(ExtractParams(out["X300A30"])))
print("X300A30 Done!")
X_SigX300A75 = IterateCuts(cuts,np.ma.getdata(ExtractParams(out["X300A75"])))
print("X300A75 Done!")
X_Background = IterateCuts(cuts,np.ma.getdata(ExtractParams(out["Background"])))
print("Cuts Done")

print("Calculating S/sqrt(S+B)")
sigs15=CalcSvsB(X_SigX300A15,X_Background)
print("X300A15 Done!")
sigs30=CalcSvsB(X_SigX300A30,X_Background)
print("X300A30 Done!")
sigs75=CalcSvsB(X_SigX300A75,X_Background)
print("Done!")

print("Making plots...")
PlotCutVsSum("MA",sigs15,"15",cuts)
PlotCutVsSum(r'$\Delta R_1$',sigs15,"15",cuts)
PlotCutVsSum(r'$\Delta R_2$',sigs15,"15",cuts)
PlotCutVsSum(r'$\Delta\eta$',sigs15,"15",cuts)

PlotCutVsSum("MA",sigs30,"30",cuts)
PlotCutVsSum(r'$\Delta R_1$',sigs30,"30",cuts)
PlotCutVsSum(r'$\Delta R_2$',sigs30,"30",cuts)
PlotCutVsSum(r'$\Delta\eta$',sigs30,"30",cuts)

PlotCutVsSum("MA",sigs75,"75",cuts)
PlotCutVsSum(r'$\Delta R_1$',sigs75,"75",cuts)
PlotCutVsSum(r'$\Delta R_2$',sigs75,"75",cuts)
PlotCutVsSum(r'$\Delta\eta$',sigs75,"75",cuts)
print("All Done!")