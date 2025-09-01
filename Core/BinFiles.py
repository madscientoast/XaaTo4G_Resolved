import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import uproot as up
from Pairing import PairDeltaR, unique_pairing_perms
import hist
import matplotlib.pyplot as plt
from plotting import ConvertToROOT

NanoAODSchema.warn_missing_crossrefs=False

# Define File(s) #
fileset = {
    "Background": {
        "files": {
            "/project01/ndcms/rsnuggs/DataFiles/merged_output.root" : "Events"
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
    "X3000A150": {
        "files": {
            "../EHS_simulation/X3000A150/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
    "X3000A300": {
        "files": {
            "../EHS_simulation/X3000A300/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
    "X3000A750": {
        "files": {
            "../EHS_simulation/X3000A750/result.root" : "Events"
        },
        "metadata": {
            "is_mc": True
        }
    },
}

# Define What to actually do here #
class MyProcessor(processor.ProcessorABC):
    def __init__(self, mode="virtual"):
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata['dataset']
        all_photons = events.Photon[ak.num(events.Photon.pt) >= 4 & events.HLT.TriplePhoton_35_35_5_CaloIdLV2_R9IdVL] #Grab events with at least 4 photons

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

        mass_asym = abs(Phi0.mass - Phi1.mass)/(Phi0.mass + Phi1.mass)
        X = X[mass_asym < 0.6]
        Phi0 = Phi0[mass_asym < 0.6]
        Phi1 = Phi1[mass_asym < 0.6]
        masses = ak.to_numpy(X.mass)
        
        return {
            dataset: {
                "entries": len(events),
                "masses": masses,
                "Phi0": ak.to_numpy(Phi0.mass),
                "Phi1": ak.to_numpy(Phi1.mass)
            }
        }
    
    def postprocess(self, accumulator):
        pass

# Do things here #
iterative_run = processor.Runner(
    executor = processor.FuturesExecutor(workers=4, compression=None),
    schema=NanoAODSchema,
    chunksize=100000000,
    savemetrics=True,
)

out, metrics = iterative_run(
    fileset,
    processor_instance=MyProcessor("virtual"),
)

# X300 Files #
X300A15 = out["X300A15"]["masses"]
X300A30 = out["X300A30"]["masses"]
X300A75 = out["X300A75"]["masses"]
# X3000 Files #
X3000A150 = out["X3000A150"]["masses"]
X3000A300 = out["X3000A300"]["masses"]
X3000A750 = out["X3000A750"]["masses"]
# Data #
BKG = out["Background"]["masses"]

# X300 File #
# Make ROOT plots
h_X300A15 = ConvertToROOT(X300A15,100,[0.0,1000.0])
h_X300A30 = ConvertToROOT(X300A30,100,[0.0,1000.0])
h_X300A75 = ConvertToROOT(X300A75,100,[0.0,1000.0])
h_BKG = ConvertToROOT(BKG,100,[0.0,1000.0])

# Dump To File
with up.recreate("shapes_X300.root") as fout:
    fout["signal_m15"] = h_X300A15
    fout["signal_m30"] = h_X300A30
    fout["signal_m75"] = h_X300A75
    fout["data_obs"] = h_BKG
#######################################################

# X 3000 File #
# Make ROOT plots
h_X3000A150 = ConvertToROOT(X3000A150,100,[0.0,3500.0])
h_X3000A300 = ConvertToROOT(X3000A300,100,[0.0,3500.0])
h_X3000A750 = ConvertToROOT(X3000A750,100,[0.0,3500.0])
h_BKG = ConvertToROOT(BKG,100,[0.0,3500.0])

# Dump To File
with up.recreate("shapes_X3000.root") as fout:
    fout["signal_m150"] = h_X3000A150
    fout["signal_m300"] = h_X3000A300
    fout["signal_m750"] = h_X3000A750
    fout["data_obs"] = h_BKG


selection = (BKG >= 400) & (BKG <= 410)
Phi0 = out["Background"]["Phi0"][selection]
Phi1 = out["Background"]["Phi1"][selection]

print(Phi0,Phi1)