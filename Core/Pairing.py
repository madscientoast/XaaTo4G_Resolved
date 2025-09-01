import awkward as ak
import numpy as np
from itertools import permutations

def PairDeltaR(photons):
    # Evaluate DeltaR for all possible pairs #
    DR_AB = photons[:,0].deltaR(photons[:,1])
    DR_AC = photons[:,0].deltaR(photons[:,2])
    DR_AD = photons[:,0].deltaR(photons[:,3])
    DR_CD = photons[:,2].deltaR(photons[:,3])
    DR_BD = photons[:,1].deltaR(photons[:,3])
    DR_BC = photons[:,1].deltaR(photons[:,2])

    DR1 = DR_AB + DR_CD 
    DR2 = DR_AC + DR_BD
    DR3 = DR_AD + DR_BC

    DR = ak.concatenate([DR1[:, np.newaxis], DR2[:, np.newaxis], DR3[:,np.newaxis]], axis=1)

    # Find index (0, 1, or 2) of min-ΔR pairing for each event
    return ak.argmin(DR,axis=1)



def unique_pairing_perms(n):
    """
    Return representative permutations of 0..n-1 such that
    permutations that correspond to the same unordered pairing
    (and one singleton if n is odd) are deduplicated.

    Interpretation:
      - Break a permutation into consecutive pairs; if n is odd,
        the last element is the singleton.
      - Order inside each pair doesn't matter.
      - Order of the pairs doesn't matter.
      - (For odd n) which position the singleton appears in the
        original permutation doesn't matter.
    """
    parts = np.arange(n)
    perms = np.array(list(permutations(parts)))  # shape (n!, n)

    def key_arr(p):
        if n % 2 == 0:
            pairs = p.reshape(-1, 2)
            pairs = np.sort(pairs, axis=1)                    # sort inside each pair
            pairs = pairs[np.lexsort(pairs.T[::-1])]          # sort pairs lexicographically
            return pairs.flatten()
        else:
            singleton = np.array([p[-1]])
            pairs = p[:-1].reshape(-1, 2)
            pairs = np.sort(pairs, axis=1)
            pairs = pairs[np.lexsort(pairs.T[::-1])]
            return np.concatenate([singleton, pairs.flatten()])

    keys = np.array(list(map(key_arr, perms)))
    _, idx = np.unique(keys, axis=0, return_index=True)
    return perms[idx]   # representative permutations for each unique pairing