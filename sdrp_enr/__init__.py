"""Paper-facing SDRP-ENR solver package.

SDRP-ENR stands for Service-Decoded Route Pool Matheuristic with Elite
Neighborhood Refinement.
"""

__version__ = "1.0.0"

FINAL_NEIGHBORHOODS = frozenset({"Drop-and-reinsert-LNS", "Rebalance-LNS"})

__all__ = ["FINAL_NEIGHBORHOODS", "__version__"]
