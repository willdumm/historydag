import pickle
from gctree import phylip_parse as pps

treelist = pps.parse_outfile("outfile", root="GL")
with open("resolvedtrees.p", "wb") as fh:
    fh.write(pickle.dumps(treelist))
