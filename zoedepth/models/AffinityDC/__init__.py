# from .AffinityDC_v1 import AffinityDC
# from .AffinityDC_v2 import AffinityDC
from .AffinityDC_v3 import AffinityDC

all_versions = {
    # "v1": AffinityDC,
    # "v2": AffinityDC,
    "v3": AffinityDC
}

get_version = lambda v : all_versions[v]