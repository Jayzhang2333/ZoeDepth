from .DA_SML_v1 import DA_SML

all_versions = {
    "v1": DA_SML,
}

get_version = lambda v : all_versions[v]