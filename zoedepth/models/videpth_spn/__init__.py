from .videpth_spn_v1 import videpth_spn

all_versions = {
    "v1": videpth_spn,
}

get_version = lambda v : all_versions[v]