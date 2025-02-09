from .zoedepth_videpth_v1 import ZoeDepth_videpth

all_versions = {
    "v1": ZoeDepth_videpth,
}

get_version = lambda v : all_versions[v]