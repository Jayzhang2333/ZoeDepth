from .zoedepth_geometry_v1 import ZoeDepth_geometry

all_versions = {
    "v1": ZoeDepth_geometry,
}

get_version = lambda v : all_versions[v]