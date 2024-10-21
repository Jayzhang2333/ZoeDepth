from .zoedepth_sparse_feature_v1 import ZoeDepth_sparse_feature

all_versions = {
    "v1": ZoeDepth_sparse_feature,
}

get_version = lambda v : all_versions[v]