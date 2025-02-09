from .zoedepth_sparse_feature_ga_v1 import ZoeDepth_sparse_feature_ga

all_versions = {
    "v1": ZoeDepth_sparse_feature_ga,
}

get_version = lambda v : all_versions[v]