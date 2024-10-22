from .zoedepth_sparse_feature_da_v1 import ZoeDepth_sparse_feature_da

all_versions = {
    "v1": ZoeDepth_sparse_feature_da,
}

get_version = lambda v : all_versions[v]