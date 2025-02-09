from .zoedepth_sparse_feature_fusion_v1 import ZoeDepth_sparse_feature_fusion

all_versions = {
    "v1": ZoeDepth_sparse_feature_fusion,
}

get_version = lambda v : all_versions[v]