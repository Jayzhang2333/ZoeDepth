from .zoedepth_conv_trans_v1 import ZoeDepth_conv_trans

all_versions = {
    "v1": ZoeDepth_conv_trans,
}

get_version = lambda v : all_versions[v]