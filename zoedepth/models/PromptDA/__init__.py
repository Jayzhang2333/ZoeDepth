from .PromptDA_v1 import PromptDA

all_versions = {
    "v1": PromptDA,
}

get_version = lambda v : all_versions[v]