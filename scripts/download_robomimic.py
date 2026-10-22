"""Download robomimic v1.5 hdf5s from HF, skipping files already present; verify sizes against HF metadata."""
import argparse, os
from huggingface_hub import HfApi
import robomimic
from robomimic import DATASET_REGISTRY, HF_REPO_ID
import robomimic.utils.file_utils as FileUtils

p = argparse.ArgumentParser()
p.add_argument("--download_dir", required=True)
p.add_argument("--spec", nargs="+", required=True, help="task:dataset_type:hdf5_type, e.g. square:ph:raw")
args = p.parse_args()

api = HfApi()
for spec in args.spec:
    task, dtype, htype = spec.split(":")
    rel = DATASET_REGISTRY[task][dtype][htype]["url"]
    assert rel is not None, f"{spec} is not hosted (generate it from raw)"
    out_dir = os.path.join(args.download_dir, task, dtype)
    out = os.path.join(out_dir, os.path.basename(rel))
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(out):
        print(f"downloading {spec} -> {out}", flush=True)
        FileUtils.download_file_from_hf(HF_REPO_ID, rel, out_dir, check_overwrite=False)
    remote = api.get_paths_info(HF_REPO_ID, [rel], repo_type="dataset")[0].size
    local = os.path.getsize(out)
    print(f"{'OK  ' if local == remote else 'SIZE MISMATCH'} {out} local={local} hf={remote}", flush=True)
