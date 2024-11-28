from huggingface_hub import snapshot_download
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
snapshot_download(repo_id="TianxingChen/G3Flow", local_dir=parent_dir, repo_type="dataset")