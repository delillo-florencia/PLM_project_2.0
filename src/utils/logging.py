import os
import re

def get_latest_version(base_dir, run_name):
    """
    it is to get a directory name for current run 
    to synchronize places where metrics and outputs are saved :-)
    """
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    version_dirs = [d for d in os.listdir(run_dir) if re.match(r'version_\d+', d)]
    if not version_dirs:
        version_str = "version_0"
        new_version_dir = os.path.join(run_dir, version_str)
        os.makedirs(new_version_dir, exist_ok=True)
        return new_version_dir, version_str

    latest_version = max(version_dirs, key=lambda x: int(re.search(r'version_(\d+)', x).group(1)))
    latest_num = int(re.search(r'version_(\d+)', latest_version).group(1))
    new_version_num = latest_num + 1
    version_str = f"version_{new_version_num}"
    new_version_dir = os.path.join(run_dir, version_str)
    os.makedirs(new_version_dir, exist_ok=True)
    return new_version_dir, version_str