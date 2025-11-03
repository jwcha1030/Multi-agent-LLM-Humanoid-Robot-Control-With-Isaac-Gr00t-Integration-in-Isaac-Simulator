from huggingface_hub import snapshot_download

DATASET_NAME = "Exhaust-Pipe-Sorting-task"
# DATASET_NAME = "Nut-Pouring-task"
# Limit the number of workers to a low number, like 1 or 2
# The default is 5.
snapshot_download(
    repo_id="nvidia/PhysicalAI-GR00T-Tuned-Tasks",
    repo_type="dataset",
    # allow_patterns=["Nut-Pouring-task/*"], # Equivalent to --include
    allow_patterns=[f"{DATASET_NAME}/*"], # Equivalent to --include
    local_dir="./datasets/", # Create the structure locally
    max_workers=10 # Optimized for your system: 28 cores, 1Gbps network, HDD storage
)
print(f"{DATASET_NAME} dataset downloaded to ./datasets.")