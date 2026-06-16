from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

repo_id = "bojian1/PHMFD-data"
repo_type = "dataset"

# 本地目录：按你的实际路径改
local_root = Path("Process_data")

# 远端目录：本地 Process_data/xxx 会上传到 HF 的 Process_data/xxx
remote_root = "Process_data"

# True 只打印，不上传；确认无误后改成 False
DRY_RUN = False

api = HfApi()

print("Fetching remote file list...")
remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))

ops = []

for p in local_root.rglob("*"):
    if not p.is_file():
        continue

    rel = p.relative_to(local_root).as_posix()
    path_in_repo = f"{remote_root}/{rel}"

    if path_in_repo in remote_files:
        print(f"[SKIP existing] {path_in_repo}")
        continue

    print(f"[ADD new] {p} -> {path_in_repo}")
    ops.append(
        CommitOperationAdd(
            path_in_repo=path_in_repo,
            path_or_fileobj=str(p),
        )
    )

print(f"\nNew files to upload: {len(ops)}")

if DRY_RUN:
    print("DRY_RUN=True, no files uploaded.")
    print("After checking the paths, set DRY_RUN=False and run again.")
else:
    batch_size = 100
    for i in range(0, len(ops), batch_size):
        batch = ops[i:i + batch_size]
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=batch,
            commit_message=f"Upload new Process_data files batch {i // batch_size + 1}",
        )
        print(f"Uploaded batch {i // batch_size + 1}: {len(batch)} files")