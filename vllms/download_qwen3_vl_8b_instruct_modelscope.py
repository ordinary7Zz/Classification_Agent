import os
import shutil
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

cache_dir = "/mnt/wangbd8/workspace/modelscope_cache"
target_dir = "/mnt/wangbd8/workspace/Qwen3-VL-8B-Instruct"
model_id = "Qwen/Qwen3-VL-8B-Instruct"

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

try:
    # 某些 ModelScope 版本支持 local_dir（但不一定支持 local_dir_use_symlinks）
    path = snapshot_download(
        model_id=model_id,
        cache_dir=cache_dir,
        local_dir=target_dir,
        revision=None,
    )
    print("✅ Downloaded directly to:", path)
except TypeError:
    # 回退：先下载到缓存，再拷贝到 target_dir
    path = snapshot_download(
        model_id=model_id,
        cache_dir=cache_dir,
        revision=None,
    )
    print("✅ Downloaded to cache:", path)

    src = Path(path)
    dst = Path(target_dir)

    # 把缓存内容拷贝到目标目录（不会用软链接）
    # 如果目标目录已存在文件，会覆盖同名文件
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)

    print("✅ Copied to:", target_dir)

