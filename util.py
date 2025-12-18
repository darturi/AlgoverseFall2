import shutil
from pathlib import Path

USER_ID = "user_alpha"
DATA_DIR = "/Users/danielarturi/Desktop/PersonalProjects/AlgoverseFall2/data"
BASE_MEM_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": "./chroma_db"
        }
    }
}
MEM_LOCATION = "./chroma_db"

def delete_directory(path: str) -> None:
    dir_path = Path(path)

    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    else:
        raise FileNotFoundError(f"Directory not found: {dir_path}")

def copy_directory(src: str, dst: str) -> None:
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists() or not src_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    if dst_path.exists():
        raise FileExistsError(f"Destination already exists: {dst_path}")

    shutil.copytree(src_path, dst_path)