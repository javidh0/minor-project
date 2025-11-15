import os
import pickle
import hashlib

ENV_VAR = "VIDEO_PROCESSOR_SAVE_DIR"

SAVE_DIR = "./"

def setSaveDir(loc):
    SAVE_DIR = loc

def _key_from_path(video_path: str):
    """Create a stable, filesystem-safe key for each video path."""
    return hashlib.md5(video_path.encode()).hexdigest()


def save_processor(video_path: str, processor_obj):
    """Save ONE VideoProcessor object safely."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    key = _key_from_path(video_path)
    file_path = os.path.join(SAVE_DIR, key + ".pkl")

    with open(file_path, "wb") as f:
        pickle.dump(processor_obj, f)

    print(f"[SAVED] {video_path} → {file_path}")
    return file_path


def load_all_processors():
    """Load all saved processors into a dict."""
    processors = {}

    if not os.path.exists(SAVE_DIR):
        raise FileNotFoundError(f"Save directory does not exist: {SAVE_DIR}")

    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".pkl"):
            key = filename.replace(".pkl", "")
            file_path = os.path.join(SAVE_DIR, filename)

            with open(file_path, "rb") as f:
                processors[key] = pickle.load(f)

    print(f"[LOADED] {len(processors)} processors from {SAVE_DIR}")
    return processors
