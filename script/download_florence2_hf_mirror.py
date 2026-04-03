import os
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Download a Florence-2 model from Hugging Face through a mirror endpoint")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Florence-2-large-ft",
        help="Hugging Face model id, e.g. microsoft/Florence-2-base-ft",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://hugging-face.cn",
        help="Mirror endpoint. You can also try https://hf-mirror.com if needed.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional cache directory for Hugging Face downloads",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="",
        help="Optional output directory. If set, files are materialized there.",
    )
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

    try:
        from huggingface_hub import HfApi, hf_hub_download
        try:
            from huggingface_hub.utils import enable_progress_bars
            from huggingface_hub.utils import logging as hf_logging
            enable_progress_bars()
            hf_logging.set_verbosity_info()
        except Exception:
            pass
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Run `pip install -U huggingface_hub` first."
        ) from exc

    print("Start downloading from Hugging Face mirror...", flush=True)
    print("Model:", args.model_id, flush=True)
    print("Endpoint:", args.endpoint, flush=True)
    if args.local_dir:
        print("Local dir:", args.local_dir, flush=True)
    if args.cache_dir:
        print("Cache dir:", args.cache_dir, flush=True)

    api = HfApi(endpoint=args.endpoint)
    model_info = api.model_info(args.model_id)
    filenames = [sibling.rfilename for sibling in model_info.siblings]
    print("Files to download:", len(filenames), flush=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    downloaded_paths = []
    iterable = tqdm(filenames, desc="Downloading files") if tqdm is not None else filenames

    for filename in iterable:
        if tqdm is None:
            print("Downloading:", filename, flush=True)
        else:
            iterable.set_postfix_str(filename)

        file_path = hf_hub_download(
            repo_id=args.model_id,
            filename=filename,
            endpoint=args.endpoint,
            cache_dir=args.cache_dir if args.cache_dir else None,
            local_dir=args.local_dir if args.local_dir else None,
            local_dir_use_symlinks=False if args.local_dir else "auto",
            resume_download=True,
        )
        downloaded_paths.append(file_path)

    if args.local_dir:
        model_dir = args.local_dir
    elif downloaded_paths:
        first_path = Path(downloaded_paths[0])
        parts = first_path.parts
        if "snapshots" in parts:
            snap_idx = parts.index("snapshots")
            model_dir = str(Path(*parts[: snap_idx + 2]))
        else:
            model_dir = str(first_path.parent)
    else:
        model_dir = ""

    print("Download finished.")
    print("Mirror endpoint:")
    print(args.endpoint)
    print("Local model path:")
    print(model_dir)
    print("")
    print("Example usage:")
    print(
        "python render_lerf_mask_ours_v2.py "
        "-m /path/to/output "
        "--skip_train --iteration -1 "
        f"--florence_model_id {model_dir}"
    )


if __name__ == "__main__":
    main()
