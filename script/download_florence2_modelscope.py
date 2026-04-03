from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Download a Florence-2 model from ModelScope")
    parser.add_argument(
        "--model_id",
        type=str,
        default="AI-ModelScope/Florence-2-large-ft",
        help="ModelScope model id, e.g. AI-ModelScope/Florence-2-base-ft",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional cache directory for ModelScope downloads",
    )
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope is not installed. Run `pip install -U modelscope` first."
        ) from exc

    download_kwargs = {"model_id": args.model_id}
    if args.cache_dir:
        download_kwargs["cache_dir"] = args.cache_dir

    model_dir = snapshot_download(**download_kwargs)

    print("Download finished.")
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
