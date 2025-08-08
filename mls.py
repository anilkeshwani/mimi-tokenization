#!/usr/bin/env python

"""
Encode the Multilingual LibriSpeech (MLS) dataset using SpeechTokenizer.

Recall:
Layer 0          -> "Semantic" content distilled down from WavLM
Layers 1:        -> Acoustic content; trained like EnCodec (reconstruction + adversarial losses)

Speed of Mimi tokenization:
- GPU: ~28.7096774194 (890 samples / 31s) -> ~4.45 GPU days for 10.8M trainset
- CPU: ~2.1293103448 (741 samples / 348s) -> ~58.9 CPU days for 10.8M trainset <- WINNER

"""

import json
import logging
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor


# Logging configuration
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

# Constants
MLS_SPLIT_SIZES = {"train": 10_808_037, "dev": 3_807, "test": 3_769}
# DEVICE = torch.device("cuda")  # leave GPU assignment to Slurm
DEVICE = torch.device("cpu")  # leave GPU assignment to Slurm
# Constants: Mimi
MIMI_REPO_ID = "kyutai/mimi"
_mimi_feat_extractor: EncodecFeatureExtractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
MIMI_SR = _mimi_feat_extractor.sampling_rate
assert MIMI_SR == 24_000

# Local MLS Dataset Paths
_MLS_SEGMENTS_PATH = "/mnt/scratch-artemis/shared/datasets/MLS/{}/segments.txt"
_MLS_AUDIO_DIR = "/mnt/scratch-artemis/shared/datasets/MLS/{}/audio"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "mimi_mls"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Encode MLS dataset with Mimi.")
    # Required
    parser.add_argument("idx_block", type=int, help="Block index to process (0-based)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev", "test"])
    # Optional
    parser.add_argument("--block_size", type=int, default=1_000)  # TODO finesse value
    parser.add_argument("--output_jsonl", type=Path)
    args = parser.parse_args()

    if args.idx_block < 0 or args.idx_block * args.block_size >= MLS_SPLIT_SIZES[args.split]:
        raise ValueError(
            f"Invalid block index {args.idx_block} for split '{args.split}' and block size {args.block_size}."
        )

    return args


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    """Infer path of the audio file from the MLS ID and audio directory.

    Args:
        mls_id (str): ID as found in transcripts.txt file e.g. 10214_10108_000000
        audio_dir (Path): "audio" directory e.g. /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio
        suffix (str, optional): File extension. Defaults to ".flac".

    Returns:
        Path: Resolved path pointing to audio file
    """
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)


@torch.inference_mode()
def mimi_encode_mls(idx_block: int, block_size: int, split: str, output_jsonl: Path | None):
    mimi = MimiModel.from_pretrained(MIMI_REPO_ID)
    mimi.eval()
    mimi.to(DEVICE)

    mls_split_size = MLS_SPLIT_SIZES[split]
    mls_segments = _MLS_SEGMENTS_PATH.format(split)
    mls_audio_dir = Path(_MLS_AUDIO_DIR.format(split))
    n_blocks = ceil(mls_split_size / block_size)

    with open(mls_segments, "r") as f:
        mls_ids: list[str] = [line.strip().split(None, 1)[0] for line in f]

    if len(mls_ids) != mls_split_size:
        raise ValueError(f"Expected {mls_split_size} MLS IDs in {mls_segments}, but found {len(mls_ids)}.")

    # Get the block of MLS IDs to process
    start_idx = idx_block * block_size
    end_idx = min((idx_block + 1) * block_size, mls_split_size)
    mls_ids = mls_ids[start_idx:end_idx]

    if output_jsonl is None:
        idx_block_label = str(idx_block + 1).zfill(len(str(n_blocks)))  # NOTE 1-indexed block label
        jsonl_filename = f"{split}-mls-mimi-{idx_block_label}-of-{n_blocks}.jsonl"
        output_jsonl = OUTPUT_DIR / split / jsonl_filename

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "x") as f:
        for mls_id in tqdm(mls_ids, desc="Processing MLS with Mimi"):
            # Load and pre-process speech waveform
            wav, sr = torchaudio.load(mls_id_to_path(mls_id, mls_audio_dir))

            # monophonic checking
            if wav.size(0) > 1:
                warnings.warn(f"Audio {mls_id} is not monophonic. Shape: {wav.shape}. Taking the first channel.")
                wav = wav[:1, :]  # take first dimension whilst maintaining second - channel - dimension

            # sample rate checking
            if sr != MIMI_SR:
                LOGGER.debug(f"Audio {mls_id} has sample rate {sr}, expected {MIMI_SR}. Resampling...")
                wav = torchaudio.functional.resample(wav, sr, MIMI_SR)

            # Extract discrete codes from Mimi
            encoded_frames, encoder_past_key_values, padding_cache = mimi.encode(
                wav.to(DEVICE).unsqueeze(0),
                num_quantizers=8,
                return_dict=False,
            )

            encoded_frames.squeeze_(0)  # in-place

            # Write Split RVQ codes to file
            mimi_srvqs: dict[str, list[int]] = {f"SRVQ_{idx_q}": st.tolist() for idx_q, st in enumerate(encoded_frames)}
            mimi_sample: dict[str, Any] = {"ID": mls_id} | mimi_srvqs

            f.write(json.dumps(mimi_sample) + "\n")
            # f.flush()

    print(f"Completed. Encoded block {idx_block} to {output_jsonl}.")


if __name__ == "__main__":
    mimi_encode_mls(**vars(parse_args()))
