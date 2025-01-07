from utils.ply import read_ply

import multiprocessing as mp
import numpy as np
import argparse
from pathlib import Path
import os

LABEL_DICT = {
    0: 'Ground',
    1: 'Vegetation',
    2: 'Rail',
    3: 'Poles',
    4: 'Wires',
    5: 'Signalling',
    6: 'Fences',
    7: 'Installation',
    8: 'Building',
    9: 'Other',
}

# Function to convert .ply to .npy
def process_ply_file(ply_file_path, output_dir, include_intensity):
    data = read_ply(ply_file_path)
    coord = np.vstack([data['x'], data['y'], data['z']]).T
    labels = data['scalar_Classification'] if 'scalar_Classification' in data.dtype.names else np.zeros(coord.shape[0], dtype=np.int32)
    
    # Filter out points with labels 0 or above 10
    valid_indices = (labels > 0) & (labels < 10)
    coord = coord[valid_indices]

    # Subtract 1 from labels to make the range from 0-9 instead of 1-10
    labels = labels[valid_indices] - 1  
    
    # Create output directory for the current file
    file_output_dir = output_dir / ply_file_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save coord and labels
    np.save(file_output_dir / 'coord.npy', coord.astype(np.float32))
    np.save(file_output_dir / 'segment.npy', labels.astype(np.int16))
    
    if include_intensity:
        intensities = data['scalar_Intensity'][valid_indices] if 'scalar_Intensity' in data.dtype.names else np.zeros(coord.shape[0], dtype=np.float32)
        np.save(file_output_dir / 'strength.npy', intensities.astype(np.float32))

# Function to process a directory
def process_directory(input_dir, output_dir, num_workers, include_intensity):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ply_files = [f for f in input_dir.iterdir() if f.suffix == '.ply']
    tasks = []
    
    for ply_file in ply_files:
        tasks.append((ply_file, output_dir, include_intensity))
    
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_ply_file, tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Rail3D dataset containing train, test, and val folders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/test/val folders will be located.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--include_intensity",
        action='store_true',
        help="Include intensities in the output files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for split in ['train', 'test', 'val']:
        input_dir = dataset_root / split
        output_dir = output_root / split
        process_directory(input_dir, output_dir, args.num_workers, args.include_intensity)

