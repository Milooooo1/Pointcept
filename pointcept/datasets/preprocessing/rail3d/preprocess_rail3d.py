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

# Function to convert files from .ply to .npy and break it up in smaller tiles
def process_ply_file(ply_file_path, output_dir, include_intensity, tile_size):
    print(f"Processing file: {ply_file_path}...")
    # Read the .ply file
    data = read_ply(ply_file_path)
    coords = np.vstack([data['x'], data['y'], data['z']]).T
    labels = data['scalar_Classification'] if 'scalar_Classification' in data.dtype.names else np.zeros(coords.shape[0], dtype=np.int32)
    
    # Filter out points with labels 0 or above 10
    valid_indices = (labels > 0) & (labels < 10)
    coords = coords[valid_indices]
    labels = labels[valid_indices] - 1  # Adjust labels to range [0-9]
    
    # Include intensity if needed
    intensities = data['vertex']['scalar_Intensity'][valid_indices] if include_intensity and 'scalar_Intensity' in data['vertex'].dtype.names else None

    # Determine the bounding box of the point cloud
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
    
    
    # Create a grid of tiles based on the tile size
    x_ranges = np.arange(min_x, max_x, tile_size)
    y_ranges = np.arange(min_y, max_y, tile_size)
    
    # Process each tile
    for i, x_start in enumerate(x_ranges):
        for j, y_start in enumerate(y_ranges):
            x_end = x_start + tile_size
            y_end = y_start + tile_size
            
            # Find points within the current tile
            tile_indices = (
                (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) &
                (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
            )
            
            tile_coords = coords[tile_indices]
            tile_labels = labels[tile_indices]
            
            if tile_coords.shape[0] == 0:
                continue  # Skip empty tiles
            
            # Create output directory
            file_output_dir = Path(output_dir) / Path(f"{ply_file_path.stem}-tile_{i}_{j}")
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cropped data for this tile
            np.save(file_output_dir / 'coord.npy', tile_coords.astype(np.float32))
            np.save(file_output_dir / 'segment.npy', tile_labels.astype(np.int16))
            
            if include_intensity and intensities is not None:
                tile_intensities = intensities[tile_indices]
                np.save(file_output_dir / 'strength.npy', tile_intensities.astype(np.float32))


# Function to process a directory
def process_directory(input_dir, output_dir, num_workers, include_intensity, tile_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ply_files = [f for f in input_dir.iterdir() if f.suffix == '.ply']
    tasks = []
    
    for ply_file in ply_files:
        tasks.append((ply_file, output_dir, include_intensity, tile_size))
    
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
    parser.add_argument(
        "--tile_size",
        help="Include intensities in the output files.",
        default=10,
        type=int,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for split in ['train', 'test', 'val']:
        input_dir = dataset_root / split
        output_dir = output_root / split
        process_directory(input_dir, output_dir, args.num_workers, args.include_intensity, args.tile_size)

