#!/usr/bin/env python
"""
Create LIBERO Dataset Training Metadata Configuration

Usage:
    python create_libero_meta.py \\
        --data_dir /datasets/metas \\
        --output ./datasets/metas/libero_train.json

This will scan the LIBERO dataset directory and generate a metadata file 
containing all HDF5 file paths and task descriptions.

Note: Each LIBERO HDF5 file contains 50 demos (episodes).
"""

import argparse
import json
import os
import glob
import re
from typing import List, Dict

import h5py


def parse_task_from_filename(filepath: str) -> str:
    """Parse task description from filename."""
    base = os.path.basename(filepath)
    # e.g., KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
    task = re.sub(r"_demo\.hdf5$", "", base)
    m = re.search(r"SCENE\d+_", task)
    if m:
        task = task[m.end():]
    task = task.replace("_", " ")
    return task


def count_demos_in_h5(h5_path: str) -> int:
    """Count demos in HDF5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'data' in f:
                return len([k for k in f['data'].keys() if k.startswith('demo')])
    except Exception as e:
        print(f"Warning: Failed to read {h5_path}: {e}")
    return 0


def create_libero_meta(
    data_dir: str,
    subsets: List[str] = None,
    output_path: str = None
) -> Dict:
    """
    Create LIBERO dataset meta configuration.
    
    Args:
        data_dir: LIBERO dataset root directory
        subsets: List of subsets to include
        output_path: Output JSON path
        
    Returns:
        meta dictionary
    """
    if subsets is None:
        # Default 4 subsets (excluding libero_90)
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    
    datalist = []
    stats = {}
    total_demos = 0
    
    print(f"Scanning LIBERO dataset: {data_dir}")
    
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"Warning: Skipping non-existent subset: {subset}")
            continue
            
        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        subset_demos = 0
        
        for h5_path in h5_files:
            task = parse_task_from_filename(h5_path)
            num_demos = count_demos_in_h5(h5_path)
            subset_demos += num_demos
            
            datalist.append({
                "path": h5_path,
                "task": task,
                "subset": subset,
                "num_demos": num_demos,
            })
        
        stats[subset] = {
            "num_files": len(h5_files),
            "num_demos": subset_demos,
        }
        total_demos += subset_demos
        
        print(f"   {subset}: {len(h5_files)} files, {subset_demos} demos")
    
    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": data_dir,
        "datalist": datalist,
        "num_files": len(datalist),
        "num_episodes": total_demos,
        "subsets": list(stats.keys()),
        "subset_stats": stats,
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }
    
    print(f"\nFound {len(datalist)} HDF5 files, {total_demos} episodes (demos)")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to: {output_path}")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Create LIBERO training metadata")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO dataset root directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["libero_10", "libero_goal", "libero_object", "libero_spatial"],
                        help="Subsets to include (default 4 subsets, excluding libero_90)")
    parser.add_argument("--output", type=str,
                        default="./datasets/metas/libero_train.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    create_libero_meta(
        data_dir=args.data_dir,
        subsets=args.subsets,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
