"""
Create train/val/test splits from synthetic peptides dataset.
Maintains the same folder structure so you only need to change the path.

Usage:
    python create_data_splits.py

Output structure:
    data/synthetic_peptides_split/
        train/
            nanotubes/
            micelles/
            nanofibers/
            random_aggregates/
        val/
            nanotubes/
            micelles/
            nanofibers/
            random_aggregates/
        test/
            nanotubes/
            micelles/
            nanofibers/
            random_aggregates/
"""

import os
import shutil
import glob
from pathlib import Path
import random
import json

# Configuration
SOURCE_DIR = './data/synthetic_peptides'
OUTPUT_DIR = './data/synthetic_peptides_split'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Structure type proportions (None = use all available files equally)
# Set to a dict to control proportions, e.g., {'nanotubes': 0.4, 'micelles': 0.3, ...}
# Values should sum to 1.0, or use absolute counts like {'nanotubes': 500, 'micelles': 300, ...}
STRUCTURE_PROPORTIONS = {
    'nanotubes': 1000,
    'micelles': 1000,
    'nanofibers': 1000,
    'lamellar_sheets': 1000,
    'random_aggregates': 10000
}
# Maximum total files to use (None = use all)
# If set with STRUCTURE_PROPORTIONS, proportions are applied to this total
MAX_TOTAL_FILES = None

def create_splits():
    """Create train/val/test splits maintaining folder structure."""
    
    random.seed(RANDOM_SEED)
    
    # Validate ratios
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all subdirectories (structure types)
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        print(f"❌ Error: Source directory '{SOURCE_DIR}' does not exist!")
        return
    
    structure_types = [d.name for d in source_path.iterdir() if d.is_dir()]
    
    if not structure_types:
        print(f"❌ Error: No subdirectories found in '{SOURCE_DIR}'")
        return
    
    print(f"Found structure types: {', '.join(structure_types)}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Handle structure proportions
    files_per_structure = {}
    
    for struct_type in structure_types:
        pattern = str(source_path / struct_type / '*.gro')
        all_files = glob.glob(pattern)
        files_per_structure[struct_type] = all_files
    
    # Determine how many files to use per structure type
    files_to_use = {}
    
    if STRUCTURE_PROPORTIONS is None:
        # Use all files
        for struct_type in structure_types:
            files_to_use[struct_type] = files_per_structure[struct_type]
        print(f"Using all available files per structure type")
    else:
        # Validate proportions
        if not all(st in structure_types for st in STRUCTURE_PROPORTIONS.keys()):
            unknown = set(STRUCTURE_PROPORTIONS.keys()) - set(structure_types)
            print(f"❌ Error: Unknown structure types in STRUCTURE_PROPORTIONS: {unknown}")
            return
        
        # Check if proportions or absolute counts
        prop_sum = sum(STRUCTURE_PROPORTIONS.values())
        
        if abs(prop_sum - 1.0) < 1e-6:
            # Proportions mode
            if MAX_TOTAL_FILES is None:
                # Use minimum available as basis
                min_available = min(len(files_per_structure[st]) for st in STRUCTURE_PROPORTIONS.keys())
                total_target = min_available * len(STRUCTURE_PROPORTIONS)
                print(f"Using proportions with auto-calculated total: {total_target} files")
            else:
                total_target = MAX_TOTAL_FILES
                print(f"Using proportions with MAX_TOTAL_FILES: {total_target} files")
            
            for struct_type in structure_types:
                if struct_type in STRUCTURE_PROPORTIONS:
                    n_files = int(total_target * STRUCTURE_PROPORTIONS[struct_type])
                    available = files_per_structure[struct_type]
                    
                    if n_files > len(available):
                        print(f"⚠️  Warning: Requested {n_files} for {struct_type}, but only {len(available)} available. Using {len(available)}.")
                        n_files = len(available)
                    
                    # Randomly sample
                    random.shuffle(available)
                    files_to_use[struct_type] = available[:n_files]
                else:
                    # Structure type not in proportions, skip it
                    files_to_use[struct_type] = []
        else:
            # Absolute counts mode
            print(f"Using absolute counts per structure type")
            for struct_type in structure_types:
                if struct_type in STRUCTURE_PROPORTIONS:
                    n_files = int(STRUCTURE_PROPORTIONS[struct_type])
                    available = files_per_structure[struct_type]
                    
                    if n_files > len(available):
                        print(f"⚠️  Warning: Requested {n_files} for {struct_type}, but only {len(available)} available. Using {len(available)}.")
                        n_files = len(available)
                    
                    # Randomly sample
                    random.shuffle(available)
                    files_to_use[struct_type] = available[:n_files]
                else:
                    files_to_use[struct_type] = []
        
        # Print selection summary
        print(f"\nStructure type selection:")
        for struct_type in structure_types:
            available = len(files_per_structure[struct_type])
            selected = len(files_to_use[struct_type])
            print(f"  {struct_type}: {selected}/{available} files ({selected/max(available,1)*100:.1f}%)")
    
    print()
    
    # Create output directory structure
    output_path = Path(OUTPUT_DIR)
    for split in ['train', 'val', 'test']:
        for struct_type in structure_types:
            split_dir = output_path / split / struct_type
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each structure type
    split_info = {}
    total_files = {'train': 0, 'val': 0, 'test': 0}
    
    for struct_type in structure_types:
        print(f"Processing {struct_type}...")
        
        # Get selected files for this structure type
        files = files_to_use[struct_type]
        
        if not files:
            print(f"  ⚠️  No files selected for {struct_type}, skipping")
            continue
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split sizes
        n_total = len(files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val  # Ensure all files are used
        
        # Split files
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Copy files to respective directories
        for file_list, split_name in [(train_files, 'train'), 
                                       (val_files, 'val'), 
                                       (test_files, 'test')]:
            for src_file in file_list:
                src_path = Path(src_file)
                dst_path = output_path / split_name / struct_type / src_path.name
                shutil.copy2(src_path, dst_path)
        
        # Store split information
        split_info[struct_type] = {
            'total': n_total,
            'train': n_train,
            'val': n_val,
            'test': n_test
        }
        
        total_files['train'] += n_train
        total_files['val'] += n_val
        total_files['test'] += n_test
        
        print(f"  ✅ {struct_type}: {n_train} train, {n_val} val, {n_test} test (total: {n_total})")
    
    # Save split information
    metadata = {
        'source_directory': SOURCE_DIR,
        'output_directory': OUTPUT_DIR,
        'random_seed': RANDOM_SEED,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'structure_proportions': STRUCTURE_PROPORTIONS,
        'max_total_files': MAX_TOTAL_FILES,
        'structure_types': split_info,
        'totals': total_files
    }
    
    metadata_path = output_path / 'split_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Split complete!")
    print(f"{'='*60}")
    print(f"Total files split:")
    print(f"  Train: {total_files['train']}")
    print(f"  Val:   {total_files['val']}")
    print(f"  Test:  {total_files['test']}")
    print(f"  Total: {sum(total_files.values())}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\n💡 To use in your notebook, change the path to:")
    print(f"   './data/synthetic_peptides_split/train'")
    print(f"{'='*60}")

if __name__ == '__main__':
    create_splits()
