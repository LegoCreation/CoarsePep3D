#!/usr/bin/env python3
"""
Script to extract last 5 frames from peptide fiber simulation trajectories.
For each peptide, creates a 'last_five_frames' folder containing:
- .gro files for each of the last 5 frames
- numpy arrays with position data
- extraction summary
"""

import os
import shutil
import numpy as np
from pathlib import Path
import MDAnalysis as mda
import warnings
warnings.filterwarnings('ignore')

def extract_last_5_frames(xtc_file, gro_file, peptide_dir):
    """Extract last 5 frames from trajout.xtc file and save as .gro files"""
    try:
        # Read trajectory using MDAnalysis
        u = mda.Universe(gro_file, xtc_file)
        
        total_frames = len(u.trajectory)
        print(f"  ✓ Total frames in trajectory: {total_frames}")
        
        if total_frames < 5:
            print(f"  ! Warning: Only {total_frames} frames available, extracting all")
            start_frame = 0
        else:
            start_frame = total_frames - 5
        
        # Create last_five_frames directory in the original peptide folder
        frames_dir = peptide_dir / "last_five_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Extract last 5 (or fewer) frames and save as .gro files only
        frame_indices = []
        
        for i, ts in enumerate(u.trajectory[start_frame:]):
            frame_idx = start_frame + i
            frame_indices.append(frame_idx)
            
            # Save as .gro file only
            output_gro = frames_dir / f"frame_{frame_idx:06d}.gro"
            u.atoms.write(str(output_gro))
        
        # Create a summary file
        with open(frames_dir / "extraction_info.txt", "w") as f:
            f.write(f"Last 5 frames extraction summary\n")
            f.write(f"Total frames in trajectory: {total_frames}\n")
            f.write(f"Extracted frames: {frame_indices}\n")
            f.write(f"Number of atoms: {len(u.atoms)}\n")
            f.write(f"Files created:\n")
            for frame_idx in frame_indices:
                f.write(f"  - frame_{frame_idx:06d}.gro\n")
        
        print(f"  ✓ Extracted last {len(frame_indices)} frames as .gro files")
        return True
        
    except Exception as e:
        print(f"  ✗ Error extracting trajectory frames: {e}")
        return False

def process_single_peptide(peptide_dir, run_name, peptide_name):
    """Process a single peptide directory"""
    print(f"\n  Processing {peptide_name}...")
    
    # Find the required files
    gro_file = peptide_dir / "peptide-cg.gro"
    xtc_file = peptide_dir / "trajout.xtc"
    
    # Check if files exist
    if not gro_file.exists():
        print(f"  ✗ Missing .gro file: {gro_file}")
        return False
    
    if not xtc_file.exists():
        print(f"  ✗ Missing .xtc file: {xtc_file}")
        return False
    
    # Process the trajectory to extract last 5 frames
    success = extract_last_5_frames(xtc_file, gro_file, peptide_dir)
    
    return success

def process_run_directory(run_dir):
    """Process all peptides in a run directory"""
    run_name = run_dir.name
    print(f"\n{'='*50}")
    print(f"Processing {run_name}")
    print(f"{'='*50}")
    
    peptide_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    peptide_dirs.sort()  # Sort for consistent processing order
    
    success_count = 0
    total_count = len(peptide_dirs)
    
    for peptide_dir in peptide_dirs:
        peptide_name = peptide_dir.name
        success = process_single_peptide(peptide_dir, run_name, peptide_name)
        if success:
            success_count += 1
    
    print(f"\n{run_name} Summary: {success_count}/{total_count} peptides processed successfully")
    return success_count, total_count

def main():
    """Main processing function"""
    # Define paths
    base_path = Path("/home/go73dov/CoarsePep3D/data/peptides/fibers")
    
    print("Peptide Last 5 Frames Extraction Script")
    print("="*50)
    
    # Check if MDAnalysis is available
    try:
        import MDAnalysis as mda
        print("✓ MDAnalysis is available")
    except ImportError:
        print("✗ MDAnalysis is not installed. Please install it with:")
        print("  pip install MDAnalysis")
        return
    
    # Process each run
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_dirs.sort()
    
    total_success = 0
    total_peptides = 0
    
    for run_dir in run_dirs:
        success_count, peptide_count = process_run_directory(run_dir)
        total_success += success_count
        total_peptides += peptide_count
    
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total peptides processed: {total_success}/{total_peptides}")
    print(f"Success rate: {(total_success/total_peptides)*100:.1f}%")
    print(f"Last 5 frames saved in 'last_five_frames' folders within each peptide directory")
    
    # Create overall summary in the base directory
    with open(base_path / "last_frames_extraction_log.txt", "w") as f:
        f.write(f"Last 5 Frames Extraction Log\n")
        f.write(f"Generated on: {Path().cwd()}\n")
        f.write(f"Total peptides processed: {total_success}/{total_peptides}\n")
        f.write(f"Success rate: {(total_success/total_peptides)*100:.1f}%\n")
        f.write(f"\nProcessed runs:\n")
        for run_dir in run_dirs:
            f.write(f"  - {run_dir.name}\n")
        f.write(f"\nOutput: Each peptide directory now contains a 'last_five_frames' folder with:\n")
        f.write(f"  - frame_XXXXXX.gro files (last 5 frames in GROMACS format)\n")
        f.write(f"  - extraction_info.txt (extraction summary)\n")

if __name__ == "__main__":
    main()