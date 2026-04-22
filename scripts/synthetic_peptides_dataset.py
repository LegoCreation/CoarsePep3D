"""Dataset utilities for residue-level centroid clouds from synthetic GRO files."""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_gro_atom_line(line: str) -> Tuple[int, str, str, int, np.ndarray]:
    """Parse a single fixed-width GRO atom line.

    GRO atom lines follow the format:
    resid(5) resname(3) atomname(7) atomid(5) x(8) y(8) z(8)
    """
    residue_id = int(line[:5].strip())
    residue_name = line[5:8].strip()
    bead_type = line[8:15].strip()
    atom_id = int(line[15:20].strip())
    coords = np.array(
        [
            float(line[20:28].strip()),
            float(line[28:36].strip()),
            float(line[36:44].strip()),
        ],
        dtype=np.float32,
    )
    return residue_id, residue_name, bead_type, atom_id, coords


def load_residue_centroids_from_gro(file_path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load one GRO file and convert each residue to a single centroid point.

    Residues are identified by the residue id stored in the first GRO column.

    Args:
        file_path: Path to a GRO file.
        normalize: If True, normalize centroids to the ConvOccNet-style range
            [-0.5, 0.5].

    Returns:
        A tuple of:
        - centroids: [num_residues, 3] float32 array
        - residue_ids: [num_residues] int64 array with residue ids
    """
    with open(file_path, "r") as handle:
        lines = handle.readlines()

    residue_points: "OrderedDict[int, List[np.ndarray]]" = OrderedDict()

    for raw_line in lines[2:-1]:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        residue_id, _, _, _, coords = _parse_gro_atom_line(line)
        residue_points.setdefault(residue_id, []).append(coords)

    if not residue_points:
        raise ValueError(f"No residue coordinates found in {file_path}")

    residue_ids = np.array(list(residue_points.keys()), dtype=np.int64)
    centroids = np.array(
        [np.mean(points, axis=0) for points in residue_points.values()],
        dtype=np.float32,
    )

    if normalize:
        centroid = np.mean(centroids, axis=0)
        centroids = centroids - centroid
        max_dist = np.max(np.abs(centroids))
        if max_dist > 0:
            centroids = centroids / (2.0 * max_dist)

    return centroids, residue_ids


def load_peptide_centroids_from_gro(file_path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for load_residue_centroids_from_gro."""
    return load_residue_centroids_from_gro(file_path, normalize=normalize)


class SyntheticPeptidesDataset(Dataset):
    """Dataset that represents each residue by one centroid point.

    The dataset scans synthetic GRO files, groups bead coordinates by residue id,
    and replaces each residue with its centroid.

    Args:
        data_path: Root folder containing structure subfolders with GRO files.
        structure_types: Optional list of subfolders to include.
        num_files: Optional number of files to subsample.
        target_num_points: Optional fixed output size. If provided, point clouds are
            randomly subsampled or zero-padded to this number of residue points.
        normalize: Whether to normalize centroids to [-0.5, 0.5].
        return_peptide_ids: Whether to include residue ids in the sample output.
        seed: Optional random seed used for file subsampling and point subsampling.
    """

    def __init__(
        self,
        data_path: str = "./data/synthetic_peptides",
        structure_types: Optional[List[str]] = None,
        num_files: Optional[int] = None,
        target_num_points: Optional[int] = None,
        normalize: bool = True,
        return_peptide_ids: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.data_path = data_path
        self.target_num_points = target_num_points
        self.normalize = normalize
        self.return_peptide_ids = return_peptide_ids
        self.rng = np.random.default_rng(seed)

        self.files: List[str] = []
        self.file_types: Dict[str, str] = {}
        files_by_type: Dict[str, List[str]] = {}

        if structure_types is None:
            structure_types = sorted(
                [
                    folder
                    for folder in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, folder))
                ]
            )

        for structure_type in structure_types:
            pattern = os.path.join(data_path, structure_type, "*.gro")
            matched_files = sorted(glob.glob(pattern))
            files_by_type[structure_type] = matched_files
            self.files.extend(matched_files)
            for file_path in matched_files:
                self.file_types[file_path] = structure_type

        if num_files is not None and len(self.files) > num_files:
            if num_files <= 0:
                raise ValueError("num_files must be a positive integer when provided")

            total_available = len(self.files)
            sampled_files: List[str] = []

            for structure_type in structure_types:
                available = files_by_type[structure_type]
                if not available:
                    continue

                percent = len(available) / total_available
                requested = min(int(percent * num_files), len(available))
                if requested == 0:
                    continue

                sampled_indices = self.rng.choice(len(available), size=requested, replace=False)
                sampled_files.extend([available[index] for index in sampled_indices])

            # Mix structure-specific picks while keeping the selected set fixed.
            self.rng.shuffle(sampled_files)
            self.files = sampled_files

        print(f"Loading centroid dataset from: {data_path}")
        print(f"Structure types: {', '.join(structure_types) if structure_types else 'none'}")
        selected_counts = {st: 0 for st in structure_types}
        for file_path in self.files:
            selected_counts[self.file_types[file_path]] += 1
        print("Selected files per structure:")
        for st in structure_types:
            print(f"  {st}: {selected_counts[st]}/{len(files_by_type[st])}")
        print(
            f"Dataset ready: {len(self.files)} files, "
            f"target_points={self.target_num_points}, normalize={self.normalize}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def get_file_type(self, idx: int) -> str:
        return self.file_types[self.files[idx]]

    def _sample_or_pad(
        self, centroids: np.ndarray, residue_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subsample or pad centroid clouds to a fixed size."""
        num_points = len(centroids)

        if self.target_num_points is None:
            mask = np.ones(num_points, dtype=np.float32)
            return centroids, residue_ids, mask

        if num_points >= self.target_num_points:
            selected = self.rng.choice(num_points, size=self.target_num_points, replace=False)
            selected.sort()
            centroids = centroids[selected]
            residue_ids = residue_ids[selected]
            mask = np.ones(self.target_num_points, dtype=np.float32)
            return centroids, residue_ids, mask

        pad_count = self.target_num_points - num_points
        centroids = np.concatenate(
            [centroids, np.zeros((pad_count, 3), dtype=np.float32)],
            axis=0,
        )
        residue_ids = np.concatenate(
            [residue_ids, -np.ones(pad_count, dtype=np.int64)],
            axis=0,
        )
        mask = np.concatenate(
            [np.ones(num_points, dtype=np.float32), np.zeros(pad_count, dtype=np.float32)],
            axis=0,
        )
        return centroids, residue_ids, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        centroids, residue_ids = load_residue_centroids_from_gro(
            file_path,
            normalize=self.normalize,
        )
        centroids, residue_ids, mask = self._sample_or_pad(centroids, residue_ids)

        num_residues = torch.tensor(int(mask.sum()), dtype=torch.int64)

        sample: Dict[str, torch.Tensor] = {
            "points": torch.tensor(centroids, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "num_residues": num_residues,
            # Legacy key kept for existing training/evaluation code.
            "num_peptides": num_residues,
        }

        if self.return_peptide_ids:
            residue_ids_tensor = torch.tensor(residue_ids, dtype=torch.int64)
            sample["residue_ids"] = residue_ids_tensor
            # Legacy key kept for existing training/evaluation code.
            sample["peptide_ids"] = residue_ids_tensor

        sample["label"] = self.get_file_type(idx)
        sample["file_path"] = file_path
        return sample


__all__ = [
    "SyntheticPeptidesDataset",
    "load_residue_centroids_from_gro",
    "load_peptide_centroids_from_gro",
]