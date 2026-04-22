"""Configuration file for synthetic peptide generation."""

# Box parameters
BOX_SIZE = 15.0  # nm (simulation box is 15x15x15)

# Peptide structure parameters
BEADS_PER_PEPTIDE = 4
BEAD_TYPES = ['BB', 'SC1', 'SC2', 'SC3', 'SC4']
RESIDUE_NAMES = [
    'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'PRO', 
    'GLY', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
    'GLU', 'LYS', 'ARG', 'HIS'
]

# Nanotube parameters
NANOTUBE = {
    'num_peptides': 3000,
    'inner_radius': 1.5,
    'outer_radius': 2.5,
    'tube_radius_variance': 0.5,
    'length': 10.0,
    'bend_strength': 0.45,
    'bend_strength_std': 0.2,
    'step_size': 0.2,
    'walk_steps': 5,
    'bead_offset_noise': 0.1,
    'sample_radius_variance': 1.0,
    'sample_length_variance': 2.0,
    'inflection_range': (0, 1),
}

# Nanofiber parameters (solid fiber, no hollow interior)
NANOFIBER = {
    'num_peptides': 3000,
    'fiber_radius': 2.5,
    'fiber_radius_variance': 0.1,
    'length': 10.0,
    'bend_strength': 0.45,
    'bend_strength_std': 0.2,
    'step_size': 0.2,
    'walk_steps': 5,
    'bead_offset_noise': 0.07,
    'sample_radius_variance': 1.0,
    'sample_length_variance': 2.0,
    'inflection_range': (0, 1),
}

# Random aggregate parameters
RANDOM_AGGREGATE = {
    'num_peptides': 1700,
    'step_size': 1.5,
    'noise_level': 0.2
}

# Micelle parameters
MICELLE = {
    'num_peptides': 3000,
    'core_radius': 2.5,
    'shell_thickness': 0.25,
    'radius_variance': 0.2,
    'elongation_factor': 1.0,
    'sample_elongation_variance': 0.5,
    'radial_step_size': 0.07,
    'walk_steps': 3,
    'bead_offset_noise': 0.02,
    'sample_radius_variance': 0.5,
}

# Lamellar sheets parameters
LAMELLAR_SHEETS = {
    'num_peptides': 3000,
    'num_sheets': 3,
    'sheet_gap': 3.0,
    'sheet_width': 9.0,
    'sheet_length': 9.0,
    'distortion_strength': 0.5,
    'sheet_thickness': 0.3,
    'step_size': 0.5,
    'walk_steps': 5,
    'bead_offset_noise': 0.2,
    'sample_gap_variance': 0.4,
    'sample_width_variance': 2.0,
    'sample_length_variance': 2.0,
    'sample_distortion_variance': 0.25,
}

# Dataset generation parameters
DATASET = {
    'output_dir': 'data/synthetic_peptides',
    'num_samples_per_type': 50
}
