import numpy as np
import os
import json
from config import *

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SyntheticPeptideGenerator:
    """Generate synthetic coarse-grained peptide assemblies with GPU support."""
    
    def __init__(self, beads_per_peptide: int = None, beads_per_peptide_range: tuple = None, use_gpu: bool = True):
        if beads_per_peptide_range is not None:
            range_start, range_end = beads_per_peptide_range
            min_beads = max(1, int(min(range_start, range_end)))
            max_beads = max(1, int(max(range_start, range_end)))
            self.beads_per_peptide_range = (min_beads, max_beads)
            self.beads_per_peptide = max_beads
        else:
            fixed_beads = int(beads_per_peptide or BEADS_PER_PEPTIDE)
            self.beads_per_peptide = fixed_beads
            self.beads_per_peptide_range = (fixed_beads, fixed_beads)
        self.bead_types = BEAD_TYPES
        self.residue_names = RESIDUE_NAMES
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

    def _sample_bead_count(self) -> int:
        """Sample the number of beads to use for one peptide."""
        min_beads, max_beads = self.beads_per_peptide_range
        if min_beads == max_beads:
            return min_beads
        return int(np.random.randint(min_beads, max_beads + 1))
    
    def _apply_smooth_bending(self, positions, length, bend_strength=0.3, inflection_range=(0, 2)):
        """Apply smooth bending distortion with controllable inflection points.
        
        Uses polynomial transformations to create realistic C-shaped or S-shaped bends.
        The bending creates non-parallel ends naturally.
        
        Args:
            positions: Nx3 array of positions (z-axis is the main axis)
            length: length of the structure along z-axis
            bend_strength: strength of bending (0.0 = straight, higher = more bent)
            inflection_range: tuple (min, max) for number of inflection points
        
        Returns:
            bent_positions: Nx3 array with smooth bending applied
        """
        if bend_strength <= 0.0:
            return positions
        
        # Randomly pick number of inflections from the specified range
        min_inflections, max_inflections = inflection_range
        num_inflections = np.random.randint(min_inflections, max_inflections + 1)
        
        # Normalize z to [0, 1]
        z_normalized = positions[:, 2] / length
        
        # Random bending in two perpendicular directions for 3D bending
        # This naturally creates non-parallel ends
        bend_dir1 = np.random.randn(2)
        bend_dir1 = bend_dir1 / np.linalg.norm(bend_dir1)
        
        # Second direction for twist/spiral effect
        bend_dir2 = np.array([-bend_dir1[1], bend_dir1[0]])  # Perpendicular
        
        # Bending parameters - scaled by bend_strength
        bend_magnitude = bend_strength * length
        
        # Different bending profiles based on inflection points
        if num_inflections == 0:
            # Simple C-shaped bend: quadratic - ends not parallel
            bend_amount1 = bend_magnitude * z_normalized**2
            bend_amount2 = bend_magnitude * 0.3 * z_normalized  # Linear component for twist
        elif num_inflections == 1:
            # Single inflection: cubic (S-shaped but asymmetric)
            bend_amount1 = bend_magnitude * (3 * z_normalized**2 - 2 * z_normalized**3)
            bend_amount2 = bend_magnitude * 0.3 * (z_normalized - z_normalized**2)
        else:  # 2 or more inflections
            # Multiple inflections: sinusoidal with phase shift
            bend_amount1 = bend_magnitude * 0.5 * np.sin(num_inflections * np.pi * z_normalized)
            bend_amount2 = bend_magnitude * 0.3 * np.sin(num_inflections * np.pi * z_normalized + np.pi/4)
        
        bent_positions = positions.copy()
        # Apply bending in both directions to create non-parallel ends
        bent_positions[:, 0] += bend_amount1 * bend_dir1[0] + bend_amount2 * bend_dir2[0]
        bent_positions[:, 1] += bend_amount1 * bend_dir1[1] + bend_amount2 * bend_dir2[1]
        
        # Apply random rotation to avoid preferential orientation
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        
        # Rotate around z-axis
        x_rot = bent_positions[:, 0] * cos_a - bent_positions[:, 1] * sin_a
        y_rot = bent_positions[:, 0] * sin_a + bent_positions[:, 1] * cos_a
        bent_positions[:, 0] = x_rot
        bent_positions[:, 1] = y_rot
        
        # Additional tilt rotation for more natural 3D orientation
        tilt_axis = np.random.choice(['x', 'y', 'xy'])
        tilt_angle = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
        cos_t, sin_t = np.cos(tilt_angle), np.sin(tilt_angle)
        
        if tilt_axis == 'x':
            # Rotate around x-axis
            y_tilt = bent_positions[:, 1] * cos_t - bent_positions[:, 2] * sin_t
            z_tilt = bent_positions[:, 1] * sin_t + bent_positions[:, 2] * cos_t
            bent_positions[:, 1] = y_tilt
            bent_positions[:, 2] = z_tilt
        elif tilt_axis == 'y':
            # Rotate around y-axis
            x_tilt = bent_positions[:, 0] * cos_t + bent_positions[:, 2] * sin_t
            z_tilt = -bent_positions[:, 0] * sin_t + bent_positions[:, 2] * cos_t
            bent_positions[:, 0] = x_tilt
            bent_positions[:, 2] = z_tilt
        else:  # 'xy' - rotate around both for maximum twist
            # First around x
            y_tilt = bent_positions[:, 1] * cos_t - bent_positions[:, 2] * sin_t
            z_tilt = bent_positions[:, 1] * sin_t + bent_positions[:, 2] * cos_t
            bent_positions[:, 1] = y_tilt
            bent_positions[:, 2] = z_tilt
            # Then around y with different angle
            tilt_angle2 = np.random.uniform(-np.pi/6, np.pi/6)
            cos_t2, sin_t2 = np.cos(tilt_angle2), np.sin(tilt_angle2)
            x_tilt = bent_positions[:, 0] * cos_t2 + bent_positions[:, 2] * sin_t2
            z_tilt = -bent_positions[:, 0] * sin_t2 + bent_positions[:, 2] * cos_t2
            bent_positions[:, 0] = x_tilt
            bent_positions[:, 2] = z_tilt
        
        return bent_positions
    
    def _apply_smooth_elongation(self, positions, elongation_factor=1.0):
        """Apply smooth elongation to spherical structures (micelles).
        
        Args:
            positions: Nx3 array centered at origin
            elongation_factor: 1.0 = sphere, >1.0 = elongated ellipsoid
        
        Returns:
            elongated_positions: Nx3 array
        """
        if elongation_factor <= 1.0:
            return positions
        
        # Random elongation direction
        elongation_axis = np.random.randint(0, 3)  # 0=x, 1=y, 2=z
        
        elongated = positions.copy()
        elongated[:, elongation_axis] *= elongation_factor
        
        # Optional: add smooth sinusoidal perturbation for realism
        if np.random.random() < 0.5:
            # Compute angle around elongation axis
            other_axes = [i for i in range(3) if i != elongation_axis]
            r = np.sqrt(positions[:, other_axes[0]]**2 + positions[:, other_axes[1]]**2)
            perturbation = 0.1 * np.sin(3 * positions[:, elongation_axis]) * r
            elongated[:, other_axes[0]] += perturbation * np.cos(np.arctan2(positions[:, other_axes[1]], positions[:, other_axes[0]]))
            elongated[:, other_axes[1]] += perturbation * np.sin(np.arctan2(positions[:, other_axes[1]], positions[:, other_axes[0]]))
        
        return elongated
    
    def _to_numpy(self, tensor):
        """Convert PyTorch tensor to NumPy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def generate_nanotube(self, num_peptides: int = None, 
                         inner_radius: float = None,
                         outer_radius: float = None,
                         tube_radius_variance: float = None,
                         length: float = None,
                         step_size: float = None,
                         bead_offset_noise: float = None,
                         bend_strength: float = None,
                         bend_strength_std: float = None,
                         inflection_range: tuple = None,
                         sample_radius_variance: float = None,
                         sample_length_variance: float = None) -> np.ndarray:
        """Generate a nanotube as a simple cylinder with smooth bending distortion.
        
        Args:
            bend_strength: Bending strength (0.0 = straight, 0.3 = moderate, 0.5+ = strong bend)
            bend_strength_std: Standard deviation for bend strength variation between samples
            inflection_range: Tuple (min, max) for number of inflection points (e.g., (0, 2))
            sample_radius_variance: Variance for sampling different radius values between samples (default: 0.0)
            sample_length_variance: Variance for sampling different length values between samples (default: 0.0)
        """
        # Use config defaults if not provided
        cfg = NANOTUBE
        num_peptides = num_peptides or cfg['num_peptides']
        inner_radius = inner_radius or cfg['inner_radius']
        outer_radius = outer_radius or cfg['outer_radius']
        tube_radius_variance = tube_radius_variance or cfg['tube_radius_variance']
        length = length or cfg['length']
        step_size = step_size or cfg['step_size']
        bead_offset_noise = bead_offset_noise if bead_offset_noise is not None else cfg.get('bead_offset_noise', 0.01)
        bend_strength = bend_strength if bend_strength is not None else cfg.get('bend_strength', 0.3)
        bend_strength_std = bend_strength_std if bend_strength_std is not None else cfg.get('bend_strength_std', 0.0)
        inflection_range = inflection_range if inflection_range is not None else cfg.get('inflection_range', (0, 2))
        sample_radius_variance = sample_radius_variance if sample_radius_variance is not None else cfg.get('sample_radius_variance', 0.0)
        sample_length_variance = sample_length_variance if sample_length_variance is not None else cfg.get('sample_length_variance', 0.0)
        
        # Apply sample-level variance (creates variation between different samples)
        if sample_radius_variance > 0:
            radius_offset = np.random.normal(0, sample_radius_variance)
            inner_radius = max(0.5, inner_radius + radius_offset)  # Ensure minimum radius
            outer_radius = max(inner_radius + 0.1, outer_radius + radius_offset)  # Ensure outer > inner
        
        if sample_length_variance > 0:
            length = max(1.0, length + np.random.normal(0, sample_length_variance))  # Ensure minimum length
        
        # Apply bend strength variance for this sample
        if bend_strength_std > 0:
            bend_strength = max(0.0, bend_strength + np.random.normal(0, bend_strength_std))
        
        if self.use_gpu:
            return self._generate_nanotube_gpu(num_peptides, inner_radius, outer_radius, 
                                               tube_radius_variance, length, step_size, 
                                               bead_offset_noise, cfg, length, bend_strength, inflection_range)
        else:
            return self._generate_nanotube_cpu(num_peptides, inner_radius, outer_radius, 
                                               tube_radius_variance, length, step_size, 
                                               bead_offset_noise, cfg, length, bend_strength, inflection_range)
    
    def _generate_nanotube_gpu(self, num_peptides, inner_radius, outer_radius, 
                               tube_radius_variance, length, step_size, 
                               bead_offset_noise, cfg, total_length, bend_strength, inflection_range):
        """GPU-accelerated nanotube generation - simple cylinder with smooth bending."""
        # Generate straight cylindrical tube first
        z_positions = torch.rand(num_peptides, device=self.device) * length
        angles = torch.rand(num_peptides, device=self.device) * 2 * np.pi
        
        # Vary radius slightly along the tube for realism
        local_radius_variation = torch.randn(num_peptides, device=self.device) * tube_radius_variance
        inner_r = inner_radius + local_radius_variation
        outer_r = outer_radius + local_radius_variation
        inner_r = torch.clamp(inner_r, min=inner_radius * 0.7)
        outer_r = torch.clamp(outer_r, min=inner_r + 0.3)
        
        # Random radius within tube wall
        r = torch.rand(num_peptides, device=self.device) * (outer_r - inner_r) + inner_r
        
        # Initial positions on straight cylinder
        positions = torch.stack([
            r * torch.cos(angles),
            r * torch.sin(angles),
            z_positions
        ], dim=1)
        
        # Random walk for local variation
        for walk_step in range(cfg.get('walk_steps', 5)):
            steps = torch.randn(num_peptides, 3, device=self.device) * step_size
            new_pos = positions + steps
            
            # Keep z within bounds
            new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0, length)
            
            # Keep radial distance within tube wall
            radial_dist = torch.sqrt(new_pos[:, 0]**2 + new_pos[:, 1]**2)
            
            # Recompute local radius at new z position
            z_frac = new_pos[:, 2] / length
            local_var = torch.randn(num_peptides, device=self.device) * tube_radius_variance * 0.5
            inner_r_local = inner_radius + local_var
            outer_r_local = outer_radius + local_var
            inner_r_local = torch.clamp(inner_r_local, min=inner_radius * 0.7)
            outer_r_local = torch.clamp(outer_r_local, min=inner_r_local + 0.3)
            
            # Fix positions outside tube wall
            mask = (radial_dist < inner_r_local) | (radial_dist > outer_r_local)
            if mask.any():
                angles_new = torch.atan2(new_pos[mask, 1], new_pos[mask, 0])
                r_new = torch.rand(mask.sum(), device=self.device) * (outer_r_local[mask] - inner_r_local[mask]) + inner_r_local[mask]
                new_pos[mask, 0] = r_new * torch.cos(angles_new)
                new_pos[mask, 1] = r_new * torch.sin(angles_new)
            
            positions = new_pos
        
        # Convert to CPU and apply smooth bending
        positions_np = self._to_numpy(positions)
        bent_positions = self._apply_smooth_bending(positions_np, length, bend_strength=bend_strength, inflection_range=inflection_range)
        
        return self._generate_beads(bent_positions, bead_offset_noise)
    
    def _generate_nanotube_cpu(self, num_peptides, inner_radius, outer_radius, 
                               tube_radius_variance, length, step_size, 
                               bead_offset_noise, cfg, total_length, bend_strength, inflection_range):
        """CPU-based nanotube generation - simple cylinder with smooth bending."""
        # Generate straight cylindrical tube first
        positions = []
        
        for i in range(num_peptides):
            # Random z position along tube
            z = np.random.uniform(0, length)
            
            # Vary radius slightly along the tube for realism
            local_radius_variation = np.random.normal(0, tube_radius_variance)
            inner_r = max(inner_radius * 0.7, inner_radius + local_radius_variation)
            outer_r = max(inner_r + 0.3, outer_radius + local_radius_variation)
            
            # Random angle and radius within tube wall
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(inner_r, outer_r)
            
            # Starting position on straight cylinder
            start_pos = np.array([
                r * np.cos(angle),
                r * np.sin(angle),
                z
            ])
            
            # Small random walk around starting position
            for walk_step in range(cfg.get('walk_steps', 5)):
                step = np.random.normal(0, step_size, 3)
                new_pos = start_pos + step
                
                # Keep z within bounds
                new_pos[2] = np.clip(new_pos[2], 0, length)
                
                # Keep radial distance within tube wall
                radial_dist = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                
                # Recompute local radius at new z position
                local_var = np.random.normal(0, tube_radius_variance * 0.5)
                inner_r_local = max(inner_radius * 0.7, inner_radius + local_var)
                outer_r_local = max(inner_r_local + 0.3, outer_radius + local_var)
                
                # Fix positions outside tube wall
                if radial_dist < inner_r_local or radial_dist > outer_r_local:
                    angle_new = np.arctan2(new_pos[1], new_pos[0])
                    r_new = np.random.uniform(inner_r_local, outer_r_local)
                    new_pos[0] = r_new * np.cos(angle_new)
                    new_pos[1] = r_new * np.sin(angle_new)
                
                start_pos = new_pos
            
            positions.append(start_pos)
        
        positions = np.array(positions)
        
        # Apply smooth bending distortion with controllable inflection points
        bent_positions = self._apply_smooth_bending(positions, length, bend_strength=bend_strength, inflection_range=inflection_range)
        
        return self._generate_beads(bent_positions, bead_offset_noise)
    
    def _generate_beads(self, positions, bead_offset_noise):
        """Generate bead data from peptide positions."""
        data = []
        atom_id = 1
        
        for i, current_pos in enumerate(positions):
            residue_name = np.random.choice(self.residue_names)
            residue_id = i + 1

            bead_count = self._sample_bead_count()
            for b in range(bead_count):
                bead_offset = np.random.normal(0, bead_offset_noise, 3)
                bx = current_pos[0] + bead_offset[0]
                by = current_pos[1] + bead_offset[1]
                bz = current_pos[2] + bead_offset[2]
                
                bead_type = self.bead_types[min(b, len(self.bead_types) - 1)]
                data.append([residue_id, atom_id, bead_type, bx, by, bz, residue_name])
                atom_id += 1
        
        arr = np.array(data)
        return self._center_data(arr)

    def _center_data(self, data: np.ndarray) -> np.ndarray:
        """Center coordinates in `data` to BOX_SIZE/2 along all three axes.

        Expects data rows of the form [residue_id, atom_id, bead_type, x, y, z, residue_name].
        """
        if data.size == 0:
            return data

        coords = data[:, 3:6].astype(float)
        centroid = coords.mean(axis=0)
        box_center = np.array([BOX_SIZE / 2.0, BOX_SIZE / 2.0, BOX_SIZE / 2.0])
        shift = box_center - centroid

        coords += shift
        # Clamp to box boundaries to avoid invalid positions
        coords[:, 0] = np.clip(coords[:, 0], 0.1, BOX_SIZE - 0.1)
        coords[:, 1] = np.clip(coords[:, 1], 0.1, BOX_SIZE - 0.1)
        coords[:, 2] = np.clip(coords[:, 2], 0.1, BOX_SIZE - 0.1)

        data[:, 3:6] = coords
        return data
    
    def generate_nanofiber(self, num_peptides: int = None,
                          fiber_radius: float = None,
                          fiber_radius_variance: float = None,
                          length: float = None,
                          step_size: float = None,
                          bead_offset_noise: float = None,
                          bend_strength: float = None,
                          bend_strength_std: float = None,
                          inflection_range: tuple = None,
                          sample_radius_variance: float = None,
                          sample_length_variance: float = None) -> np.ndarray:
        """Generate a solid nanofiber as a simple cylinder with smooth bending.
        
        Args:
            bend_strength: Bending strength (0.0 = straight, 0.3 = moderate, 0.5+ = strong bend)
            bend_strength_std: Standard deviation for bend strength variation between samples
            inflection_range: Tuple (min, max) for number of inflection points (e.g., (0, 2))
            sample_radius_variance: Variance for sampling different radius values between samples (default: 0.0)
            sample_length_variance: Variance for sampling different length values between samples (default: 0.0)
        """
        # Use config defaults if not provided
        cfg = NANOFIBER
        num_peptides = num_peptides or cfg['num_peptides']
        fiber_radius = fiber_radius or cfg['fiber_radius']
        fiber_radius_variance = fiber_radius_variance or cfg['fiber_radius_variance']
        length = length or cfg['length']
        step_size = step_size or cfg['step_size']
        bead_offset_noise = bead_offset_noise if bead_offset_noise is not None else cfg.get('bead_offset_noise', 0.01)
        bend_strength = bend_strength if bend_strength is not None else cfg.get('bend_strength', 0.3)
        bend_strength_std = bend_strength_std if bend_strength_std is not None else cfg.get('bend_strength_std', 0.0)
        inflection_range = inflection_range if inflection_range is not None else cfg.get('inflection_range', (0, 2))
        sample_radius_variance = sample_radius_variance if sample_radius_variance is not None else cfg.get('sample_radius_variance', 0.0)
        sample_length_variance = sample_length_variance if sample_length_variance is not None else cfg.get('sample_length_variance', 0.0)
        
        # Apply sample-level variance (creates variation between different samples)
        if sample_radius_variance > 0:
            fiber_radius = max(0.5, fiber_radius + np.random.normal(0, sample_radius_variance))  # Ensure minimum radius
        
        if sample_length_variance > 0:
            length = max(1.0, length + np.random.normal(0, sample_length_variance))  # Ensure minimum length
        
        # Apply bend strength variance for this sample
        if bend_strength_std > 0:
            bend_strength = max(0.0, bend_strength + np.random.normal(0, bend_strength_std))
        
        if self.use_gpu:
            return self._generate_nanofiber_gpu(num_peptides, fiber_radius, fiber_radius_variance,
                                                length, step_size, bead_offset_noise, cfg, length, bend_strength, inflection_range)
        else:
            return self._generate_nanofiber_cpu(num_peptides, fiber_radius, fiber_radius_variance,
                                                length, step_size, bead_offset_noise, cfg, length, bend_strength, inflection_range)
    
    def _generate_nanofiber_gpu(self, num_peptides, fiber_radius, fiber_radius_variance,
                                length, step_size, bead_offset_noise, cfg, total_length, bend_strength, inflection_range):
        """GPU-accelerated nanofiber generation - simple solid cylinder with smooth bending."""
        # Generate straight solid cylindrical fiber first
        z_positions = torch.rand(num_peptides, device=self.device) * length
        angles = torch.rand(num_peptides, device=self.device) * 2 * np.pi
        
        # Vary radius slightly along the fiber for realism
        local_radius_variation = torch.randn(num_peptides, device=self.device) * fiber_radius_variance
        max_r = fiber_radius + local_radius_variation
        max_r = torch.clamp(max_r, min=fiber_radius * 0.6)
        
        # Random radius from center to edge (solid fiber)
        r = torch.rand(num_peptides, device=self.device) * max_r
        
        # Initial positions on straight cylinder
        positions = torch.stack([
            r * torch.cos(angles),
            r * torch.sin(angles),
            z_positions
        ], dim=1)
        
        # Random walk for local variation
        for walk_step in range(cfg.get('walk_steps', 5)):
            steps = torch.randn(num_peptides, 3, device=self.device) * step_size
            new_pos = positions + steps
            new_pos[:, 2] = torch.clamp(new_pos[:, 2], 0, length)
            
            # Keep radial distance within fiber radius
            radial_dist = torch.sqrt(new_pos[:, 0]**2 + new_pos[:, 1]**2)
            
            # Recompute local radius at new z position
            local_var = torch.randn(num_peptides, device=self.device) * fiber_radius_variance * 0.5
            max_r_local = fiber_radius + local_var
            max_r_local = torch.clamp(max_r_local, min=fiber_radius * 0.6)
            
            # Fix positions outside fiber
            mask = radial_dist > max_r_local
            if mask.any():
                angles_new = torch.atan2(new_pos[mask, 1], new_pos[mask, 0])
                r_new = torch.rand(mask.sum(), device=self.device) * max_r_local[mask]
                new_pos[mask, 0] = r_new * torch.cos(angles_new)
                new_pos[mask, 1] = r_new * torch.sin(angles_new)
            
            positions = new_pos
        
        # Convert to CPU and apply smooth bending
        positions_np = self._to_numpy(positions)
        bent_positions = self._apply_smooth_bending(positions_np, length, bend_strength=bend_strength, inflection_range=inflection_range)
        
        return self._generate_beads(bent_positions, bead_offset_noise)
    
    def _generate_nanofiber_cpu(self, num_peptides, fiber_radius, fiber_radius_variance,
                                length, step_size, bead_offset_noise, cfg, total_length, bend_strength, inflection_range):
        """CPU-based nanofiber generation - simple solid cylinder with smooth bending."""
        # Generate straight solid cylindrical fiber first
        positions = []
        
        for i in range(num_peptides):
            # Random z position along fiber
            z = np.random.uniform(0, length)
            
            # Vary radius slightly along the fiber for realism
            local_radius_variation = np.random.normal(0, fiber_radius_variance)
            max_r = max(fiber_radius * 0.6, fiber_radius + local_radius_variation)
            
            # Random angle and radius from center to edge (solid fiber)
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, max_r)
            
            # Starting position on straight cylinder
            start_pos = np.array([
                r * np.cos(angle),
                r * np.sin(angle),
                z
            ])
            
            # Small random walk around starting position
            for walk_step in range(cfg.get('walk_steps', 5)):
                step = np.random.normal(0, step_size, 3)
                new_pos = start_pos + step
                
                # Keep z within bounds
                new_pos[2] = np.clip(new_pos[2], 0, length)
                
                # Keep radial distance within fiber radius
                radial_dist = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                
                # Recompute local radius at new z position
                local_var = np.random.normal(0, fiber_radius_variance * 0.5)
                max_r_local = max(fiber_radius * 0.6, fiber_radius + local_var)
                
                # Fix positions outside fiber
                if radial_dist > max_r_local:
                    angle_new = np.arctan2(new_pos[1], new_pos[0])
                    r_new = np.random.uniform(0, max_r_local)
                    new_pos[0] = r_new * np.cos(angle_new)
                    new_pos[1] = r_new * np.sin(angle_new)
                
                start_pos = new_pos
            
            positions.append(start_pos)
        
        positions = np.array(positions)
        
        # Apply smooth bending distortion with controllable inflection points
        bent_positions = self._apply_smooth_bending(positions, length, bend_strength=bend_strength, inflection_range=inflection_range)
        
        return self._generate_beads(bent_positions, bead_offset_noise)
    
    def generate_random_aggregate(self, num_peptides: int = None, 
                                 step_size: float = None, 
                                 noise_level: float = None) -> np.ndarray:
        """Generate random aggregate using random walk, constrained within box."""
        cfg = RANDOM_AGGREGATE
        num_peptides = num_peptides or cfg['num_peptides']
        step_size = step_size or cfg['step_size']
        noise_level = noise_level or cfg['noise_level']
        
        if self.use_gpu:
            return self._generate_random_aggregate_gpu(num_peptides, step_size, noise_level)
        else:
            return self._generate_random_aggregate_cpu(num_peptides, step_size, noise_level)
    
    def _generate_random_aggregate_gpu(self, num_peptides, step_size, noise_level):
        """GPU-accelerated random aggregate generation."""
        # Random walk for peptide centers on GPU
        steps = torch.randn(num_peptides, 3, device=self.device) * step_size
        positions = torch.cumsum(steps, dim=0)
        
        # Center the aggregate
        positions -= positions.mean(dim=0)
        
        # Scale to fit within box with margin
        margin = 2.0
        max_extent = torch.abs(positions).max()
        if max_extent > 0:
            scale_factor = min(1.0, (BOX_SIZE / 2 - margin) / max_extent.item())
            positions *= scale_factor
        
        # Shift to box center
        box_center = torch.tensor([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 2], device=self.device)
        positions += box_center
        
        # Convert to CPU for bead generation (which includes random choices)
        positions_np = self._to_numpy(positions)
        
        # Generate beads with orientation
        data = []
        atom_id = 1
        
        for i, center_pos in enumerate(positions_np):
            # Random orientation
            orientation = np.random.normal(0, 1, 3)
            orientation = orientation / np.linalg.norm(orientation)
            
            residue_name = np.random.choice(self.residue_names)
            residue_id = i + 1

            bead_count = self._sample_bead_count()
            for b in range(bead_count):
                noise = np.random.normal(0, noise_level, 3)
                bx = center_pos[0] + b * 0.5 * orientation[0] + noise[0]
                by = center_pos[1] + b * 0.5 * orientation[1] + noise[1]
                bz = center_pos[2] + b * 0.5 * orientation[2] + noise[2]
                
                bx = np.clip(bx, 0.1, BOX_SIZE - 0.1)
                by = np.clip(by, 0.1, BOX_SIZE - 0.1)
                bz = np.clip(bz, 0.1, BOX_SIZE - 0.1)
                
                bead_type = self.bead_types[min(b, len(self.bead_types) - 1)]
                data.append([residue_id, atom_id, bead_type, bx, by, bz, residue_name])
                atom_id += 1
        
        return self._center_data(np.array(data))
    
    def _generate_random_aggregate_cpu(self, num_peptides, step_size, noise_level):
        """CPU-based random aggregate generation."""
        data = []
        atom_id = 1
        
        # Random walk for peptide centers
        steps = np.random.normal(0, step_size, (num_peptides, 3))
        positions = np.cumsum(steps, axis=0)
        
        # Center the aggregate at origin first
        positions -= positions.mean(axis=0)
        
        # Calculate current extent and scale to fit within box with margin
        # Leave margin for peptide length and noise
        margin = 2.0  # nm margin on each side
        max_extent = np.max(np.abs(positions))
        if max_extent > 0:
            scale_factor = min(1.0, (BOX_SIZE / 2 - margin) / max_extent)
            positions *= scale_factor
        
        # Shift to box center
        box_center = np.array([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 2])
        positions += box_center
        
        for i, (cx, cy, cz) in enumerate(positions):
            # Random orientation for each peptide
            orientation = np.random.normal(0, 1, 3)
            orientation = orientation / np.linalg.norm(orientation)
            
            residue_name = np.random.choice(self.residue_names)
            residue_id = i + 1

            bead_count = self._sample_bead_count()
            for b in range(bead_count):
                noise = np.random.normal(0, noise_level, 3)
                
                # Beads along random direction
                bx = cx + b * 0.5 * orientation[0] + noise[0]
                by = cy + b * 0.5 * orientation[1] + noise[1]
                bz = cz + b * 0.5 * orientation[2] + noise[2]
                
                # Clamp coordinates to box boundaries
                bx = np.clip(bx, 0.1, BOX_SIZE - 0.1)
                by = np.clip(by, 0.1, BOX_SIZE - 0.1)
                bz = np.clip(bz, 0.1, BOX_SIZE - 0.1)
                
                bead_type = self.bead_types[min(b, len(self.bead_types) - 1)]
                data.append([residue_id, atom_id, bead_type, bx, by, bz, residue_name])
                atom_id += 1
                
        return self._center_data(np.array(data))
    
    def generate_micelle(self, num_peptides: int = None,
                        core_radius: float = None,
                        shell_thickness: float = None,
                        radius_variance: float = None,
                        elongation_factor: float = None,
                        sample_elongation_variance: float = None,
                        bead_offset_noise: float = None,
                        radial_step_size: float = None,
                        walk_steps: int = None,
                        sample_radius_variance: float = None) -> np.ndarray:
        """Generate a spherical micelle with optional elongation for realism.
        
        Args:
            elongation_factor: Factor to elongate the sphere (1.0 = sphere, >1.0 = elongated)
            sample_elongation_variance: Variance for elongation factor between samples (default: 0.0)
            sample_radius_variance: Variance for sampling different core radius values between samples (default: 0.0)
        """
        cfg = MICELLE
        num_peptides = num_peptides or cfg['num_peptides']
        core_radius = core_radius or cfg['core_radius']
        shell_thickness = shell_thickness or cfg['shell_thickness']
        radius_variance = radius_variance or cfg['radius_variance']
        elongation_factor = elongation_factor if elongation_factor is not None else cfg.get('elongation_factor', 1.0)
        sample_elongation_variance = sample_elongation_variance if sample_elongation_variance is not None else cfg.get('sample_elongation_variance', 0.0)
        bead_offset_noise = bead_offset_noise if bead_offset_noise is not None else cfg.get('bead_offset_noise', 0.05)
        radial_step_size = radial_step_size or cfg.get('radial_step_size', 0.3)
        walk_steps = walk_steps or cfg.get('walk_steps', 3)
        sample_radius_variance = sample_radius_variance if sample_radius_variance is not None else cfg.get('sample_radius_variance', 0.0)
        
        # Apply sample-level variance (creates variation between different samples)
        if sample_radius_variance > 0:
            core_radius = max(0.5, core_radius + np.random.normal(0, sample_radius_variance))  # Ensure minimum radius
        
        # Apply elongation variance for this sample
        if sample_elongation_variance > 0:
            elongation_factor = max(1.0, elongation_factor + np.random.normal(0, sample_elongation_variance))
        
        # Randomly vary elongation for each sample (between 1.0 and specified factor)
        if elongation_factor > 1.0:
            elongation_factor = np.random.uniform(1.0, elongation_factor)
        
        if self.use_gpu:
            return self._generate_micelle_gpu(num_peptides, core_radius, shell_thickness, radius_variance,
                                             elongation_factor, bead_offset_noise,
                                             radial_step_size, walk_steps)
        else:
            return self._generate_micelle_cpu(num_peptides, core_radius, shell_thickness, radius_variance,
                                             elongation_factor, bead_offset_noise,
                                             radial_step_size, walk_steps)
    
    def _generate_micelle_gpu(self, num_peptides, core_radius, shell_thickness, radius_variance,
                              elongation_factor, bead_offset_noise,
                              radial_step_size, walk_steps):
        """GPU-accelerated micelle generation with elongation."""
        # Generate spherical shell
        theta = torch.rand(num_peptides, device=self.device) * 2 * np.pi
        phi = torch.rand(num_peptides, device=self.device) * np.pi
        
        # Random radius within shell
        base_radius = torch.rand(num_peptides, device=self.device) * shell_thickness + core_radius
        base_radius += torch.randn(num_peptides, device=self.device) * radius_variance
        base_radius = torch.clamp(base_radius, min=core_radius * 0.8)
        
        # Starting positions on sphere
        positions = torch.stack([
            base_radius * torch.sin(phi) * torch.cos(theta),
            base_radius * torch.sin(phi) * torch.sin(theta),
            base_radius * torch.cos(phi)
        ], dim=1)
        
        # Random walk
        for walk_step in range(walk_steps):
            steps = torch.randn(num_peptides, 3, device=self.device) * radial_step_size
            new_pos = positions + steps
            
            # Keep within reasonable distance
            distance_from_center = torch.norm(new_pos, dim=1)
            
            # Push back if too close to center
            mask_close = distance_from_center < core_radius * 0.5
            if mask_close.any():
                new_pos[mask_close] = new_pos[mask_close] / distance_from_center[mask_close].unsqueeze(1) * (core_radius * 0.8)
            
            # Pull back if too far
            mask_far = distance_from_center > (core_radius + shell_thickness) * 1.2
            if mask_far.any():
                new_pos[mask_far] = new_pos[mask_far] / distance_from_center[mask_far].unsqueeze(1) * (core_radius + shell_thickness)
            
            positions = new_pos
        
        # Convert to CPU and apply elongation
        positions_np = self._to_numpy(positions)
        elongated_positions = self._apply_smooth_elongation(positions_np, elongation_factor)
        
        # Shift to box center
        box_center_np = np.array([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 2])
        elongated_positions += box_center_np
        
        # Generate beads with radial orientations
        data = []
        atom_id = 1
        
        for i, current_pos in enumerate(elongated_positions):
            # Orientation pointing radially outward from center
            radial_vector = current_pos - box_center_np
            if np.linalg.norm(radial_vector) > 0:
                orientation = radial_vector / np.linalg.norm(radial_vector)
            else:
                orientation = np.array([1, 0, 0])
            
            orientation += np.random.normal(0, 0.3, 3)
            if np.linalg.norm(orientation) > 0:
                orientation = orientation / np.linalg.norm(orientation)
            
            residue_name = np.random.choice(self.residue_names)
            residue_id = i + 1

            bead_count = self._sample_bead_count()
            for b in range(bead_count):
                bead_offset = np.random.normal(0, bead_offset_noise, 3)
                bx = current_pos[0] + b * 0.4 * orientation[0] + bead_offset[0]
                by = current_pos[1] + b * 0.4 * orientation[1] + bead_offset[1]
                bz = current_pos[2] + b * 0.4 * orientation[2] + bead_offset[2]
                
                bx = np.clip(bx, 0.1, BOX_SIZE - 0.1)
                by = np.clip(by, 0.1, BOX_SIZE - 0.1)
                bz = np.clip(bz, 0.1, BOX_SIZE - 0.1)
                
                bead_type = self.bead_types[min(b, len(self.bead_types) - 1)]
                data.append([residue_id, atom_id, bead_type, bx, by, bz, residue_name])
                atom_id += 1
        
        return self._center_data(np.array(data))
    
    def _generate_micelle_cpu(self, num_peptides, core_radius, shell_thickness, radius_variance,
                              elongation_factor, bead_offset_noise,
                              radial_step_size, walk_steps):
        """CPU-based micelle generation with elongation."""
        # Generate peptide positions on spherical shell
        positions = []
        
        for i in range(num_peptides):
            # Random spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            # Random radius within shell (core_radius to core_radius + shell_thickness)
            base_radius = np.random.uniform(core_radius, core_radius + shell_thickness)
            base_radius += np.random.normal(0, radius_variance)
            base_radius = max(core_radius * 0.8, base_radius)  # Ensure minimum radius
            
            # Starting position on sphere
            start_pos = np.array([
                base_radius * np.sin(phi) * np.cos(theta),
                base_radius * np.sin(phi) * np.sin(theta),
                base_radius * np.cos(phi)
            ])
            
            # Small random walk around starting position
            for walk_step in range(walk_steps):
                step = np.random.normal(0, radial_step_size, 3)
                new_pos = start_pos + step
                
                # Keep within reasonable distance from center
                distance_from_center = np.linalg.norm(new_pos)
                if distance_from_center < core_radius * 0.5:
                    # Push back toward core surface
                    new_pos = new_pos / distance_from_center * core_radius * 0.8
                elif distance_from_center > (core_radius + shell_thickness) * 1.2:
                    # Pull back toward shell
                    new_pos = new_pos / distance_from_center * (core_radius + shell_thickness)
                
                start_pos = new_pos
            
            positions.append(start_pos)
        
        positions = np.array(positions)
        
        # Apply smooth elongation
        elongated_positions = self._apply_smooth_elongation(positions, elongation_factor)
        
        # Shift to box center
        box_center = np.array([BOX_SIZE / 2, BOX_SIZE / 2, BOX_SIZE / 2])
        elongated_positions += box_center
        
        # Generate peptides at calculated positions
        data = []
        atom_id = 1
        
        for i, current_pos in enumerate(elongated_positions):
            # Orientation pointing radially outward from center
            radial_vector = current_pos - box_center
            if np.linalg.norm(radial_vector) > 0:
                orientation = radial_vector / np.linalg.norm(radial_vector)
            else:
                orientation = np.array([1, 0, 0])
            
            # Add some randomness to orientation
            orientation += np.random.normal(0, 0.3, 3)
            if np.linalg.norm(orientation) > 0:
                orientation = orientation / np.linalg.norm(orientation)
            
            residue_name = np.random.choice(self.residue_names)
            residue_id = i + 1

            bead_count = self._sample_bead_count()
            for b in range(bead_count):
                bead_offset = np.random.normal(0, bead_offset_noise, 3)
                bx = current_pos[0] + b * 0.4 * orientation[0] + bead_offset[0]
                by = current_pos[1] + b * 0.4 * orientation[1] + bead_offset[1]
                bz = current_pos[2] + b * 0.4 * orientation[2] + bead_offset[2]
                
                # Clamp coordinates to box boundaries
                bx = np.clip(bx, 0.1, BOX_SIZE - 0.1)
                by = np.clip(by, 0.1, BOX_SIZE - 0.1)
                bz = np.clip(bz, 0.1, BOX_SIZE - 0.1)
                
                bead_type = self.bead_types[min(b, len(self.bead_types) - 1)]
                data.append([residue_id, atom_id, bead_type, bx, by, bz, residue_name])
                atom_id += 1
        
        return self._center_data(np.array(data))

    def _sample_plane_basis(self):
        """Sample an orthonormal basis (u, v, n) for a random plane in 3D."""
        n = np.random.normal(0, 1, 3)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / n_norm

        # Pick a stable helper vector not parallel to n.
        helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(n, helper)
        u_norm = np.linalg.norm(u)
        if u_norm == 0:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = u / u_norm
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return u, v, n

    def generate_lamellar_sheets(self, num_peptides: int = None,
                                 num_sheets: int = None,
                                 sheet_gap: float = None,
                                 sheet_width: float = None,
                                 sheet_length: float = None,
                                 distortion_strength: float = None,
                                 step_size: float = None,
                                 walk_steps: int = None,
                                 bead_offset_noise: float = None,
                                 sample_gap_variance: float = None,
                                 sample_width_variance: float = None,
                                 sample_length_variance: float = None,
                                 sample_distortion_variance: float = None,
                                 sheet_thickness: float = None) -> np.ndarray:
        """Generate non-overlapping lamellar sheets from sampled hyperplanes.

        Workflow:
        1) Build a random abstract hyperplane orientation.
        2) Place multiple parallel sheets with fixed gap along the hyperplane normal.
        3) Fill each sheet using random walk constrained to in-plane bounds.
        4) Add slight bounded distortions so sheets are not perfectly flat.
        """
        cfg = LAMELLAR_SHEETS
        num_peptides = num_peptides or cfg['num_peptides']
        num_sheets = max(1, int(num_sheets or cfg.get('num_sheets', 3)))
        sheet_gap = sheet_gap if sheet_gap is not None else cfg['sheet_gap']
        sheet_width = sheet_width if sheet_width is not None else cfg['sheet_width']
        sheet_length = sheet_length if sheet_length is not None else cfg['sheet_length']
        distortion_strength = distortion_strength if distortion_strength is not None else cfg['distortion_strength']
        sample_distortion_variance = sample_distortion_variance if sample_distortion_variance is not None else cfg.get('sample_distortion_variance', 0.0)
        step_size = step_size if step_size is not None else cfg['step_size']
        walk_steps = int(walk_steps if walk_steps is not None else cfg.get('walk_steps', 5))
        bead_offset_noise = bead_offset_noise if bead_offset_noise is not None else cfg.get('bead_offset_noise', 0.03)
        sheet_thickness = sheet_thickness if sheet_thickness is not None else cfg.get('sheet_thickness', 0.08)
        sample_gap_variance = sample_gap_variance if sample_gap_variance is not None else cfg.get('sample_gap_variance', 0.0)
        sample_width_variance = sample_width_variance if sample_width_variance is not None else cfg.get('sample_width_variance', 0.0)
        sample_length_variance = sample_length_variance if sample_length_variance is not None else cfg.get('sample_length_variance', 0.0)

        # Sample-level variation similar to nanotube/nanofiber pattern.
        if sample_gap_variance > 0:
            sheet_gap += np.random.normal(0, sample_gap_variance)
        if sample_width_variance > 0:
            sheet_width += np.random.normal(0, sample_width_variance)
        if sample_length_variance > 0:
            sheet_length += np.random.normal(0, sample_length_variance)
        if sample_distortion_variance > 0:
            distortion_strength = max(0.0, distortion_strength + np.random.normal(0, sample_distortion_variance))

        # Keep geometry valid and ensure sheets do not overlap.
        sheet_width = max(1.0, sheet_width)
        sheet_length = max(1.0, sheet_length)
        min_gap_for_separation = max(0.6, 2.5 * sheet_thickness + 2.0 * bead_offset_noise)
        sheet_gap = max(min_gap_for_separation, sheet_gap)

        if self.use_gpu:
            return self._generate_lamellar_sheets_gpu(
                num_peptides=num_peptides,
                num_sheets=num_sheets,
                sheet_gap=sheet_gap,
                sheet_width=sheet_width,
                sheet_length=sheet_length,
                distortion_strength=distortion_strength,
                step_size=step_size,
                walk_steps=walk_steps,
                bead_offset_noise=bead_offset_noise,
                sheet_thickness=sheet_thickness,
            )
        return self._generate_lamellar_sheets_cpu(
            num_peptides=num_peptides,
            num_sheets=num_sheets,
            sheet_gap=sheet_gap,
            sheet_width=sheet_width,
            sheet_length=sheet_length,
            distortion_strength=distortion_strength,
            step_size=step_size,
            walk_steps=walk_steps,
            bead_offset_noise=bead_offset_noise,
            sheet_thickness=sheet_thickness,
        )

    def _distort_sheet_positions(self, positions, plane_u, plane_v, plane_n,
                                 sheet_gap, distortion_strength, sheet_thickness,
                                 sheet_ids=None):
        """Apply bounded smooth distortion while preserving inter-sheet separation.

        When sheet_ids is provided each sheet receives its own independent random
        wave parameters so the sheets distort non-parallelly.
        """
        if distortion_strength <= 0:
            return positions

        # Limit normal displacement so neighboring sheets remain non-overlapping.
        max_safe_amplitude = max(0.0, 0.5 * sheet_gap - 1.2 * sheet_thickness)
        amplitude = min(distortion_strength, max_safe_amplitude)
        in_plane_jitter = 0.2 * amplitude

        distorted = positions.copy()

        if sheet_ids is not None:
            # Independent distortion per sheet.
            for sid in np.unique(sheet_ids):
                mask = sheet_ids == sid
                pts = positions[mask]
                u_coord = pts @ plane_u
                v_coord = pts @ plane_v

                freq_u = np.random.uniform(0.35, 0.9)
                freq_v = np.random.uniform(0.35, 0.9)
                phase_u = np.random.uniform(0, 2 * np.pi)
                phase_v = np.random.uniform(0, 2 * np.pi)

                wave = (
                    np.sin(freq_u * u_coord + phase_u) + 0.6 * np.cos(freq_v * v_coord + phase_v)
                )
                distorted[mask] += (amplitude * (wave / 1.6))[:, None] * plane_n

                if in_plane_jitter > 0:
                    distorted[mask] += (in_plane_jitter * np.sin(0.7 * v_coord + phase_u))[:, None] * plane_u
                    distorted[mask] += (in_plane_jitter * np.cos(0.7 * u_coord + phase_v))[:, None] * plane_v
        else:
            # Fallback: single shared wave (original behaviour).
            u_coord = positions @ plane_u
            v_coord = positions @ plane_v

            freq_u = np.random.uniform(0.35, 0.9)
            freq_v = np.random.uniform(0.35, 0.9)
            phase_u = np.random.uniform(0, 2 * np.pi)
            phase_v = np.random.uniform(0, 2 * np.pi)

            wave = (
                np.sin(freq_u * u_coord + phase_u) + 0.6 * np.cos(freq_v * v_coord + phase_v)
            )
            distorted += (amplitude * (wave / 1.6))[:, None] * plane_n

            if in_plane_jitter > 0:
                distorted += (in_plane_jitter * np.sin(0.7 * v_coord + phase_u))[:, None] * plane_u
                distorted += (in_plane_jitter * np.cos(0.7 * u_coord + phase_v))[:, None] * plane_v

        return distorted

    def _generate_lamellar_sheets_gpu(self, num_peptides, num_sheets, sheet_gap,
                                      sheet_width, sheet_length, distortion_strength,
                                      step_size, walk_steps, bead_offset_noise, sheet_thickness):
        """GPU-assisted lamellar sheet generation (sampling/walk on GPU, distortion on CPU)."""
        plane_u, plane_v, plane_n = self._sample_plane_basis()
        plane_u_t = torch.tensor(plane_u, dtype=torch.float32, device=self.device)
        plane_v_t = torch.tensor(plane_v, dtype=torch.float32, device=self.device)
        plane_n_t = torch.tensor(plane_n, dtype=torch.float32, device=self.device)

        # Balanced assignment of peptides across sheets.
        sheet_ids = torch.arange(num_peptides, device=self.device) % num_sheets
        sheet_center_offsets = (sheet_ids.float() - (num_sheets - 1) / 2.0) * sheet_gap

        u_coord = (torch.rand(num_peptides, device=self.device) - 0.5) * sheet_width
        v_coord = (torch.rand(num_peptides, device=self.device) - 0.5) * sheet_length
        n_coord = torch.randn(num_peptides, device=self.device) * sheet_thickness

        for _ in range(walk_steps):
            u_coord = torch.clamp(u_coord + torch.randn(num_peptides, device=self.device) * step_size,
                                  -sheet_width / 2.0, sheet_width / 2.0)
            v_coord = torch.clamp(v_coord + torch.randn(num_peptides, device=self.device) * step_size,
                                  -sheet_length / 2.0, sheet_length / 2.0)
            n_coord = torch.clamp(n_coord + torch.randn(num_peptides, device=self.device) * (0.25 * step_size),
                                  -sheet_thickness, sheet_thickness)

        n_total = sheet_center_offsets + n_coord
        positions = (
            u_coord.unsqueeze(1) * plane_u_t +
            v_coord.unsqueeze(1) * plane_v_t +
            n_total.unsqueeze(1) * plane_n_t
        )

        positions_np = self._to_numpy(positions)
        sheet_ids_np = self._to_numpy(sheet_ids)
        positions_np = self._distort_sheet_positions(
            positions_np, plane_u, plane_v, plane_n, sheet_gap, distortion_strength, sheet_thickness,
            sheet_ids=sheet_ids_np,
        )
        return self._generate_beads(positions_np, bead_offset_noise)

    def _generate_lamellar_sheets_cpu(self, num_peptides, num_sheets, sheet_gap,
                                      sheet_width, sheet_length, distortion_strength,
                                      step_size, walk_steps, bead_offset_noise, sheet_thickness):
        """CPU lamellar sheet generation with constrained random walk on each sheet."""
        plane_u, plane_v, plane_n = self._sample_plane_basis()

        positions = np.zeros((num_peptides, 3), dtype=float)
        for i in range(num_peptides):
            sheet_id = i % num_sheets
            sheet_center = (sheet_id - (num_sheets - 1) / 2.0) * sheet_gap

            u_coord = np.random.uniform(-sheet_width / 2.0, sheet_width / 2.0)
            v_coord = np.random.uniform(-sheet_length / 2.0, sheet_length / 2.0)
            n_coord = np.random.normal(0, sheet_thickness)

            for _ in range(walk_steps):
                u_coord = np.clip(u_coord + np.random.normal(0, step_size),
                                  -sheet_width / 2.0, sheet_width / 2.0)
                v_coord = np.clip(v_coord + np.random.normal(0, step_size),
                                  -sheet_length / 2.0, sheet_length / 2.0)
                n_coord = np.clip(n_coord + np.random.normal(0, 0.25 * step_size),
                                  -sheet_thickness, sheet_thickness)

            n_total = sheet_center + n_coord
            positions[i] = u_coord * plane_u + v_coord * plane_v + n_total * plane_n

        sheet_ids_np = np.array([i % num_sheets for i in range(num_peptides)])
        positions = self._distort_sheet_positions(
            positions, plane_u, plane_v, plane_n, sheet_gap, distortion_strength, sheet_thickness,
            sheet_ids=sheet_ids_np,
        )
        return self._generate_beads(positions, bead_offset_noise)
    
    def save_to_gro(self, data: np.ndarray, filename: str, title: str = "Synthetic peptide"):
        """Save data in GROMACS .gro format."""
        with open(filename, "w") as f:
            f.write(f"{title}\n")
            f.write(f"{len(data):5d}\n")
            
            for row in data:
                residue_id, atom_id, bead_type, x, y, z, residue_name = row
                residue_id = int(residue_id)
                atom_id = int(atom_id)
                bead_type = str(bead_type)
                residue_name = str(residue_name)
                x, y, z = float(x), float(y), float(z)
                
                f.write(f"{residue_id:5d}{residue_name:>3s}{bead_type:>7s}{atom_id:5d}"
                       f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
            
            f.write(f"{BOX_SIZE:10.5f}{BOX_SIZE:10.5f}{BOX_SIZE:10.5f}\n")
    
    def save_metadata(self, structure_info: dict, filename: str):
        """Save metadata about generated structures."""
        with open(filename, 'w') as f:
            json.dump(structure_info, f, indent=2)

def generate_dataset(output_dir: str = None, 
                    num_samples_per_type: int = None):
    """Generate synthetic peptide dataset in .gro format."""
    output_dir = output_dir or DATASET['output_dir']
    num_samples_per_type = num_samples_per_type or DATASET['num_samples_per_type']
    
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticPeptideGenerator()
    
    # Structure types: nanotube, nanofiber, random_aggregate, micelle, and lamellar sheets
    structure_types = {
        'nanotube': {
            'func': generator.generate_nanotube,
            'params': {}  # Uses config defaults
        },
        'nanofiber': {
            'func': generator.generate_nanofiber,
            'params': {}  # Uses config defaults
        },
        'random_aggregate': {
            'func': generator.generate_random_aggregate,
            'params': {}  # Uses config defaults
        },
        'micelle': {
            'func': generator.generate_micelle,
            'params': {}  # Uses config defaults
        },
        'lamellar_sheets': {
            'func': generator.generate_lamellar_sheets,
            'params': {}  # Uses config defaults
        }
    }
    
    dataset_info = {
        'description': 'Synthetic coarse-grained peptide assemblies',
        'beads_per_peptide_range': list(generator.beads_per_peptide_range),
        'structures': {},
        'total_samples': 0
    }
    
    print("Generating synthetic peptide dataset...")
    
    for structure_type, config in structure_types.items():
        print(f"Generating {num_samples_per_type} {structure_type} samples...")
        
        # Create separate directory for this structure type
        if structure_type == 'nanotube':
            structure_dir = f"{output_dir}/nanotubes"
        elif structure_type == 'nanofiber':
            structure_dir = f"{output_dir}/nanofibers"
        elif structure_type == 'random_aggregate':
            structure_dir = f"{output_dir}/random_aggregates"
        elif structure_type == 'micelle':
            structure_dir = f"{output_dir}/micelles"
        elif structure_type == 'lamellar_sheets':
            structure_dir = f"{output_dir}/lamellar_sheets"
        else:
            structure_dir = f"{output_dir}/{structure_type}s"
        
        os.makedirs(structure_dir, exist_ok=True)
        
        structure_info = {
            'count': num_samples_per_type,
            'parameters': config['params'],
            'files': []
        }
        
        for i in range(num_samples_per_type):
            # Generate structure (uses config defaults)
            data = config['func']()
            
            # Save as .gro file
            base_name = f"{structure_type}_{i:03d}"
            gro_file = f"{structure_dir}/{base_name}.gro"
            
            generator.save_to_gro(data, gro_file, f"Synthetic {structure_type} #{i}")
            
            structure_info['files'].append({
                'id': i,
                'gro_file': gro_file,
                'num_atoms': len(data)
            })
        
        dataset_info['structures'][structure_type] = structure_info
        dataset_info['total_samples'] += num_samples_per_type
    
    # Save dataset metadata
    generator.save_metadata(dataset_info, f"{output_dir}/dataset_info.json")
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Structure types: {list(structure_types.keys())}")
    
    return dataset_info

if __name__ == "__main__":
    dataset_info = generate_dataset(
        output_dir="data/synthetic_peptides", 
        num_samples_per_type=2
    )
    
    print(f"\nDataset summary:")
    for structure_type, info in dataset_info['structures'].items():
        avg_atoms = np.mean([f['num_atoms'] for f in info['files']])
        print(f"  {structure_type}: {info['count']} samples, avg {avg_atoms:.0f} atoms")