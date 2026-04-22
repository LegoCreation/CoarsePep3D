"""Interactive visualization of synthetic peptides using Streamlit - Refactored."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from config import *
from generate_synthetic_peptides import SyntheticPeptideGenerator, TORCH_AVAILABLE
import json
import os

st.set_page_config(layout="wide", page_title="Peptide Structure Visualizer")
st.title("Synthetic Peptide Structure Visualizer")

# Check if GPU is available
if TORCH_AVAILABLE:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
else:
    GPU_AVAILABLE = False

# Initialize session state
if 'auto_generate' not in st.session_state:
    st.session_state.auto_generate = False

# Structure configuration mapping (without generator functions)
STRUCTURE_CONFIGS = {
    "Nanotube": {
        "config": NANOTUBE,
        "method": "generate_nanotube",
        "params_def": [
            ("Geometry", [
                ("Number of Peptides", 10, 5000, 'num_peptides', 100),
                ("Inner Radius (nm)", 0.5, BOX_SIZE, 'inner_radius', 0.01),
                ("Outer Radius (nm)", lambda p: p['inner_radius'], BOX_SIZE, 'outer_radius', 0.01),
                ("Radius Variance", 0.0, 2.0, 'tube_radius_variance', 0.01),
                ("Length (nm)", 1.0, 20.0, 'length', 0.5),
            ]),
            ("Distortion", [
                ("Bend Strength", 0.0, 1.0, 'bend_strength', 0.05),
                ("Bend Strength Std", 0.0, 0.5, 'bend_strength_std', 0.01),
                ("Inflection Min", 0, 5, 'inflection_min', 1),
                ("Inflection Max", 0, 5, 'inflection_max', 1),
            ]),
            ("Random Walk", [
                ("Step Size", 0.0, 2.0, 'step_size', 0.01),
                ("Walk Steps", 0, 50, 'walk_steps', 1),
            ]),
            ("Noise", [
                ("Bead Offset Noise", 0.0, 0.5, 'bead_offset_noise', 0.01),
            ]),
            ("Sample Variance", [
                ("Sample Radius Variance", 0.0, 2.0, 'sample_radius_variance', 0.01),
                ("Sample Length Variance", 0.0, 5.0, 'sample_length_variance', 0.1),
            ])
        ]
    },
    "Nanofiber": {
        "config": NANOFIBER,
        "method": "generate_nanofiber",
        "params_def": [
            ("Geometry", [
                ("Number of Peptides", 10, 5000, 'num_peptides', 100),
                ("Fiber Radius (nm)", 0.5, BOX_SIZE, 'fiber_radius', 0.01),
                ("Radius Variance", 0.0, 2.0, 'fiber_radius_variance', 0.01),
                ("Length (nm)", 1.0, 20.0, 'length', 0.5),
            ]),
            ("Distortion", [
                ("Bend Strength", 0.0, 1.0, 'bend_strength', 0.05),
                ("Bend Strength Std", 0.0, 0.5, 'bend_strength_std', 0.01),
                ("Inflection Min", 0, 5, 'inflection_min', 1),
                ("Inflection Max", 0, 5, 'inflection_max', 1),
            ]),
            ("Random Walk", [
                ("Step Size", 0.0, 2.0, 'step_size', 0.01),
                ("Walk Steps", 0, 50, 'walk_steps', 1),
            ]),
            ("Noise", [
                ("Bead Offset Noise", 0.0, 0.5, 'bead_offset_noise', 0.01),
            ]),
            ("Sample Variance", [
                ("Sample Radius Variance", 0.0, 2.0, 'sample_radius_variance', 0.01),
                ("Sample Length Variance", 0.0, 5.0, 'sample_length_variance', 0.1),
            ])
        ]
    },
    "Random Aggregate": {
        "config": RANDOM_AGGREGATE,
        "method": "generate_random_aggregate",
        "params_def": [
            ("Parameters", [
                ("Number of Peptides", 500, 3000, 'num_peptides', 100),
                ("Step Size", 0.5, 3.0, 'step_size', 0.1),
                ("Noise Level", 0.0, 0.5, 'noise_level', 0.01),
            ])
        ]
    },
    "Micelle": {
        "config": MICELLE,
        "method": "generate_micelle",
        "params_def": [
            ("Geometry", [
                ("Number of Peptides", 500, 3000, 'num_peptides', 100),
                ("Core Radius (nm)", 1.0, 5.0, 'core_radius', 0.1),
                ("Shell Thickness (nm)", 0.0, 3.0, 'shell_thickness', 0.01),
                ("Radius Variance", 0.0, 0.5, 'radius_variance', 0.01),
            ]),
            ("Elongation", [
                ("Elongation Factor", 1.0, 3.0, 'elongation_factor', 0.1),
                ("Sample Elongation Variance", 0.0, 1.0, 'sample_elongation_variance', 0.05),
            ]),
            ("Random Walk", [
                ("Radial Step Size", 0.0, 1.0, 'radial_step_size', 0.01),
                ("Walk Steps", 0, 10, 'walk_steps', 1),
            ]),
            ("Noise", [
                ("Bead Offset Noise", 0.0, 0.2, 'bead_offset_noise', 0.01),
            ]),
            ("Sample Variance", [
                ("Sample Radius Variance", 0.0, 2.0, 'sample_radius_variance', 0.01),
            ])
        ]
    },
    "Lamellar Sheets": {
        "config": LAMELLAR_SHEETS,
        "method": "generate_lamellar_sheets",
        "params_def": [
            ("Geometry", [
                ("Number of Peptides", 100, 6000, 'num_peptides', 100),
                ("Number of Sheets", 1, 12, 'num_sheets', 1),
                ("Sheet Gap (nm)", 0.5, 6.0, 'sheet_gap', 0.05),
                ("Sheet Width (nm)", 1.0, BOX_SIZE, 'sheet_width', 0.1),
                ("Sheet Length (nm)", 1.0, BOX_SIZE, 'sheet_length', 0.1),
            ]),
            ("Distortion", [
                ("Distortion Strength", 0.0, 1.5, 'distortion_strength', 0.01),
                ("Sheet Thickness", 0.01, 0.5, 'sheet_thickness', 0.01),
            ]),
            ("Random Walk", [
                ("Step Size", 0.0, 1.0, 'step_size', 0.01),
                ("Walk Steps", 0, 20, 'walk_steps', 1),
            ]),
            ("Noise", [
                ("Bead Offset Noise", 0.0, 1.0, 'bead_offset_noise', 0.01),
            ]),
            ("Sample Variance", [
                ("Sample Gap Variance", 0.0, 2.0, 'sample_gap_variance', 0.01),
                ("Sample Width Variance", 0.0, 3.0, 'sample_width_variance', 0.01),
                ("Sample Length Variance", 0.0, 3.0, 'sample_length_variance', 0.01),
                ("Sample Distortion Variance", 0.0, 0.5, 'sample_distortion_variance', 0.01),
            ])
        ]
    }
}

def update_config_dict(config_dict, params):
    """Update config dictionary with parameters."""
    for key, value in params.items():
        if key in config_dict:
            config_dict[key] = value

# Define accepted parameters for each generation method
METHOD_PARAMS = {
    "generate_nanotube": {
        'num_peptides', 'inner_radius', 'outer_radius', 'tube_radius_variance',
        'length', 'step_size', 'bead_offset_noise', 'bend_strength', 'bend_strength_std',
        'inflection_range', 'sample_radius_variance', 'sample_length_variance'
    },
    "generate_nanofiber": {
        'num_peptides', 'fiber_radius', 'fiber_radius_variance',
        'length', 'step_size', 'bead_offset_noise', 'bend_strength', 'bend_strength_std',
        'inflection_range', 'sample_radius_variance', 'sample_length_variance'
    },
    "generate_random_aggregate": {
        'num_peptides', 'step_size', 'noise_level'
    },
    "generate_micelle": {
        'num_peptides', 'core_radius', 'shell_thickness', 'radius_variance',
        'elongation_factor', 'sample_elongation_variance', 'bead_offset_noise',
        'radial_step_size', 'walk_steps', 'sample_radius_variance'
    },
    "generate_lamellar_sheets": {
        'num_peptides', 'num_sheets', 'sheet_gap', 'sheet_width', 'sheet_length',
        'distortion_strength', 'step_size', 'walk_steps', 'bead_offset_noise',
        'sample_gap_variance', 'sample_width_variance', 'sample_length_variance',
        'sample_distortion_variance', 'sheet_thickness'
    }
}

def generate_structure(structure_type, params, generator):
    """Generate structure with given parameters using the provided generator."""
    config = STRUCTURE_CONFIGS[structure_type]
    update_config_dict(config["config"], params)
    method = getattr(generator, config["method"])
    
    # Filter params to only include those accepted by the method
    method_name = config["method"]
    accepted_params = METHOD_PARAMS.get(method_name, set())
    filtered_params = {k: v for k, v in params.items() if k in accepted_params}
    
    # Pass filtered parameters as keyword arguments to the generation method
    return method(**filtered_params)

def render_parameter_sliders(structure_type):
    """Render parameter sliders for a structure type and return params dict."""
    config = STRUCTURE_CONFIGS[structure_type]
    params = {}
    
    for section_name, sliders in config["params_def"]:
        st.sidebar.markdown(f"**{section_name}**")
        
        for slider_def in sliders:
            label, min_val, max_val, param_key, step = slider_def
            
            # Handle dynamic min value (e.g., outer_radius depends on inner_radius)
            if callable(min_val):
                min_val = min_val(params)
            
            default_val = config["config"].get(param_key, min_val)
            
            # Handle inflection_range specially
            if param_key == 'inflection_min':
                default_val = config["config"].get('inflection_range', (0, 2))[0]
            elif param_key == 'inflection_max':
                default_val = config["config"].get('inflection_range', (0, 2))[1]

            if param_key == 'num_peptides':
                params[param_key] = st.sidebar.number_input(
                    label,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    step=int(step),
                )
            else:
                params[param_key] = st.sidebar.slider(label, min_val, max_val, default_val, step)
    
    # Combine inflection_min and inflection_max into inflection_range tuple
    if 'inflection_min' in params and 'inflection_max' in params:
        # Ensure min <= max
        infl_min = min(params['inflection_min'], params['inflection_max'])
        infl_max = max(params['inflection_min'], params['inflection_max'])
        params['inflection_range'] = (infl_min, infl_max)
        # Remove the individual parameters
        del params['inflection_min']
        del params['inflection_max']
    
    return params

def render_dataset_generation(structure_type, current_params, generator):
    """Render dataset generation section with checkboxes and sample count inputs."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Generate Dataset")
    
    st.sidebar.markdown("**Structure Types to Generate:**")
    
    # Create checkboxes and number inputs for all structure types
    structure_keys = {
        "Nanotube": f"gen_nanotube_{structure_type[:2].lower()}",
        "Nanofiber": f"gen_nanofiber_{structure_type[:2].lower()}",
        "Random Aggregate": f"gen_aggregate_{structure_type[:2].lower()}",
        "Micelle": f"gen_micelle_{structure_type[:2].lower()}",
        "Lamellar Sheets": f"gen_lamellar_{structure_type[:2].lower()}"
    }
    
    selected_structures = {}
    num_samples_per_type = {}
    
    for struct_name, key in structure_keys.items():
        # Create two columns for checkbox and number input
        col1, col2 = st.sidebar.columns([2, 1])
        
        with col1:
            if struct_name in ["Nanotube", "Nanofiber", "Micelle"]:
                display_name = struct_name + "s"
            else:
                display_name = struct_name
            selected_structures[struct_name] = st.checkbox(
                display_name,
                value=True,
                key=key
            )
        
        with col2:
            num_samples_per_type[struct_name] = st.number_input(
                "Samples",
                min_value=1,
                max_value=100000,
                value=2,
                step=1,
                label_visibility="collapsed",
                key=f"num_samples_{struct_name.replace(' ', '_').lower()}_{structure_type[:2].lower()}"
            )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="data/synthetic_peptides",
        help="Directory where generated files will be saved",
        key=f"output_dir_{structure_type[:2].lower()}"
    )
    
    if st.sidebar.button("Generate Dataset", type="secondary", key=f"gen_dataset_{structure_type[:2].lower()}"):
        if not any(selected_structures.values()):
            st.sidebar.error("Please select at least one structure type!")
        else:
            generate_dataset_files(selected_structures, num_samples_per_type, output_dir, structure_type, current_params, generator)

def generate_dataset_files(selected_structures, num_samples_per_type, output_dir, current_structure_type, current_params, generator):
    """Generate dataset files for selected structure types with individual sample counts."""
    with st.spinner("Generating dataset..."):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            dataset_info = {
                'description': 'Synthetic coarse-grained peptide assemblies',
                'beads_per_peptide_range': list(generator.beads_per_peptide_range),
                'structures': {},
                'total_samples': 0
            }
            
            structure_mapping = {
                "Nanotube": ("nanotube", "nanotubes", NANOTUBE),
                "Nanofiber": ("nanofiber", "nanofibers", NANOFIBER),
                "Random Aggregate": ("random_aggregate", "random_aggregates", RANDOM_AGGREGATE),
                "Micelle": ("micelle", "micelles", MICELLE),
                "Lamellar Sheets": ("lamellar_sheets", "lamellar_sheets", LAMELLAR_SHEETS)
            }
            
            for struct_name, is_selected in selected_structures.items():
                if not is_selected:
                    continue
                
                num_samples = num_samples_per_type[struct_name]
                base_name, folder_name, default_config = structure_mapping[struct_name]
                
                st.sidebar.info(f"Generating {num_samples} {base_name} samples...")
                struct_dir = f"{output_dir}/{folder_name}"
                os.makedirs(struct_dir, exist_ok=True)
                
                # Use current params if generating current structure, else use defaults
                if struct_name == current_structure_type:
                    params = current_params
                else:
                    params = default_config.copy()
                
                structure_info = {'count': num_samples, 'files': []}
                
                for i in range(num_samples):
                    data = generate_structure(struct_name, params, generator)
                    file_name = f"{base_name}_{i:03d}"
                    gro_file = f"{struct_dir}/{file_name}.gro"
                    generator.save_to_gro(data, gro_file, f"Synthetic {base_name} #{i}")
                    
                    structure_info['files'].append({
                        'id': i,
                        'gro_file': gro_file,
                        'num_atoms': len(data),
                        'num_peptides': len(np.unique(data[:, 0].astype(int))),
                    })
                
                dataset_info['structures'][base_name] = structure_info
                dataset_info['total_samples'] += num_samples
            
            # Save metadata
            generator.save_metadata(dataset_info, f"{output_dir}/dataset_info.json")
            
            st.sidebar.success("Dataset generated successfully!")
            st.sidebar.write(f"Total samples: {dataset_info['total_samples']}")
            st.sidebar.write(f"Output: {output_dir}")
            
        except Exception as e:
            st.sidebar.error(f"Error generating dataset: {str(e)}")


def compute_peptide_centroids(data):
    """Collapse bead coordinates to one centroid per peptide using residue ids."""
    residue_ids = data[:, 0]
    coords = data[:, 3:6].astype(float)

    grouped_coords = {}
    for residue_id, coord in zip(residue_ids, coords):
        residue_key = int(residue_id)
        grouped_coords.setdefault(residue_key, []).append(coord)

    centroid_ids = []
    centroids = []
    for residue_id, peptide_coords in grouped_coords.items():
        centroid_ids.append(residue_id)
        centroids.append(np.mean(peptide_coords, axis=0))

    return np.array(centroid_ids), np.array(centroids, dtype=float)


def compute_bead_count_stats(data):
    """Summarize how many beads each peptide contains."""
    residue_ids = data[:, 0].astype(int)
    unique_ids, counts = np.unique(residue_ids, return_counts=True)
    return unique_ids, counts

def render_visualization(data, structure_type):
    """Render the 3D visualization and statistics."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Statistics")
        coords = data[:, 3:6].astype(float)
        peptide_ids, bead_counts = compute_bead_count_stats(data)
        min_beads = int(bead_counts.min())
        max_beads = int(bead_counts.max())
        avg_beads = float(bead_counts.mean())
        bead_range_label = str(min_beads) if min_beads == max_beads else f"{min_beads}-{max_beads}"
        
        st.metric("Total Atoms", len(data))
        st.metric("Total Peptides", len(peptide_ids))
        st.metric("Beads per Peptide", bead_range_label)
        st.metric("Avg Beads per Peptide", f"{avg_beads:.2f}")
        
        st.markdown("**Dimensions**")
        st.write(f"Width (X): {coords[:, 0].max() - coords[:, 0].min():.2f} nm")
        st.write(f"Width (Y): {coords[:, 1].max() - coords[:, 1].min():.2f} nm")
        st.write(f"Height (Z): {coords[:, 2].max() - coords[:, 2].min():.2f} nm")
    
    with col2:
        st.subheader(f"{structure_type} Atom Visualization")
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        bead_types = data[:, 2]
        centroid_ids, centroid_coords = compute_peptide_centroids(data)
        centroid_x, centroid_y, centroid_z = (
            centroid_coords[:, 0],
            centroid_coords[:, 1],
            centroid_coords[:, 2],
        )
        
        # Color configuration
        if color_by_bead_type:
            unique_beads = np.unique(bead_types)
            color_map = {bead: i for i, bead in enumerate(unique_beads)}
            colors = [color_map[bead] for bead in bead_types]
            marker_config = dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Bead Type"),
                opacity=0.8
            )
        else:
            marker_config = dict(size=3, color='lightblue', opacity=0.8)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=marker_config,
            text=[f"Bead: {bead}" for bead in bead_types],
            hoverinfo='text'
        )])
        
        # Add box outline
        box_coords = [
            [0, 0, 0], [BOX_SIZE, 0, 0], [BOX_SIZE, BOX_SIZE, 0], [0, BOX_SIZE, 0], [0, 0, 0],
            [0, 0, BOX_SIZE], [BOX_SIZE, 0, BOX_SIZE], [BOX_SIZE, BOX_SIZE, BOX_SIZE],
            [0, BOX_SIZE, BOX_SIZE], [0, 0, BOX_SIZE]
        ]
        box_coords += [[np.nan, np.nan, np.nan], [BOX_SIZE, 0, 0], [BOX_SIZE, 0, BOX_SIZE]]
        box_coords += [[np.nan, np.nan, np.nan], [BOX_SIZE, BOX_SIZE, 0], [BOX_SIZE, BOX_SIZE, BOX_SIZE]]
        box_coords += [[np.nan, np.nan, np.nan], [0, BOX_SIZE, 0], [0, BOX_SIZE, BOX_SIZE]]
        
        box_array = np.array(box_coords)
        fig.add_trace(go.Scatter3d(
            x=box_array[:, 0], y=box_array[:, 1], z=box_array[:, 2],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, BOX_SIZE], title="X (nm)"),
                yaxis=dict(range=[0, BOX_SIZE], title="Y (nm)"),
                zaxis=dict(range=[0, BOX_SIZE], title="Z (nm)"),
                aspectmode='cube'
            ),
            height=700,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"{structure_type} Peptide Centroid Visualization")

        centroid_fig = go.Figure(data=[go.Scatter3d(
            x=centroid_x, y=centroid_y, z=centroid_z,
            mode='markers',
            marker=dict(size=5, color='tomato', opacity=0.85),
            text=[f"Peptide ID: {peptide_id}" for peptide_id in centroid_ids],
            hoverinfo='text'
        )])

        centroid_fig.add_trace(go.Scatter3d(
            x=box_array[:, 0], y=box_array[:, 1], z=box_array[:, 2],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        centroid_fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, BOX_SIZE], title="X (nm)"),
                yaxis=dict(range=[0, BOX_SIZE], title="Y (nm)"),
                zaxis=dict(range=[0, BOX_SIZE], title="Z (nm)"),
                aspectmode='cube'
            ),
            height=700,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(centroid_fig, use_container_width=True)
        
        # Download button
        if st.button("Save as .gro file"):
            filename = f"data/synthetic_peptides/preview_{structure_type.lower().replace(' ', '_')}.gro"
            generator.save_to_gro(data, filename, f"Preview {structure_type}")
            st.success(f"Saved to {filename}")

# ===== MAIN UI =====

# Sidebar for structure type selection
structure_type = st.sidebar.selectbox(
    "Structure Type",
    ["Nanotube", "Nanofiber", "Random Aggregate", "Micelle", "Lamellar Sheets"]
)

st.sidebar.markdown("---")

# GPU Toggle and Generator initialization
st.sidebar.markdown("**Performance Settings**")

if GPU_AVAILABLE:
    use_gpu = st.sidebar.checkbox(
        "Enable GPU Acceleration",
        value=True,
        help="Use GPU for faster generation (PyTorch CUDA)"
    )
    if use_gpu:
        st.sidebar.success("GPU Mode: ON")
        st.sidebar.caption(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("GPU Mode: OFF (using CPU)")
else:
    use_gpu = False
    if TORCH_AVAILABLE:
        st.sidebar.warning("GPU not available (CUDA not detected)")
    else:
        st.sidebar.warning("PyTorch not installed")

# Generator settings
st.sidebar.markdown("**Generator Settings**")
bead_col1, bead_col2 = st.sidebar.columns(2)
with bead_col1:
    min_beads_per_peptide = st.number_input(
        "Min Beads / Peptide",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )
with bead_col2:
    max_beads_per_peptide = st.number_input(
        "Max Beads / Peptide",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
    )

beads_per_peptide_range = (
    int(min(min_beads_per_peptide, max_beads_per_peptide)),
    int(max(min_beads_per_peptide, max_beads_per_peptide)),
)

st.sidebar.caption(
    f"Variable bead count per peptide: {beads_per_peptide_range[0]}-{beads_per_peptide_range[1]}"
)

st.sidebar.markdown("---")

# Initialize or update generator based on GPU setting
if (
    'generator' not in st.session_state
    or st.session_state.get('use_gpu') != use_gpu
    or st.session_state.get('beads_per_peptide_range') != beads_per_peptide_range
):
    st.session_state.generator = SyntheticPeptideGenerator(
        use_gpu=use_gpu,
        beads_per_peptide_range=beads_per_peptide_range,
    )
    st.session_state.use_gpu = use_gpu
    st.session_state.beads_per_peptide_range = beads_per_peptide_range

generator = st.session_state.generator

st.sidebar.markdown("---")

# Random seed input
st.sidebar.markdown("**Random Seed**")
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1,
                                       help="Set seed for reproducible results")
np.random.seed(random_seed)

st.sidebar.markdown("---")

# Visualization options
st.sidebar.markdown("**Visualization Options**")
color_by_bead_type = st.sidebar.checkbox("Color by Bead Type", value=False,
                                          help="Enable to color beads by type, disable for uniform color")

st.sidebar.markdown("---")

# Render structure-specific parameters
st.sidebar.subheader(f"{structure_type} Parameters")
params = render_parameter_sliders(structure_type)

# Generate button or auto-generate
if st.sidebar.button("Generate Structure", type="primary"):
    st.session_state.auto_generate = True
    with st.spinner(f"Generating {structure_type.lower()}..."):
        np.random.seed(random_seed)
        data = generate_structure(structure_type, params, generator)
        st.session_state.data = data
        st.session_state.structure_type = structure_type
elif st.session_state.auto_generate and 'data' in st.session_state:
    # Auto-regenerate on slider change
    np.random.seed(random_seed)
    data = generate_structure(structure_type, params, generator)
    st.session_state.data = data
    st.session_state.structure_type = structure_type

# Save config button
st.sidebar.markdown("---")
if st.sidebar.button("Save Config to File", key="save_config"):
    try:
        # Update the dict in memory
        STRUCTURE_CONFIGS[structure_type]["config"].update(params)
        
        # Write changes permanently to config.py
        config_path = os.path.join(os.path.dirname(__file__), 'config.py')
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Map structure type to config variable name
        config_var_map = {
            "Nanotube": "NANOTUBE",
            "Nanofiber": "NANOFIBER",
            "Random Aggregate": "RANDOM_AGGREGATE",
            "Micelle": "MICELLE",
            "Lamellar Sheets": "LAMELLAR_SHEETS"
        }
        
        var_name = config_var_map.get(structure_type)
        if var_name:
            # Find the config dictionary in the file and replace it
            import re
            pattern = rf"({var_name}\s*=\s*\{{[^}}]*\}})"
            
            # Build the new config string
            new_config_str = f"{var_name} = {{\n"
            for key, value in params.items():
                if isinstance(value, str):
                    new_config_str += f"    '{key}': '{value}',\n"
                elif isinstance(value, (int, float)):
                    new_config_str += f"    '{key}': {value},\n"
                else:
                    new_config_str += f"    '{key}': {value},\n"
            new_config_str += "}"
            
            # Replace the old config with the new one
            content = re.sub(
                rf"{var_name}\s*=\s*\{{[^}}]*\}}",
                new_config_str,
                content,
                flags=re.DOTALL
            )
            
            # Write back to file
            with open(config_path, 'w') as f:
                f.write(content)
            
            st.sidebar.success(f"✅ Config permanently saved to config.py!")
            st.sidebar.info("The changes will persist across sessions.")
        else:
            st.sidebar.error(f"Unknown structure type: {structure_type}")
            
    except Exception as e:
        st.sidebar.error(f"Error saving config: {str(e)}")

# Dataset generation section
render_dataset_generation(structure_type, params, generator)

# Display visualization
if 'data' in st.session_state:
    render_visualization(st.session_state.data, st.session_state.structure_type)
else:
    st.info("Adjust parameters in the sidebar and click 'Generate Structure' to visualize")
    st.markdown("""
    ### Quick Start Guide
    
    1. **Set Random Seed** for reproducible results
    2. **Choose Structure Type** (Nanotube, Nanofiber, Random Aggregate, Micelle, or Lamellar Sheets)
    3. **Adjust Parameters** using the sliders
    4. **Click Generate** to create and visualize the structure
    5. **Auto-regeneration**: After first generation, slider changes update visualization automatically
    6. **Toggle "Color by Bead Type"** for different visualizations
    7. **Generate Dataset**: Create multiple samples with configurable structure types
    """)
