# CoarsePep3D

Tools for generating and analyzing synthetic coarse-grained peptide structures. The repo includes data generation scripts, trajectory processing utilities, a Streamlit visualizer, and notebooks for VAE, DeepSDF/DeepUDF, and Occupancy Network experiments.

## What's here

- Synthetic peptide generation in `generate_synthetic_peptides.py`
- Dataset helpers in `synthetic_peptides_dataset.py`
- Train/val/test split creation in `create_data_splits.py`
- Trajectory frame extraction in `process_peptide_trajectories.py`
- Interactive visualization in `visualize_peptides.py`
- Experiment notebooks and saved model weights

## Quick start

Install the Python packages used by the scripts and notebooks, then run the script you need:

```bash
python generate_synthetic_peptides.py
python create_data_splits.py
python process_peptide_trajectories.py
streamlit run visualize_peptides.py
```

## Data

- Generated structures are written under `data/synthetic_peptides/`
- Split datasets are written under `data/synthetic_peptides_split/`

## Notes

- Most model work lives in the notebook files in the repo root.
- The `occupancy_networks/` folder is included as a vendored dependency for Occupancy Network experiments.