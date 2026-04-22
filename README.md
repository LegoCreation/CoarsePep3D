# CoarsePep3D

Tools for generating and analyzing synthetic coarse-grained peptide structures. The repo includes data generation scripts, trajectory processing utilities, a Streamlit visualizer, and notebooks for VAE, DeepSDF/DeepUDF, and Occupancy Network experiments.

## Script folder contains:

- Synthetic peptide generation in `generate_synthetic_peptides.py`
- Dataset class in `synthetic_peptides_dataset.py`
- Train/val/test split creation in `create_data_splits.py`
- Trajectory frame extraction in `process_peptide_trajectories.py`
- Interactive visualization in `visualize_peptides.py`

## Requirements

```bash
pip3 install -r requirements.txt
```

## For data generation and visualization

```bash
python generate_synthetic_peptides.py
python create_data_splits.py
streamlit run visualize_peptides.py
```

## VAE model

./notebooks/train_synthetic_peptides.ipynb

## Occupancy model

Setup I (unconditional): occupancy_networks_peptides.ipynb

Setup II (conditional): ./notebooks/occupancy_networks_peptides_conditional.ipynb


## SDF and UDF analysis:

./notebooks/sdf_vs_udf.ipynb

## Evaluation

./notebooks/evaluate_models.ipynb