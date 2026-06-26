# MTMPIC
M³-IonNet: A Multimodal, Multitask Deep Learning Framework for Large-Scale Functional Profiling of Ion Channel Ligands

This model Training was performed on an NVIDIA H100 80 GB GPU equipped with CUDA 12.6. The current training parameters require at least 80 GB of VRAM.

1. Building Environment:
conda env create -f environment.yml
2. Generate features:
python Feature_generate.py   
3. Model training:
python MTLION_main.py
