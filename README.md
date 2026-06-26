# MTMPIC
M³-IonNet: A Multimodal, Multitask Deep Learning Framework for Large-Scale Functional Profiling of Ion Channel Ligands

This model Training was performed on an NVIDIA H100 80 GB GPU equipped with CUDA 12.6. The current training parameters require at least 80 GB of VRAM.

1. Building Environment:
conda env create -f environment.yml

2.Generate SDF:
python csv_smi.py,
.smile_sdf.sh
It needs to be replaced with the corresponding test file.

3. Generate features:
python Feature_generate.py
It needs to be replaced with the corresponding test file.

4. Model training:
python MTLION_main.py
