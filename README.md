# HMM profile for the Kunitz-type protease inhibitor domain

Laboratory of Bioinformatic 1 (University of Bologna - MSc Bioinformatics) project aiming to build a HMM model for the detection of Kunitz domains, an inportant protease inhibitor cluster that play a putative role in many biological pathways. This repository contains all the used datasets, the python script and the pipeline executed using Bash scripting within a WSL enviroment. 

__Datasets__ 
- [training set](./training%20set)
- [test_set](./test%20set)

[__Python scirpts__](./python_script)

[__Pipeline__](./HMM_kunitz_pipeline.ipynb)

### Requirements
I used the following softwere and packages to performe the project:

__Enviroment__
- Conda environment setup on WSL operating system 
```
conda create -n hmm_kunitz_project python
conda activate hmm_kunitz_project
```

__Command-line tools__
- cd-hit v.4.8.1 -- for sequence clustering
```
conda install -c bioconda cd-hit
```
- BLAST + -- for sequence alignemnet and database searching
```
conda install -c bioconda blast
```
- HMMER -- for profile HMM construction and search
```
conda install -c bioconda hmmer
```

__Python packages__
- Biopython
