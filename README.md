# HMM profile for the Kunitz-type protease inhibitor domain

Computational pipeline for build a __HMM model__ for the detection of __Kunitz domain__ (Pfam ID: PF00014), an important protease inhibitor cluster that play an significant role in many biological pathways. Some examples of them are aprotinin (bovine pancreatic trypsin inhibitor - BPTI), Alzheimer's amyloid precursor protein (APP), and tissue factor pathway inhibitor (TFPI). 
The HMM model was developed using a MSA (Multiple Sequence Alignment) of the well-known kunitz'type proteins, obtained using PDBeFOLD structural alignemnt tool. The model's performance was evaluated using both positive (Kunitz-type) and negative (non-Kunitz-type) datasets through a 2-fold cross-validation approach. Several statistical metrics were computed to assess the model's accuracy, including the confusion matrix, Matthews Correlation Coefficient (MCC), performance, recall, Q2, and Area Under the Curve (AUC).

This project was required by Laboratory 1 of Bioinformatic of the __MSc in Bioinformatics at the University of Bologna__ as part of the final evalutaion. 


## Repository organization
This repository contains all the used datasets, the python script and the pipeline executed using Bash scripting within a WSL enviroment. 

| Section         | Link                                                              |
|-----------------|-------------------------------------------------------------------------------|
| **Datasets**    | - [Training set](./training%20set) <br> - [Test set](./test%20set)            |
| **Python script** | [Python srcripts](./python_script)                                 |
| **Pipeline**     | [Jupyter Notebook](./HMM_kunitz_pipeline.ipynb)               |


## Requirements
I used the following softwere and packages to performe the project:

### Enviroment
- Conda environment setup on WSL operating system 
```
conda create -n hmm_kunitz_project python
conda activate hmm_kunitz_project
```

### Command-line tools
- cd-hit v.4.8.1 -- for sequence clustering
```
conda install -c bioconda cd-hit
```
- BLAST + -- for sequence alignemnet and database searching
```
conda install -c bioconda blast
```
- HMMER v.3.4 -- for profile HMM construction and search
```
conda install -c bioconda hmmer
```

### Python libraries
- Biopython -- for parse the FASTA file from the proteins IDs (__get_seq.py__ script)
- Pandas -- for data manipulation
- Numpy -- for data manipulation
- Matplotlib -- for data visulaization

## Output files
### Training set
- `rcsb_pdb_custom_report_20250505053057.csv` -- PDB csv file of the representative kunitz'type proteins
  
- __Clusters file__ -- file used and obtained with cd-hit clusterization
    - `pdb_kunitz_customreported_nr.fasta.clstr` -- output file 

- __MSA__ -- file used and obtained with the Multiple sequence alignment performed with PDBeFOLD
    - `tmp_pdb_efold_ids.txt` -- input file, all the IDs of the 23 representative proteins
    - `pdb_kunitz_rp.ali` -- output file
    - `pdb_kunitz_rp_formatted.ali` -- output formatted for being compatible with HMMER
  
### Test set 
- __Positive sets__
    - `all_kunitz.fasta` -- all positive human and not human kunitz's type protein retrieved from UniProt
    - `to_remove.ids` -- output of BLAST search. IDs to remove from the total positive set because the redundancy between themselves and the training set 
    - `pos_1.fasta`, `pos_2.fasta` -- FASTA file of the divided and randomized initial positive set
    -  `pos_1.ids`, `pos_2.ids` -- IDS file of the divided and randomized total positive set
    -   `pos_1.out`, `pos_2.out` -- OUT file of the two sets. This is the output of the hmmsearch tool of HMMER
    -    `pos_1.class`, `pos_2.class` -- CLASS file of the hmmsearch output

- __Negative sets__
    - `neg_1.ids` , `neg_2.ids` -- IDS file of the divided and randomized total negative set
    - `neg_1.out` , `neg_2.out` -- OUT file of the two sets. This is the output of the hmmsearch tool of HMMER
    - `neg_1.class` , `neg_2.class` -- CLASS file of the hmmsearch output
    - `neg_1_hits.class` , `neg_2_hits.class` -- manually recovered negative proteins with E-value above 10.0

- __Final used sets__
    - `set_1.class` , `set_2.class` -- CLASS file used for the 2-Fold cross validation. Obtained from the marge of `pos_1.class`/`pos_2.class` with `neg_1_hits.class`/`neg_2_hits.class`
    - `total_set.class` -- merge of the two final sets

### Results
- `hmm_results.txt` -- statistical result of the HMM model obtained with `performance_crossvalidation.py` script
- `structural_model.hmm` -- hmmbuild result. It's the model costructed on the MSA

## Author 
Anna Rossi 

MSc in Bioinformatics - University of Bologna 

Contact: rossianna557@gmail.com
      
