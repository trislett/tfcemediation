# tfcemediation

tfcemediation is a high-performance statistics toolkit specifically optimized for neuroimaging data analysis.

This package provides memory-efficient implementations of standard and advanced statistical methods for both surface-based and volumetric neuroimaging data. Voxel-wise and vertex-wise analysis with permutation testing can be done in 1000s of subjects from a single computer or compute node. This package is the sucessor of [TFCE_mediation](https://github.com/trislett/tfce_mediation), which is now  depreciated. Major improvements have been made with respective to ease of use and overall efficiency. The software is written mostly in python with some sections written in c++ and cython for efficiency.

## Key Features
### Memory Efficiency: Process large neuroimaging datasets with minimal RAM through memory mapping
### Comprehensive Statistical Tools:

Linear regression with t-statistics and F-statistics
Mediation analysis for examining indirect neural pathways
Nested model comparisons for hypothesis testing

### Advanced Multiple Comparison Correction:

Threshold-Free Cluster Enhancement (TFCE) for both surface and volumetric data
Non-parametric permutation testing with FWE correction


### Performance Optimized:

Parallel computation for permutation tests
Chunked processing to manage memory efficiently during intensive operations


### Flexible I/O:

Direct integration with pandas DataFrames
Support for both NIfTI and FreeSurfer file formats

## Installation

## Loading the neuroimaging data

## Basic analysis

## Surface TFCE

## Mediation TFCE

## Nested TFCE

## Citation
[Lett TA, Waller L, Tost H, Veer IM, Nazeri A, Erk S, Brandl EJ, Charlet K, Beck A, Vollstädt-Klein S, Jorde A, Keifer F, Heinz A, Meyer-Lindenberg A, Chakravarty MM, Walter H. Cortical Surface-Based Threshold-Free Cluster Enhancement and Cortexwise Mediation. Hum Brain Mapp. 2017 March 20. DOI: 10.1002/hbm.23563](http://onlinelibrary.wiley.com/doi/10.1002/hbm.23563/full)

The pre-print manuscript is available [here](tfce_mediation/doc/Lett_et_al_2017_HBM_Accepted.pdf) as well as the [supporting information](tfce_mediation/doc/Lett_et_al_2017_HBM_supporting_information.docx).
