# selective_recruitment

Contains code to make the dataframe for selective recruitment testing.
For more information, see (selective recruitment paper)

## Installation and dependencies
This module uses the Functional Fusion package and assumes that your data is organized according to the directory structure defined in Functional Fusion framework.
see https://github.com/DiedrichsenLab/Functional_Fusion for information on dependencies, data structures, and how to organize your dataset.

## 1. Data extraction
Data must be extracted using Functional_Fusion framework. 
Check out extract_<dataset>.py under scripts.

$ extract_wmfs(ses_id='ses-02', type='CondAll', atlas='fs32k')
  
If you want to create and save a data tensor with data for all the subjects, use save_tensor from cortico_cereb_connectivity repository.
 

## 2. Creating dataframes for plotting the scatterplots

## 3. Using Connectivity weights

##

