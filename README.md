# selective_recruitment

Contains code to make the dataframe for selective recruitment testing.
For more information, see (selective recruitment paper)

## Installation and dependencies
This module uses the Functional Fusion and corticco_cereb_connectivity packages and assumes that your data is organized according to the directory structure defined in Functional Fusion framework.
see https://github.com/DiedrichsenLab/Functional_Fusion for information on dependencies, data structures, and how to organize your dataset.

First, clone these repositories by:<br>
```git clone https://github.com/DiedrichsenLab/Functional_Fusion.git

git clone https://github.com/DiedrichsenLab/cortico_cereb_connectivity.git```

Second, clone the repository for selective recruitment by: <br>
```git clone https://github.com/DiedrichsenLab/selective_recruitment.git```

Third, open your bashrc with a text editor and add paths to these repositories. For example:.<br>
```export PYTHONPATH="${PYTHONPATH}:/home/ROBARTS/lshahsha/Documents/Projects/Functional_Fusion"
export PYTHONPATH="${PYTHONPATH}:/home/ROBARTS/lshahsha/Documents/Projects/selective_recruitment"```

Next, cd to the local folder for your repository and create a virtual environment on your computer, activate it, and install all the required dependencies: <br>
```
python3 -m venv ./env

source ./env/bin/activate

pip install -r requirements.txt
```
## 1. Data extraction
Data must be extracted using Functional_Fusion framework. 
Check out extract_<dataset>.py under scripts.

```extract_wmfs(ses_id='ses-02', type='CondAll', atlas='fs32k')```
## 2. Creating dataframes for plotting the scatterplots

```
import selective_recruitment.recruite_ana as ra

D = ra.get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_roi =None, 
                cortex_roi = None,
                add_rest = True)

D = ra.get_summary(dataset = "WMFS", 
                ses_id = 'ses-02', 
                type = "CondAll", 
                cerebellum_roi ="Verbal2Back", 
                cortex_roi = "Verbal2Back.32k",
                add_rest = True)
```

## 3. Using Connectivity weights

##

