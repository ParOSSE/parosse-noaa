# parosse-noaa
This repository contains codes required to run the ParOSSE workflow from end to end for a given architecture. 

The codes are compatible with python 3.10. To begin, install the required packages in the requirements.txt file as follows

    pip install -r requirements.txt

Then, install the Parmap package - within the Parmap/ directory, issue:

    python ./setup.py install

## General Description:

Exercises the full end-to-end ParOSSE workflow, with CSV inputs and outputs.

1. Configures settings for an experiment, including:
   a. NR type and information
   b. forward model settings
   c. instrument model settings
   d. forward model settings
   e. retrieval settings
   f. analysis and plotting settings
2. Pre-processes nature run data
3. Runs the forward model
4. Runs the instrument model
5. Runs the retrieval
6. Analyzes and plots the output

Each sub-system has its own sub-directory containing codes relevant for that particular part of the workflow

The workflow is intended to be flexible and extensible.

The top level code that runs the workflow is contained in "Run_parosse.py", and a sample jupyter notebook is provided that shows how each stage of the workflow is applied (Run_parosse.ipynb).

Users wishing to test this version will need to provide their own nature run data, and to adjust the "create_nr_config.py" file to point to their data. They may need to provide their own read function as well.

The default configuration is set up to run with a case from the GEOS-5 nature run. Test data can be obtained from:

https://zenodo.org/records/14294436

Base requirements: numpy, scipy, netCDF4, hdf5, hdf5plugin
Requirements for ParMAP: json, pysparkling, multiprocessing, pickle, cloudpickle, tqdm, boto3, pandas
 
Developers:

Derek Posselt, Brian Wilson, Diego Martinez, Vishal Lall, Hai Nguyen
