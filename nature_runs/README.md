# Nature Run Interfaces

This directory contains python codes that serve as interfaces to various nature runs.

The naming convention for each interface file is: 

read_<NR name>.py

For example, read of data from the WRF model would be read_wrf.py

The goal is to provide a single file per nature run that will pull all data needed for any forward model linked to the ParOSSE framework.
