
(1) Goal of the project

(2) Execution of the code
The project works as so : Perfome experiments
Results goes in HDF5 file
Analyse these Results

How to proceed : 

It is necessary to have a model that works as so :
it's imput is an image -> it outputs an array of neuron activities

In config.py chose every experiments/analyses you want to perform,
Chose there parameters
save

main.py executes the config.py

an exemple is shown in the config.py file

(3) Experiments 
experiment functions in experiments.py
well described there

List every experiment, describe, show prerequisite, list every parameter

Things to know about the functions and the HDF5 file :

    (a) If the file doesn't exist, it creates it. Same for every group of the file

    (b) If some group exists in the file, you have the choice of overwriting them or appending in them 

    (c) If you chose to add some data to already existing data, the functions will check 
    if the arguments are the same in both cases.

    (d) If the function requires the execution of other functions it will :

        - check if those analyses were performed
        - check if the arguments are matching between all of those analyses
        - check if the requested neurons are present in all of thoses analyses


(4) Analyses
Analyses functions in analyses.py

List every analyses, and every parameter

(5) HDF5 utilisation
Everything work in a HDF5 file 
(see functions.py)

(6) Utils functions in utils.py
- Plot images
- Manage HDF5
- other