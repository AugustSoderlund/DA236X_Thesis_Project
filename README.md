# DA236X_Thesis_Project
## [Thesis Project](thesis_project)
The main thesis project is located in this folder where the chosen methodology is implemented and tested against a baseline method.
## [Pre-Study](pre_study)
This folder contains the different methodologies that were tested during the pre-study.

# What Can This Repository Be Used For?
The code provided in [thesis project](thesis_project) can be used for computing the reachable sets of pedestrians in the SIND dataset. Some important information about the implementation:
1. It uses zonotopes for the set-based state reprensetation, see [zonotope.py](thesis_project/DRA/zonotope.py)
2. The initial set of the pedestrians have the estimated position as the center-point in the zonotope, and the generator vectors are assumed to resemble some measurement noise
3. The code has only been tested on a Windows machine, meaening that some optimization might not be used
4. This currnetly only works for th SIND dataset, and the limitation lies in the craetion of polygons that represent different parts in the map, see [map.py](thesis_project/utils/map.py)
5. The prediction horizon is currntly set to 9 seconds (that is 90 timesteps). This can be changed to something lowere without any otheer modifications, but if increased then the chunk sizes for the data must be increased also. That is, the argument **input_len** accross all functions must be equal to or larger than the desired prediction horizon (in timesteps)

## How to add the dataset
The efficacy and feasibility of the results increase if the entire dataset is used. There is a small sample of the dataset publicly available, however, it is recommended to request full access to the dataset. How to request access can be found [here](https://github.com/SOTIF-AVLab/SinD#access) (note: this will take you to another github repo)
```bash
├───thesis_project
│   └───.datasets 
│       intersection.jpg  
│       └───SinD    #<-- Put the entire SinD folder here (inside of /.datasets/)
│           └───Data   
│               mapfile-Tianjin.osm   #<-- ensure that this file is under /Data/
│               └───8_02_1
│               └───...
```

## How to run the code
The main script to reproducing the results is [main.py](thesis_project/main.py). In this script, simply import and call the functions that should be called to reproduce the results. The easiest way to reproduce the results is actually to go to [behavioral_data_driven_reachability](https://github.com/AugustSoderlund/behavioral-data-driven-reachability/tree/main) and run the pre-defined functions in its main.py script.
