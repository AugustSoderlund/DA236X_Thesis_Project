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
