# ML3D_2021WS Final Project
Reconstruction based 3D Mesh retrieval from single view  

## Folder Structure  

├── ML3Dfinal_2021WS   # Project folder  
│   │   
│   ├──Assets # folder for needed Assets used in code   
│   │   ├── Data # folder for the used datasets   
│   │   ├── Models # folder for the saved trained models   
│   │   └── Reports # folder for pdf reports written and submitted   
│   │   
│   ├── Data  # folder for data manipulation code   
│   │    └── TBD   
│   │   
│   ├── Inference # folder for model inference and full pipeline testing   
│   │    └── TBD   
│   │   
│   ├── Networks # folder for all network classes used in training    
│   │    └── TBD   
│   │   
│   ├── Utils # folder for all utility helper function classes   
│   │    └── TBD   
│   │   
│   └── Main.py    # Main File   
└──   

##
For the 3d reconstruction of partial 3d point clouds
1. first download the rendered ground truth data and the depth map from https://mega.nz/file/7QFkEJIZ#DIcZYIg7MYxKuTC6HBX9H93JNU_w83mcy7tJN3rjnZc
2. unpack them in ./Data/shapenetdata/
3. generate partial & full point cloud data using loaddata.py (split the data set based on the number of your cpu thread, might take couple of hours) example with single thread: ``loadpcd.py 0 2000``
4. start training! check out train_mesh2mesh.ipynb
5. some examples can be checked in the ./stage_1 (3d point cloud autoencoder) and ./stage_2 (partial point cloud reconstruction)

## Packages used

All packages used could be found in requirements.txt file (produced with pip freeze)
To install the packages simply use
```
pip install -r requirments.txt
```

