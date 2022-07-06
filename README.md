# USRR-method
Python program developed to apply the U-Net-based spatial reduction and reconstruction method for "Deep learning-based rapid flood inundation modelling for flat floodplains with complex flow paths"

Scripts with the "SRR_2" heading contain the basic program of the SRR method (https://github.com/yuerongz/SRR-method) and the geospatial part of the new USRR method. 
These scripts can be used to apply the original SRR method, with an example provided in "SRR_2_v1_example.py". 

"WH_srr_rl_selection.py" include the program used for the selection of representative locations. 
The development of the U-Net model is included in "WH_unet_model.py" and "WH_unet_training.py" scripts, given the William Hovell case study as example. The validation program is also provided. 

For the application of the original SRR method to the William Hovell case study:
"WH_srr_reco.py" include the program used for the reconstruction of flood surface using 2D linear intepolation. 
