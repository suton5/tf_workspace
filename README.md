# tf_workspace
The mnist/ directory contains the MNIST tutorial from the Tensorflow website.
The rmrc/ directory contains the training and detecting scripts used for RMRC 2018. 

## retrain.py
Uses transfer learning on the Inception V3 architecture to retrain the final layer for the 12 RMRC classes.

## detect.py
Takes the input image containing the four signs and slices it up to get individual signs. The net is then run on each sign, and the resultant labels are compiled and displayed in a 2x2 matrix.

