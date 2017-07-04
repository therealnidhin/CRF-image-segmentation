# CRF-image-segmentation

1. Install tensorflow, tf-slim and other dependencies required by the code (Ideally in a virtual environment).
2. Download VGG model. http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
3. Untar the model and then update the checkpoints_dir variable as folder containing the extracted vgg model
4. Update log folder variable log_folder
5. Lines 272 - 279 have been temporarily commented to prevent plots being showed after each iteration. Plots holds up further execution till the plot window is closed. 
6. The input image size should be 480x352. All images must be resized to this dimension before being fed to the algorithm.
7. Replace vaiable named 'processed_probabilities' with 'softmax'(Already done in the code).
