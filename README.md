# Super_Resolution_FOCCUS
Super Resolution of sea surface height for the FOCCUS project.

The goal is to increase the resolution of the DUACS data set consisting of full, low resolution fields of SSH with a neural network using the new SWOT SSH fields consisting of sparse, high resolution fields of SSH.

# Description of this repo

A Conditional Generative Adversarial Network (GAN) was trained on matching pair of DUACS and FOCCUS patches. The architecture was inspired by
https://jleinonen.github.io/ and is divided into

- data.py the data loader
- models.py with the generator and discriminator models
- gan.py implementing the adversarial training
- plots.py containing some plot and inference functions
- weights containing the weights of a trained network
- Testing.ipynb to test the super resolution and see different metrics (power spectrum, rmse, ensemble rmse)
