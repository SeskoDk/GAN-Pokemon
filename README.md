# GAN - Pokémons
This project focuses on generating new Pokémon images by using convolutional neural networks as a generator and discriminator. The GAN is trained with the original [Pokémon dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset). \
For simplicity, the alpha channel was omitted so that only RGB images with a size of 128 x 128 were considered.

**Remark:** The png images must be saved in the `data` folder as `pokemon_dataset`.

The dataset has the following properties:
* Total amount: 819 images
* Size: 41 MB
* MIME-Type: png
* Shape: 256 x 256 x 4

<p align="center">
  <img width="300" height="180" src="docs/imgs/figure1.png">
</p>

### Resources
[FID-Score](https://arxiv.org/abs/1706.08500) \
[FID-Implementation](https://github.com/mseitzer/pytorch-fid) \
[Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)\
[Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738) 
## Usage

### Installing Packages

For the packages use miniconda. \
See [Miniconda](https://docs.conda.io/en/latest/miniconda.html).


Use the Anaconda Prompt terminal for the following steps:
```
conda create --name myenv python=3.8.5 pip
```

When conda asks you to proceed `proceed ([y]/n)?`, type `y`:

Activate the new environment: 
```
conda activate myenv
```

Install the packages via requirements.txt
```
 pip install -r requirements.txt
```

### Preprocessing images

The images will be resized to `128 x 128 x 3` at the beginning

```
python -m preprocessing.preprocessing
```
### Train new generator
A trained model is available under`trained models`.
If you want to train a new generator, you have to execute this command
```
python -m train train_config.json
```
The trained models can be found in the folder `trained_models`
### Generate images

For generating new Pokèmon images, you can choose between two modules. \
**For single images**
```
python -m generate_images
```
<p align="center">
  <img width="250" height="90" src="docs/imgs/figure3.png">
</p>

**For grid images**
```
python -m generate_grid_images
```
<p align="center">
  <img width="250" height="300" src="docs/imgs/figure2.png">
</p>

The images can be found under the paths:
* `data\generated_images`,
* `data\generated_grid_images`.

### Evaluation
The FID-Score will be used to evalute the GAN. \
It is important to create the same number of images as the original data. \
The following folders are required:
* Images: `data\pokemon_preprocessed`
* New Images: `data\generated_images`

To use the FID-Score the follwing package has to be installed
```
pip install pytorch-fid
```

To evaluate the Pokèmon images, you have to run
```
python -m evaluation.pytorch-fid.src.pytorch_fid.fid_score --num-workers 2 src\generated_images src\pokemon_preprocessed
```

| Model | FID-Score |
|:----------|:-------------|
| 2022-03-08_15_49_59.pt |  173.13 |


## Project-Structure
```
GAN-Pokemon
|- generate_images.py
|     Generates new single/grid images and saves them.
|- train.py
|     Train and saves the generator.
|- train_config.json
|     Configuration file containing the hyperparamters. 
|- preprocessing.py
|     Resizes the images to 128 x 128, deletes the alpha channel and saves them
|- preprocessingClass.py
|     Architecture of the dataset class
|- DiffAugmentation.py
|     Required for the augmentation process of the GAN.
|- cnn.py
|     CNN architecture of the generator and discriminator.
|- mlp.py
|     MPLP architecture of the generator and discriminator.the generator
|- fid_score.py
|     Calculates the score of the generator w.r.t the original dataset
|- utils.py
|     Utility functions.
```

-----------------
#### Best wishes and have fun!
