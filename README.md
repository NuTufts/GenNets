
# Score-based Diffusion Models for Generating Liquid Argon Time Projection Chamber Images 

See https://arxiv.org/abs/2307.13687.


## Image Generation

Download our data and training checkpoints from Zenodo: 
https://zenodo.org/record/8300355

### Install the Score Network
https://github.com/yang-song/score_sde_pytorch

Move the configuration files from the configs folder to the score network.
```
score_sde_pytorch/configs/default_particle_config.py
score_sde_pytorch/configs/vp/larcv_png64_ncsnpp_continuous.py
```

Edit `dataset.py` to accept the new dataset.
```python
    elif config.data.dataset == 'larcv_png64':
        dataset_builder = tfds.builder('larcv_png64')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, 
            config.data. image_size], antialias=True)
```

Run the network in train mode or eval mode to generate images. 
```python
python main.py --config=configs/vp/larcv_png64_ncsnpp_continuous.py --mode=eval --workdir=larcv_png64_workdir
```


## Image Processing 

The script `npy_manager.py` will compile and process the generated images for analysis. Flags can be used to specify the input and output files. 

## Analyses 

Optional flags have been implemented on all non-plotting scripts. Run with `--help` for more info.  

### SSNet 

Install SSNet: https://github.com/NuTufts/LArTPC-VQVAE/tree/main and the neccessary dependencies (ROOT, etc.).

Download the model weights https://zenodo.org/record/4728517#.YqJTntLML9A and copy to the ssnet directory. 

```
mv ssnet.dlpdataset.forSimDL2021.tar /LArTPC-VQVAE/analysis/ssnet/
```

Move the updated running script to the `/py/` directory. 

```
mv ssnet_analysis/run_ssnet.py /LArTPC-VQVAE/analysis/ssnet/py/
```

Run from the ssnet directory. Use flags (or edit the file) to specify to desired event file input, output location, and output name. 

```
cd /LArTPC-VQVAE/analysis/ssnet/
python py/run_ssnet.py 
```
By default this will output the ROOT histograms and seperate track and shower events as numpy files. 

The script `ssnet_analysis/ssnet_root_comp.py` provides a rudimentary method for aggregating and comparing these ROOT histograms. 

### Physics Metrics

Run `track_analysis.py` on the track events and `shower_analysis.py` on the shower events using flags to specify input and output. 

Once you have the desired event analyses plot the track lengths and widths histogram using `length_plotting.py`. Similarly, use `charge_plotting.py` to plot the charge comparison histogram. 
Plotting files do not have flags implemented, so the filepaths will need to be manually edited. 

Decpreciated: `angle_plotting.py` is used confirm angles are uniform. Need to used `--saveAngle True` flag during track analysis first. 

### SSNet-FID

A modified version of [Fréchet Inception Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) (FID). 

Use `run_ssnet.py` with the flag `--saveFID True` to get SSNet activations. Run on all epochs desired for comparison and move outputs to a single directory. 

Use `FID.py` to calculate the FID distance and plot with `plot_FID.py`. 

### High Dimensional Goodness of Fit Tests

Three methods of directly comparing the high dimensional distribution from which the images are sampled from: Maximum Mean Discrpancy (MMD), Sinkhorn Divergence, and Wasserstein-1. 

The script `GoF.py` will generated the values, use the flag `--GoF_test` to specify which comparison to use (MMD, Sink, or W1). Plot with `plot_GoF.py`. 

### Nearest Neighbors

 [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (EMD) or [Euclidian Distance](https://en.wikipedia.org/wiki/Euclidean_distance) (l2) method for finding the nearest neighbor for a selected image. 

First choose the desired image using the flag `--select True`. This will display a grid of 16 labeled images of ascending index starting from the index input using the flag `--eventNum`. Once the desired image is found, make note of its index number as labeled. 

Run the script again ommiting the `--select` flag (False by default) and specify the desired event index using `--eventNum`. This will produce an image containing the desired event and the 5 (`--numNeighbors`) nearest neighbors from the training images on the top row and generated images on the bottom row (excluding self match). 

Note the horizontal image labels seen in the paper will need to be added separately. 

### Sample Events 

Simple script to create a grid of randomly selected images. Use the usual flags to specify inputs and outputs. Use flag `--gen True` to sample from generated images and `False` to sample from the training images. 

# Contact

Zeviel.Imani@tufts.edu 


