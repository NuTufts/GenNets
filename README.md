# 
# Score-based Diffusion Models for Generating Liquid Argon Time Projection Chamber Images 

Scripts used for our analyses. 
https://arxiv.org/abs/2307.13687

# Usage 

## Image Generation

Download our data and training checkpoints from Zenodo: 
(LINK) 

### Install the Score Network
https://github.com/yang-song/score_sde_pytorch

Add the configuration files 
```
score_sde_pytorch/configs/default_particle_config.py
score_sde_pytorch/configs/vp/larcv_png64_ncsnpp_continuous.py
```

Update the dataset.py in to accept the 
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


## Analyses 

### Install SSNet 

TODO








