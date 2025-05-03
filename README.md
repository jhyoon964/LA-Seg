# A-Seg: Disentangled Sinogram Pattern-guided Transformer for Lesion Segmentation in Limited-angle Computed Tomography


# Install the environment
## Paired Data Generation
### For training
```
git clone https://github.com/openai/guided-diffusion.git
```
### For generation
```
git clone https://github.com/andreas128/RePaint.git
pip install numpy torch blobfile tqdm pyYaml pillow
```
### Setting
```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 512 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 64 --timestep_respacing 1000"
```


## Datasets
KiTS is available at https://kits-challenge.org/kits23/

CTICH is available at https://physionet.org/content/ct-ich/1.3.1/

FUMPE is available at https://figshare.com/collections/FUMPE/4107803/1


## Training & Testing
```
# Train
python trans_train_limit.py

# Test
python trans_val_limit.py
```
