#!/bin/sh
#python run_on_titan.py -c configs/autoencoder_denoiser.json -e exp_ae
python run_on_titan.py -c configs/dae_denoiser.json -e exp_dae_mask_loss_2
python run_on_titan.py -c configs/cvae_denoiser.json -e exp_cvae_mask_loss_2
