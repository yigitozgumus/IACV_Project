#!/bin/sh

python run_on_titan.py -c configs/dae_denoiser.json -e exp_dae_add_loss
python run_on_titan.py -c configs/autoencoder_denoiser.json -e exp_ae_add_loss
python run_on_titan.py -c configs/cvae_denoiser.json -e exp_cvae_add_loss
