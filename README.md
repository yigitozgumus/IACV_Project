# IACV Project
Anomaly Detection Framework using Autoencoder and Denoising Network

Authors: [Şemsi Yiğit Özgümüş](https://github.com/yigitozgumus), [Yiğit Yusuf Pilavcı](https://github.com/Y2P)


## Instructions
- If you don't have the data folder, in the first run model will download and create the dataset.
- All the experiment configurations and model parameters can be changed from the related config files.

* To create the same environment used in the project: 

```bash
conda create --name myenv --file spec-file.txt
```

* To run the model:

```bash
python3 train.py -c ./configs/\<CONFIGFILE\> -e \<EXPERIMENTNAME\>
```

* You can also use the same experiment name and configuration file to continue unfinished experiment.
* Since it's a tensorflow based project, changes that affect the computation graph will result in failure to load the model. However you can modify the **test_epoch()** function to gain more insight about the model's predictions. To make additional predictions without training the model and by loading it :
```bash
python3 evaluate.py -c ./configs/\<CONFIGFILE\> -e \<EXPERIMENTNAME\>
```

