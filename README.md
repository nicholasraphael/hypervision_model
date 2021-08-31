<div align="center">    
 
# Reform Deep Learning Vision Models    

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Version 0.0 of Reform Image Models for sorting textiles based on various HSI images, label tags, etc.

## Quick Start   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ReformCompany/vision_models.git

# install project   
cd vision_models 
pip install -e .   
pip install -r requirements.txt
 ```   

Test installation  
 ```bash
# module folder
cd reformvision

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
