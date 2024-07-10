# OC-P10
# Semantic Segmentation
# Improve the performance of a base model - POC

## Project scenario

The company DataSpace requires a technical test as part of the recruitment process.

This company supports its clients in designing and implementing data science solutions, both for structured data and for image or text processing issues.

The objective is to:

- choose a previously developed project,
- research a more recent algorithm that can improve performance,
- adapt it to be able to compare it with the old technology,
- develop an application in the form of a dashboard to present the study.

## Solutions

- previous project : Semantic segmentation on CityScapes dataset
- previous model : *Unet-Resnet*
- new model to explore : *SegFormer*

## Run the app

Trained models are saved in the repo. Hence the Dashboard can be launched.

### Local
Launch the script :
```bash
python launch_app.py local
```

### Using Azure
First create env var
```bash
set AZURE_SUBSCRIPTION_ID=<subscription_id>
```
Launch the script :
```bash
python launch_app.py production
```
