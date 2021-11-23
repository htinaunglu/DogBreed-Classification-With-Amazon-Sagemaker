# DogBreed-Classification-With-Amazon-Sagemaker
This is a full project of imageclassification using pretrained model with sagemaker as Udacity's ML Nanodegree Project.

# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.
## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
We use the provide Dog Breed Dataset to use as data for the training jobs.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
I used Resnet34 pretrained model and 2 fully connected layers to classify the breeds of dogs.
The hyperparameter searchspaces are learning-rate and batchsize.
Deploy a hyperparameter tuning job on sagemaker and take the best job's hyperparameter to train the last model.

![hyperparameter tuning job](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/hpo-job.png)

![best job's hyperparameters](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/best-training-job.png)


## Debugging and Profiling
Profiling report is in the repo with HTML file

### Results
Low GPU Utilization & Loss alaways increasing. Can't get right even after over 100 time re-coding and tuning with different hyperparameters and different instances.

Prfiling report is in the repo as HTML file.


## Model Deployment
Model was deployed on the ml.g4dn.xlarge instance with best result. To quary, we need to transform the input image via cropping, and normalizing before pass it to the endpoint as a numpyarray.

![deployed endpoint](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/active-endpoint.png)

## Problems
There are mainly two problem in my workflow.
1. I can't get the hook data it always show the size(len) as always 1, included in the notebook output. So I can't plot as there is no enough data, I have tried over 6 days to solve this, and I decided to submit the project AS IT, so that I can get valuable corrections from the reviewers.
2. The sagemaker endpoint always show >Received server error (0)< whenever I try to pass image data to it.
