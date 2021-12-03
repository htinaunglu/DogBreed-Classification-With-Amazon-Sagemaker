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
I used Resnet34 pretrained model and 3 fully connected layers to classify the breeds of dogs.
The hyperparameter searchspaces are learning-rate and batchsize.
Deploy a hyperparameter tuning job on sagemaker and wait for the combination of hyperparameters turn out with best metric.

![hyperparameter tuning job](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/hpo-job.png)

And pick the hyperparameters from the best training job to train the vfinal model.
![best job's hyperparameters](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/best-training-job.png)


## Debugging and Profiling
Profiling report is in the repo with PDF file.

### Results
Result is pretty good, as I was using ml.g4dn.xlarge to utilize the GPU of the instance, both the hpo jobs and training job did't take too much time.


## Model Deployment
Model was deployed on the ml.g4dn.xlarge instance with best result. First version of my deployment didn't work, so I need to add new inference script call ![endpoint.py](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/endpoint.py) to set up and deploy a working endpoint.

![deployed endpoint](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/active-endpoint.png)

The quarying result is shown in the notebook and here is the cloudwatch log of the inference.
![Cloudwatch Log](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/endpoint-cloudwatch-log.png)

## Using the Endpoint
To use the endpoint, just pass a local image of dog, or the address of the jpg image to the endpoint. Every processing job is done in the inference script and user can enjoy the result easily.

