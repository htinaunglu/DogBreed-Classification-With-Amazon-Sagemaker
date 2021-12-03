# DogBreed-Classification-With-Amazon-Sagemaker
This is a full project of imageclassification using pretrained model with sagemaker as Udacity's ML Nanodegree Project.

# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. Open Jupyter Notebook and start coding by installing dependencies. `smdebug` is a must installed module.

## Dataset
We use the provide **Dog Breed Dataset** to use as data for the training jobs.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data, using `aws s3 sync bucket` command.

## Script Files used
1. `hpo.py` for hyperparameter tuning jobs where we train the model for multiple time with different hyperparameters and search for the best one based on loss metrics.
2. `train_model.py` for really training the model with the best parameters getting from the previous tuning jobs, and put debug and profiler hooks for debugging purpose.
3. `endpoint.py` to using the trained model as inference and post-processing and serializing the data before it passes to the endpoint for prediction.

## Hyperparameter Tuning
I used Resnet34 pretrained model and 3 fully connected layers to classify the breeds of dogs.
The hyperparameter searchspaces are learning-rate and batchsize.
Deploy a hyperparameter tuning job on sagemaker and wait for the combination of hyperparameters turn out with best metric.

![hyperparameter tuning job](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/hpo-job.png)

here is the *worst* training job opposite to the best one.

![worst job's hyperparameters](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/worst-training-job.png)

We pick the hyperparameters from the *best* training job to train the final model.

![best job's hyperparameters](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/best-training-job.png)


## Debugging and Profiling
The Debugger Hook is set to record the Loss Criterion of the process in both training and validation/testing.
The Plot of the *Cross Entropy Loss* is shown below. 

![Cross Entropy Loss Plot](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/cross-entropy-loss-lineplot.png)

we can see that the line is not smooth and have many ups and down during the validation phase. We can eliminate this by changing weights and/or adding more fully connected layers.


Profiling report is in the repo with PDF file.

### Results
Result is pretty good, as I was using ml.g4dn.xlarge to utilize the GPU of the instance, both the hpo jobs and training job did't take too much time.


## Model Deployment
Model was deployed on the ml.g4dn.xlarge instance with best result. First version of my deployment didn't work, so I need to add new inference script call [endpoint.py](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/endpoint.py) to set up and deploy a working endpoint.

![deployed endpoint](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/active-endpoint.png)

The quarying result is shown in the notebook and here is the cloudwatch log of the inference.
![Cloudwatch Log](https://github.com/htinaunglu/DogBreed-Classification-With-Amazon-Sagemaker/blob/main/images/endpoint-cloudwatch-log.png)

## Using the Endpoint
To use the endpoint, just pass a local image of dog, or the address of the jpg image to the endpoint. Every processing job is done in the inference script and user can enjoy the result easily.
Here is an procedure of using the deployed Endpoint.
we first open an image as file 
`with open("./testdog.jpg", "rb") as f:
payload = f.read()`
so the payload becomes a bytetype file and pass it to the endpoint with `predictor.predict(payload)` command, the `endpoint.py` script check it and convert it to image type using PIL module. After that, the transformation takes place by cropping and converting the image to tensors. The function then call the model, predict and return the prediction as array of the 133 breeds of dogs, and we take the value of max one by selecting with `np.argmax`.

**Thank You So Much For Your Time!**
