# Image Classifier
> This is the final project of the PyTorch Scholarship Challenge by Facebook and Udacity

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application.

The **original repo** of this challenge can be found [here](https://github.com/udacity/pytorch_challenge).


## TL;DR
The project was done using [Google Colaboratory](https://colab.research.google.com/) and its code can be found at the [Notebook](https://github.com/psatler/PyTorch_Challenge_Image_Classifier/blob/master/Facebook_PyTorch%60s_Challenge_Image_Classifier_Project.ipynb) in this repo.

Throughout the Notebook there are instructions guiding to executing the project.

## GPU
As the network makes use of a sophisticated deep convolutional neural network the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have three options:
1. **Cuda**: If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads).
2. **Cloud Services**: There are many paid cloud services that let you train your models like [AWS](https://aws.amazon.com/) or [Google Cloud](https://cloud.google.com/).
3. **Google Colab**: [Google Colab](https://colab.research.google.com/) gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have ellapsed you can just reload and continue! The only limitation is that you have to upload the data to Google Drive and if the dataset is massive you may run out of space.

Once a model is trained, then a normal CPU can be used to perform predictions and you will have an answer within some seconds.

## Pretrained Network
It was used the `Densenet121` pretrained model from `torchvision.models` using the so-called _Transfer Learning_. Examples on how to do this can be found at [PyTorch's Documentation](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Hyperparameters
There is a wide selection of hyperparameters avaiable and different configurations. 

- Model for transfer learning: densenet121
- Definition of classifier: one hidden layer, followed by a ReLU function with 25% dropout rate as shown below
```
Sequential(
  (fc0): Linear(in_features=2208, out_features=256, bias=True)
  (relu0): ReLU()
  (drop0): Dropout(p=0.25)
  (output): Linear(in_features=256, out_features=102, bias=True)
)
```
- Batch size: 32
- Criterion: [Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#crossentropyloss). The criterion is the method used to evaluate the model fit.
- Optimizer: [_Adam_](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam). The optimizer is the optimization method used to update the weights. More info can be found [here](https://pytorch.org/docs/stable/optim.html)
- Scheduler: [_StepLR_](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR). It provides different methods for adjusting the learning rate and step
- Learning Rate: 0.001
- Epochs: 5

Using such hyperparameters, where `learning_rate = 0.001`, `dropout = 0.25`, `optimizer = optim.Adam(model.classifier.parameters(), learning_rate )` and a scheduler such as `lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)`, which decays the learning rate by a factor of 0.1 every 4 seconds, the **training process** took about **11 minutes** with **5 epochs**, reaching an **accuracy** of about **94%**.



## Useful links
- [PyTorch Scholarship Challenge](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/home) official website
- [FAQs PyTorch Challenge](https://github.com/ishgirwan/faqs_pytorch_scholarship/blob/master/Lab.md)
- [Moving Model from Colab to Udacity's workspace](https://medium.com/@ml_kid/how-to-move-our-model-from-google-colab-to-udacitys-workspace-final-lab-project-88e1a0b7d6ab)
- [Deep Learning with PyTorch](https://www.youtube.com/playlist?list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG) playlist from Deep Lizard youtube channel.
- [Resources library](https://docs.google.com/spreadsheets/d/1HnlcuI3I-d3Cli__RxOgMrxmE3aiZ8Vw5ar14WoPVRo/edit): google spreadsheet with several links to different kinds of DL resources.
- Series of 3 articles exploring the PyTorch project: [Part 1](https://medium.com/udacity/implementing-an-image-classifier-with-pytorch-part-1-cf5444b8e9c9), [Part 2](https://medium.com/udacity/implementing-an-image-classifier-with-pytorch-part-2-ae4dd7b2f48), [Part 3](https://medium.com/udacity/implementing-an-image-classifier-with-pytorch-part-3-6ff66106ba89). 
- [A Data Science Project Walk-Through](https://towardsdatascience.com/a-data-science-for-good-machine-learning-project-walk-through-in-python-part-one-1977dd701dbc) in Python. (* Not related to PyTorch's challenge).

## License
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT) © Pablo Satler 2018
