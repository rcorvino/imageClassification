# imageClassification
This is a deep neural network (NN) tool classifing input images in 4 categories of philips products:
1. shaver, 
2. smart-baby-bottle
3. toothbrush
4. wake-up-light


## Architectural choices and rationale 

**Keras** library is used to rapidly explore design possibilities.

Due to the reduced number of available input images, training a deep NN is ineffective. To improve the accuracy results, we use **transfer learning**. We try out three well known architectures: VGG16, ResNet50 and InceptionV3.
We also evaluated the **ensembling** of multiple architectures.

Since we used transfer learning, the **hyperparameter space exploration** was limited to the only last layer's regularization parameter.

Finally, to handle input data, we used keras' ImageDataGenerator, which allows for easy data partitioning in train/dev and test sets and provides **data augmentation**.

