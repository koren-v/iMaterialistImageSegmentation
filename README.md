# iMaterialist
## or CUDA out of memory

You can find solution of test task in the [notebook](https://github.com/koren-v/iMaterialistImageSegmentation/blob/master/unet.ipynb). In short, we used unet architecture with pre-trained resnet50 encoder, we tested size of input image of 224 and 1epoch dured about 1 hour. Also, there were attemps to try size of 512, but cuda out of memory hepened. More detailed explanation in the notebook.
Obviusly, to get better performance we have to train more, train on bigger images and also load all dataset (we used just 25%). Also having more time, we would try diferent hyperparameters.
