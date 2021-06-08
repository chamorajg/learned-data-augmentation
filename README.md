
# Learned Data Augmentation.
----------------------------------------------------------------------------------------------------------------------------------------------------------------

> #### To work on Tiny ImageNet dataset, dataset has to be downloaded using the command line. Here are the following steps:
> Download the dataset in the data/raw folder.
>  1. wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
>  2. unzip tiny-imagenet-200.zip ../processed/
> This will store the dataset in the data/processed/folder. 
 
----------------------------------------------------------------------------------------------------------------------------------------------------------------

> To work on CIFAR-10 dataset, dataset can be downloaded automatically using the download argument in the data loader function for the first time.
> Dataloader for the folowing repository outputs 2 images and a label. One of the image is an undistorted image and the other being the transformed image. The third one being the label of the image.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

> Solutions worked on.
>   1. Single Encoder Single Decoder - Augmented observation on the encoder and undistorted observation on the decoder.
>   2. Single Encoder Single Decoder - Undistorted observation on the encoder and augmented observation on the decoder.
>   3. Single Encoder Dual Decoder - Augmented observation on the encoder and both undistorted observation and augmented observation on the decoder.
>   4. Single Encoder Dual Decoder - Undistorted observation on the encoder and both undistorted observation and augmented observation on the decoder.
>
> Of the 4 different solutions I have worked on, the VAE gave satisfactory results using procedure 1.) and 3.). 
> 2.) and 4.) failed when random images from the data were augmented and when multiple augmentations were used on the data. Since encoder part of the VAE encodes the undistorted observation it failed to completely learn the augmentations.
> 1.) and 3.) gave satisfactory results, 3.) gave much more clear and good results compared as the generalization was better compared to 1.).

----------------------------------------------------------------------------------------------------------------------------------------------------------------
> I have attached a pdf document about the solution 1.) attached in the repository inside procedure directory.

> To compare the generalization of various VAEs we used the encoder part of the architecture to classify the images by augmenting fully connected network on top of the encoder replacing the decoder with encoder parameters being frozen.
