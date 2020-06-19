Steps to run:

Step 1: Install the requirements mentioned in the file "requirements.txt"
Step2: Place the dataset files in the folder named "dataset" and then run "bash train.sh" on terminal. It will run for 200 epochs or until Early Stopping. If you want to reduce it the number of epochs, just edit the "bash train.sh" file and change the argument to "-e 100".
At the end, it will print the loss for the test set and it will also save the predicted images in the folder "test_predictions"

If you want to visualize all the processes, open "seep_segmentation.ipynb" using jupyter notebook. Everything is already set up, so you just need to run the whole jupyter notebook file.

Extras:
If you want to run separately instead of bash. 
Run "split.py" and then the below:

Run: "python train.py -help" to know about the command line arguments to be passed.
Example Run: "python train.py -tb 8 -e 100"


Short note on the architecture:

The architecture is based on UNet which is well known for segmentation. It is basically an auto-encoder with skip connections. Intitially the network downsamples the input image by passing through convolutional layers and max-pooling after each convolution.
The learned features at the final downsampling layer is then passed through upsampling convolutional layers, which uses bilinear upsampling after each convolution to enlarge the prediction. To improve accuracy and stability, residual connections are formed between the downsampling layers and the upsampling layers to allow features of different resolutions to be shared between the two segments of the network.
Finally, at the output of the last convolutional layer, softmax activation is applied to produce an 8 channel prediction (for 0-7 classes) with the same height and width as the input image, where each pixel location contains the predicted probability of the class of seep.

The loss function chosen for this task is cross entropy loss. The reason why I chose is that, the minimisation of the cross entropy corresponds to the maximum likelihood estimation of the network parameters. The negative log-likelihood becomes unhappy at smaller values, where it can reach infinite unhappiness (that’s too sad), and becomes less unhappy at larger values. Because we are summing the loss function to all the correct classes, what’s actually happening is that whenever the network assigns high confidence at the correct class, the unhappiness is low, but when the network assigns low confidence at the correct class, the unhappiness is high.


Results:

The data was split into ~70%:~15%:~15% between training, validation and test set resp.
So,
# train images: 553
# validation images : 118
# of test images: 119

I had setup the program to run until 200 epochs. Since the Early Stopping was setup, it was stopped early at 139th epoch as there was no minimum decrease (min_delta=0.001) in the Cross Entropy Loss for straight 10 epochs (patience = 10).

Final Test Set Cross Entropy Loss obtained: 0.0682 (after 139 epochs).

In the "saved_images" folder which contains some visualizations:
File named "predictions.svg" contains the last batch of predictions of the test set. The images shows the segmented seeps and their classes.
'black' - Represents non-seep.
'blue','red', 'green', 'brown', 'cyan','yellow','royalblue' - Represents the classes of seep 1-7 resp.

File named "trainval_loss.svg" contains the plot of cross entropy loss for training and validation set until Early Stopping.