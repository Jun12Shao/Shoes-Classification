## Training a Fully Convolutional Neural Network

### Install dependencies

* Install [PyTorch and torchvision](http://pytorch.org/).
* Install imblearn, PIL, opencv-python,onnx, onnxruntime


### Set up the shoes dataset
* Run `preprocessing.py`. This will generate the name list and labels for training set(train/valid=9:1) and test set. Validation set is used for best model selection.

### Train models

(1) basic_CNNs
* Run `python train.py --method='Basic_CNN' --best_model='cnn-'  --epochs=100 --learning_rate=0.0001`. This will train a CNN model on the training set, place the results into `results` and save the best model in "checkpoints".


(2) ResNet_CNNs
* Run `python train.py --method='ResNet_CNN' --best_model='ResNet-cnn-' --epochs=150 --learning_rate=0.001`. This will train a shallow CNN. A pretrained ResNet18 will be used to extract features (512,8,8)  for the CNN.


### Evaluate
We adopt accuracya and average F1 as the measuring metrics.
(1) For Basic_CNNs
* Run evaluation as: `python tests.py --method='Basic_CNN' --best_model= './checkpoints/cnn-81.onnx'`.

(2) For ResNet_CNNs
* Run evaluation as: `python test.py --method='Basic_CNN' --best_model='./checkpoints/ResNet-cnn-109.onnx'`.

### Experiment results
Models        	Test Accuracy           Average F1 score
Basic CNN	75.44%			75.30%			
ResNet_CNN	85.09%			84.88%


