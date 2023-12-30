<!-- ABOUT THE PROJECT -->
## About the project

This repository contains three scripts that enable the following:

1. Download and resize images from a custom CSV file containing URLs to images.
2. Use a pretrained CNN ```InceptionV3``` to train a classifier on the downloaded images.
3. Parse a new image through the trained model and predict the class of the image.

The original purpose of this project was to create a classifier that could classify different parts of a handbag, such as a brand label, zipper, made-in label, overall picture of the handbag and others.

<!-- Data -->
## Data
The dataset used in this project contains ~78K images of 13 different handbag parts. The data, along with a JSON file representing image paths and labels can be downloaded via the link found in the submitted assignment document on the first page. Unfortunately, the full data could not be shared because the company that owns it considers it highly sensitive. Therefore, a subset of the data is provided for the purposes of reproducibility.

<!-- USAGE -->
## Usage
To use the code you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment for the project. The code has been thoroughly tested and verified to work on a Mac machine running macOS Ventura 13.1. However, it should also be compatible with other Unix-based systems such as Linux. If you encounter any issues or have questions regarding compatibility on other platforms, please let me know.

1. Clone repository
2. Run ``setup.sh``
3. Run ```datapull.py``` or configure your data manually
4. Run ```nn.py```
5. Run ```pred.py```

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/sashapustota/pretrained-cnn-image-classification
cd pretrained-cnn-image-classification
```

### Run ```setup.sh```

The ``setup.sh`` script is used to automate the installation of project dependencies and configuration of the environment. By running this script, you ensure consistent setup across different environments and simplify the process of getting the project up and running.

The script performs the following steps:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the required packages
4. Deactivates the virtual environment

The script can be run using the following lines in the terminal:

```bash
bash setup.sh
```

### Run ```datapull.py``` or configure your data manually

The ```datapull.py``` script allows you to download any data from a CSV file containing URLs to images. The script downloads the images and resizes them to 224x224 pixels. The script also creates a JSON file containing the image paths and labels, to be used in the latter scripts.

When running the script, you need to specify the name of your CSV file containing image URLs, the column with the URLs and the column that contains the labels for the images.

The script can be run using the following lines in the terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/datapull.py -d <your value> -u <your value> -l <your value>
deactivate # Deactivate the virtual environment
```

Make sure to include the file extension. For example, if you want to use a CSV file called "data" with columns "link" (where URLs are contained) and "class" (where labels are contained) run the following line in the terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/datapull.py -d data.csv -u link -l class
deactivate # Deactivate the virtual environment
```

Alternatively, you can add your data manually. To do so, simply create an "images" folder and add your image data there. For this code to be compatible with you data, you also need to have a JSON file with paths to images and their respective labels. The structure of the JSON file needs to be as follows:

```
{"image_0.jpg": {"label": "X", "path": "images/image_0.jpg"}, "image_1.jpg": {"label": "Y", "path": "images/image_1.jpg"}, "image_2.jpg": {"label": "Z", "path": "images/image_2.jpg"}}
```

### Run ```nn.py```

The ```nn.py``` script performs the following steps:

1. Load the data
2. Preprocess & augment the data
3. Load the model
4. Fit the models
5. Save the model, a classification report, as well as plots of **loss** and **accuracy** metrics during the training of the model to the master folder.

The script can be run using the following lines in the terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/nn.py
```

#### Customizing model parameters

The ```nn.py``` script is designed to run the models with the default parameters. However, it is possible to customize the parameters by changing the values in the scripts or by passing the parameters as arguments in the terminal.

The following parameters can be customized:

* ```--batch_size -b``` - The number of samples per gradient update. If unspecified, ```batch_size``` will default to 32.
* ```--train_samples -trs``` - The total number of samples in the training dataset. If unspecified, ```train_samples``` will default to 50000.
* ```--val_samples -vs``` - The total number of samples in the validation dataset. If unspecified, ```validation_samples``` will default to 7000.
* ```--test_samples -tes``` - The total number of samples in the test dataset. If unspecified, ```test_samples``` will default to 2000.
* ```--epochs -e``` - The number of epochs to train the model. If unspecified, ```epochs``` will default to 15.
* ```--early_stop -es``` - The number of epochs with no improvement after which training will be stopped. If unspecified, ```early_stop``` will default to 5.

To pass the parameters as arguments in the terminal, simply run the following lines in your terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/main.py -b <your value> -trs <your value> -vs <your value> -tes <your value> -e <your value>
deactivate # Deactivate the virtual environment
```

### Run ```pred.py```

To successfully run the ```pred.py``` script, you need to first have successfully trained the model on your image data using the ```nn.py``` script, as the ```pred.py``` script loads the model saved by the ```nn.py``` script from the master folder. Additionally, the images you wish to make predictions on need to be located in the ```pred``` folder.

The ```pred.py``` script performs the following steps:

1. Loads the image
2. Preprocesses the image
3. Loads the model
4. Makes a prediction
5. Prints out top 5 predictions and their probabilities to the terminal

The script can be run using the following lines in the terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/pred.py -i <your value>
deactivate # Deactivate the virtual environment
```

Where ```<your value>``` is the name of the image you wish to make predictions on. Make sure to include the file extension. For example, if you want to make predictions on an image called "image_0.jpg", run the following line in the terminal:

```bash
source ./gucci-armani-cnn-venv/bin/activate # Activate the virtual environment
python3 src/pred.py -i image_0.jpg
deactivate # Deactivate the virtual environment
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure:
```
│   README.md
│   requirements.txt
│   setup.sh
│
├──pred
|
├──images
│
└──src
      datapull.py
      nn.py
      pred.py

```
<!-- FINDINGS -->
## Findings

When training the model on the handbag data with the default parameters, the following results were obtained:

```
                       precision  recall     f1-score   support
  authenticity-card    0.0139     0.0116     0.0127     86.0
        brand-logo     0.0805     0.0964     0.0878     197.0
            button     0.0521     0.0376     0.0437     133.0
          dust-bag     0.0594     0.0561     0.0577     107.0
hardware-engravings    0.0857     0.1207     0.1002     174.0
   hologram-Label      0.0000     0.0000     0.0000     100.0
     inside-label      0.1004     0.1837     0.1299     245.0
     logo-texture      0.0000     0.0000     0.0000     24.0
    madein-label       0.0000     0.0000     0.0000     176.0
 overall-picture       0.0875     0.1398     0.1077     236.0
   serial-number       0.1111     0.1027     0.1067     185.0
zipper-head-back       0.1163     0.0340     0.0526     147.0
zipper-head-front      0.0794     0.0790     0.0792     190.0
                             
         accuracy       0.0845     0.0845     0.0845        
        macro avg       0.0605     0.0663     0.0599    2000.0
     weighted avg       0.0716     0.0845     0.0738    2000.0

```
