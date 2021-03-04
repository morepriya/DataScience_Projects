# German Traffic Sign Classification Using CNN and Keras
### In this project, I used Python and TensorFlow to classify traffic signs.
### Dataset used: [German Traffic Sign Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This dataset has more than 50,000 images of 43 classes.
### 96.06% testing accuracy.

# Pipeline architecture:
- Load The Data.
- Dataset Summary & Exploration
- Data Preprocessing.
  - Shuffling.
  - Grayscaling.
  - Local Histogram Equalization.
  - Normalization.
- Design a Model Architecture.
  - LeNet-5.
  - VGGNet.
- Model Training and Evaluation.
- Testing the Model Using the Test Set.
- Testing the Model on New Images.

# Step 1: Load The Data
Download the dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). This is a pickled dataset in which we've already resized the images to 32x32.
We already have three .p files of 32x32 resized images:
* train.p: The training set.
* test.p: The testing set.
* valid.p: The validation set.
We will use Python pickle to load the data.

# Step 2: Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:

- 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
- 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
- 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.
First, we will use numpy provide the number of images in each subset, in addition to the image size, and the number of unique classes. Number of training examples: 34799 Number of testing examples: 12630 Number of validation examples: 4410 Image data shape = (32, 32, 3) Number of classes = 43

Then, we used matplotlib plot sample images from each subset.


