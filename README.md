# Deep Learning Based Sentiment and Sarcasm Analysis

In this Jupyter Notebook, we employ deep neural network architecture for sentiment analysis and sarcasm detection at the sentence level. We detail each stage of the process, including selecting and importing training data, preprocessing the data, creating the model architecture, training the model, and using the model for new text.

## 1. Selecting and Importing Training Data

### i. Import Libraries

We begin by importing necessary libraries such as pandas, numpy, tensorflow, keras, nltk, and others for data manipulation, deep learning, and natural language processing tasks.

### ii. Selecting the Datasets

We select two datasets for sentiment analysis and two for sarcasm detection:
- **Sentiment Datasets:** Yelp reviews and Amazon reviews
- **Sarcasm Datasets:** Reddit sarcastic comments and The Onion headlines

We load these datasets, clean them, and prepare them for preprocessing.

## 2. Preprocessing the Training Data

### i. Upsample Dataset

We address class imbalance by upsampling the minority classes in both sentiment and sarcasm datasets. This ensures balanced representation of different sentiment labels and sarcasm/non-sarcasm labels.

### ii. Data Preprocess

We preprocess the data by removing URLs, emails, new line characters, and single quotes. We tokenize the sentences, remove stopwords, and lemmatize the words to prepare them for the model.

### iii. Tokenize Words

We tokenize the words in the preprocessed data and pad the sequences to ensure uniform length for input to the neural network.

### iv. Label Encoding

For sentiment analysis, we convert the labels into one-hot encoded vectors. For sarcasm detection, binary labels are used.

### v. Train-Test Split

We split the data into training and testing sets for both sentiment analysis and sarcasm detection.

### vi. Embedding

We use pre-trained GloVe word embeddings to create embedding matrices for both sentiment analysis and sarcasm detection.

## 3. Machine Learning Algorithm

We design a deep learning architecture with bidirectional LSTM layers for both sentiment analysis and sarcasm detection.

- **Sentiment Branch:** Bidirectional LSTM layers followed by dense layers for sentiment prediction.
- **Sarcasm Branch:** Bidirectional LSTM layers followed by dense layers for sarcasm detection.

We compile the model with appropriate loss functions and metrics for each output branch.

## 4. Training the Model

We train the model using the prepared data. We monitor training progress using callbacks and visualize the loss function's progress after each epoch.

## 5. Using Model for New Text

We apply the trained model to new text data. We extract text from images using OCR, preprocess it, tokenize, and pad it. Then, we feed it into the model for sentiment analysis and sarcasm detection.

## Conclusion
This project demonstrates the use of deep learning techniques for sentiment analysis and sarcasm detection tasks. By training a model on diverse datasets and leveraging Bidirectional LSTM layers, the model is able to effectively analyze sentiment and detect sarcasm in text data.
