![badge](http://ForTheBadge.com/images/badges/made-with-python.svg) ![badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# article_categorization
This program is used to develop a NLP deep learning model that categorizes articles into 5 categories based on its topic. The model was trained using a selection of BBC articles.

# How to use
Clone the repository and use the following scripts per your use case:
1. train.py is the script that is used to train the model.
2. categorization_modules.py is the class file that contains the defined functions used in the script for added robustness and reusability of the processes used.
3. The saved model, tokenizer and encoder are available in .h5 and .pkl formats respectively in the 'saved_model' folder.
4. Screenshots of the model architecture, train/test results, and TensorBoard view are available in the 'results' folder.
5. Plot of the training and testing process may be accessed through TensorBoard with the log stored in the 'logs' folder.

# Results
The model developed using an embedding layer and 2 hidden bidirectional LSTM layers was scored using accuracy and f1-score, attaining 93.26% accuracy and 0.93 f1-score on the test dataset.

Model architecture:

![model](https://github.com/khaiyuann/article_categorization/blob/main/results/model.png)

Train/test results (achieved 52.29% accuracy and f1 score):

![train_test_results](https://github.com/khaiyuann/article_categorization/blob/main/results/train_test_result.png)

TensorBoard view:

![tensorboard](https://github.com/khaiyuann/article_categorization/blob/main/results/tensorboard.png)

# Credits
Thanks to Susan Li (GitHub: susanli2016) for providing the bbc-text dataset used for the training of the model on GitHub. 
Check it out here for detailed information: https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/blob/master/bbc-text.csv
