# Word-embedding-wgan
A prototype for the seminar Chatbots and Conversational agents

Markus Roth
markus.roth1{ät}@students.unibe.ch

The code is heavily based on https://github.com/caogang/wgan-gp/blob/master/gan_language.py
The torchtext stuff is heavily based on http://anie.me/On-Torchtext/

# My contributions:
* I adapted gan_language.py to not work on characters, but on words
* The tweet dataset is loaded from CSV and cleaned
* A dictionary is built from the tweet dataset, only words of a certain frequency are considered
* The words are embedded using a pre-trained word embedding (GloVE)
* Dimensions of the networks are changed (SEQ_LEN, DIM)
* The output of the generator is no longer a softmax, but simply a linear layer
* The generation of example sequences is done via nearest-neighbor within the word embedding space
* Value logging is done with TensorBoard 
* Training duration progress bar with tqdm

# Prerequisites:
* Internet connection (to automatically download GloVE word embeddings)
* Python 3
* NumPy
* PyTorch
* SpaCY
* tensorboard_logger
* tqdm
* CUDA (for train_cuda.py)

It is probably easiest to install Anaconda, it includes a lot of this (https://www.anaconda.com/)
I didn't try to run this on Windows, only Linux

# Running
To run CUDA version (for systems with a GPU and CUDA configured):
$python train_cuda.py

To run CPU version (slower):
$python train.py
