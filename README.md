# GPT from Scratch

**Skills learnt and applied** - Pytorch, Encoder-Decoder Architecture, Text handling, Transformer Architecture, Self-Attention & Multi-head Attention Layer, Feed-Forward Neural Network.

## Idea 
This is a basic version of a GPT (Generative Pretrained Transformer) model. It is based on the same **Transformer Architecture**, which consists of a **Self-Attention Layer** and a **Feed-Forward Neural Network**. It was pretrained to generate text autoregressively, meaning it generates one token at a time based on the previously generated tokens.

### IMP Note
**This is not a program where we use an already available LLM API and fine-tune it to create our own local chatbot, instead here we are writing the very algorithm from scratch that are used to make an LLM in the first place.**

This project was done on google colab in order to use the GPU resources available on Colab.

## Installing and importing the libraries
First we need to install and import the necessary libraries.

```
pip install python-docx
```
To read doc files we need to install `python-docx` library and then we need to import `docx`. Also we need the to import `torch` as we are going to use the pytorch library for this project.   

```
import docx
import torch
```

We need to import the `torch.nn` package as this package provides a variety of classes and functions to help us create and manage neural networks. From the `nn` class we need the `functional` module. This module provides a collection of useful functions that are commonly used in Neural Network operations.

```
import torch.nn as nn
from torch.nn import functional as F
```

## Define Hyperparameters

```
batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 100
learning_rate = 1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1
```

The hyperparameter values are important as they allow us to tweak our model for optimal results.`batch_size` defines the number of samples that will be passed through the network at once before the weights are updated. `block_size` defines the length of the input sequences or tokens that the model will process. `max_iters` sets the maximum number of iterations (or steps) for training the model wherein each iteration processes one batch of data and updates the model’s parameters. `eval_interval` specifies after how many iterations the model will be evaluated. The `learning_rate` controls how much the model’s parameters are adjusted with respect to the loss gradient. Smaller learning rates provide stable and precise updates but are slower to converge.

There are some more hyperparameters that we need to set. `eval_iters` specifies the number of iterations we run during evaluating the the model’s performance on a validation set. We run the evaluation loop for number of iterations to get a reliable estimate of the model’s performance. `n_embd` stands for embedding dimension and in Neural Networks it defines the size of vector space in which the words/tokens are represented. A higher value of `n_embd` can capture more nuanced relationships between tokens but also increases the computational complexity. `n_head` in the context of transformer model is the number of attention heads in a multi-attention mechanism. Having multiple heads allows the model to focus of multiple parts of input sequence simultaneously which improves the model's ability to capture relationships between tokens. `dropout` is a regularization technique used to prevent overfitting and helps the model to generalize better. It does this by randomly setting a fraction of the input units to zero at each update during the training time.
 
```
torch.manual_seed(108)
```

So we seed the environment for ensuring reproducability. The number in the seed is set to any number that we like and is not changed so that the random processes in the program give the same output. Random processes like **Weight Initialization**, **Data Shuffling**, **Dropout** are always the same due to seeding. This is needed so that we can check that differences in results when we tweak the hyperparameters are because of hyperparameters only and not due to some random variations.

## Load the document and sort unique characters

```
doc = docx.Document('/content/Mahabharat annotated .docx')

text = ''
for paragraph in doc.paragraphs:
    text += paragraph.text + '\n'
```

We upload the document which in my case is the doc file of an ancient historical epic of India 
**The Mahabharat**. I chose this epic as it is the largest epic poem on the planet and thus we get good amount of data to train our model. Then we initialize an empty string and iterate over the paragraphs of our document to get text(characters) from every paragraph and concatenate them to our empty string `text`. The characters belonging to one paragraph is separated by that of another by the newline character '\n'. 

We can check the the total number of characters as well by simply printing the length of the text string and also some of the initial characters

```
print("length of dataset in characters: ", len(text))
print(text[:1232]) # Let's look at first 1232 characters
```
**Output**
```
length of dataset in characters:  14111937

The Mahabharata
of
Krishna-Dwaipayana Vyasa

BOOK 1
ADI PARVA


THE MAHABHARATA
ADI PARVA

SECTION I

Om! Having bowed down to Narayana and Nara, the most exalted male being, and also to the goddess Saraswati, must the word Jaya be uttered.

Ugrasrava, the son of Lomaharshana, surnamed Sauti, well-versed in the Puranas, bending with humility, one day approached the great sages of rigid vows, sitting at their ease, who had attended the twelve years' sacrifice of Saunaka, surnamed Kulapati, in the forest of Naimisha. Those ascetics, wishing to hear his wonderful narrations, presently began to address him who had thus arrived at that recluse abode of the inhabitants of the forest of Naimisha. Having been entertained with due respect by those holy men, he saluted those Munis (sages) with joined palms, even all of them, and inquired about the progress of their asceticism. Then all the ascetics being again seated, the son of Lomaharshana humbly occupied the seat that was assigned to him. Seeing that he was comfortably seated, and recovered from fatigue, one of the Rishis beginning the conversation, asked him, 'Whence comest thou, O lotus-eyed Sauti, and where hast thou spent the time? Tell me, who ask thee, in detail.'
```

We needed to see the original file so that when our model prints the learnt output we should able to see and check that the way of writing is similar to the original text or not. Now we will separate and sort out all the unique characters in the document.

```
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
```
**Output**
```
!"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz—
82
```

So there are 82 unique characters in the complete document all of which are shown above. 

## Mapping and creating encoder & decoder
From here on we will actually deep dive into the important parts of the code. First we will **Map the Characters and Integers** which is useful in tasks like text processing.
```
stoi = { ch:i for i,ch in enumerate(chars) }
```
`stoi` stands for "string to integer". The dictionary comprehension iterates over a list or string of characters `char` using `enumerate`, which provides the index(i) and character(ch) thereby mapping each character to its corresponding index by a key(characters) and value(integers) pair. 
```
itos = { i:ch for i,ch in enumerate(chars) }
```
`itos` stands for "integer to string" and is exactly opposite of the `stoi` dictionary. Here the keys are the integers whereas the values are the characters.

Our **Encoder** is a function that takes a string as an input and outputs a list of integers while the **Decoder** is a function that takes a list of integers and outputs a string.   
```
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```
`encode` is a lambda function which takes a string `s` and uses a list comprehension to convert each character in the string to corresponding integer using the `stoi` dictionary. `decode` is also lambda function that takes a list of integers `i` and uses a list comprehension to convert each integer `i` in the list `l` to corresponding character and then concatenates the the characters to form a string.
Let's check them in action:
```
print(encode("Nara Narayana"))
print(decode(encode("Nara Narayana")))
```
**Output**
```
[37, 55, 72, 55, 1, 37, 55, 72, 55, 79, 55, 68, 55]
Nara Narayana
```
As we saw they are working perfectly, let's encode the entire dataset that was kept in `text` variable and store it into torch.Tensor
```
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:10])
```
**Output**
```
torch.Size([14111937]) torch.int64 
tensor([ 0,  0, 43, 62, 59,  1, 36, 55, 62, 55])
```
The size of the tensor is exactly equal to the length of `text` variable we checked earlier so that means everything is correctly encoded.

## Training and Validation Split

We will split our data into training and validation sets wherein 90% of our data is for training and 10% is for validation.
```
n = int(0.9*len(data))
train_data = data[:n] 
val_data = data[n:]
```

## Generating Data Batches

Since we have created encoder-decoder architecture we need to create a function that splits our data in small batches of `data`, checks or takes as input the first integer from this batch and prints out the next integer as output. We need this to make our model learn which integer is followed by which integer. These integers are nothing but strings from our `data` that are converted to integers by using `stoi` dictionary. So by being able to learn which integer follows which one our model is actually learning which string is following which one. This learning will let the model to predcit strings autoregressively i.e the model will generate one token at a time based on previously generated tokens. 

```
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

In the above code snippet, first we choose between `train_data` and `val_data` using the split parameter of function `get_batch`. Then we generate the random starting index for the batches of size `batch_size`.Then it stacks slices of data in input `x` and target `y` and sends them to the device that is specified.

## Function to compute average model loss on training and validation dataset

So we will create a function `estimate_loss` to evaluate the performance of our model on the training and validation dataset. We need this for montioring the model's performance during training. We will do this by calculating the average loss over many iterations.

```
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

`@torch.no_grad()` is a decorator that disables the gradient calculation which decreases the memory usage and helps in speeding up the computations. We used it as in our evaluation there is no need for gradients. Then we initialize a dictionary for the output and set our model to `eval` mode which acts differently during the training and validation phase and affects the dropout and batch normalization layers. We loop over both the training and validation set and initialize our loss tensor which will store the loss values for each evaluation iteration. We then loop through each iteration and during each iteration we get our `X`, `Y` from the `get_batch` function, we also compute the predictions(`logits`) and losses of our model, and store the loss value in the `losses` tensor. Then outside of the evaluation loop we compute the average loss over all iterations and keep it in our `out` dictionary. At the end we set the model to training mode.

## Attention is All You Need
Self-attention layer allows the model to weigh the importance of different words in a sequence relative to each other. This is how the model knows which next word is to be printed as many words can come after the given word. So this importance helps the GPT to know the context of each word in the sequence by considering its relationship with other words. It does this by transforming each word in the sequence into 3 vectors **Query(Q), Key(K) & Value(V)**. 


















