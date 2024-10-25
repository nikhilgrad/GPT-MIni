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
Self-attention layer allows the model to weigh the importance of different words in a sequence relative to each other. This is how the model knows which word is to be printed next, from the many words that can come after the given word. So this importance helps the GPT to know the context of each word in the sequence by considering its relationship with other words. It does this by transforming each word in the sequence into 3 vectors **Query(Q), Key(K) & Value(V)**. These 3 vectors are obtained by first converting each word into an embedding vector and then calculating the dot product of this embedding vector with 3 different sets of learned weight matrices, namely WQ, WK and WV. Query represents what the current word is looking for in other words, Key represents the characteristics of each word that can be matched against the Query and Value represents the actual information of each word that will be used to compute the final output. The dot products of Query and key are multiplied together and scaled by the square root of dimension of key vectors as this stabilizes the gradients. Then softmax function is applied to this scaled scores in order to convert them into probabilities as this will make their sum to be 1 which will make them easier to be interpreted as weights. This weights that is obtained is what we call an attention score and this is where the whole mechanism gets its name as the **Attention Layer**. This attention scores or weights are then used to compute the weighted sum of the Value vectors which is the final output for each word. 

![self-attention_layer](https://github.com/user-attachments/assets/813723cb-adb3-4a94-bf8c-0ca6aa09af6c)

*A Self-attention head is shown in the above picture*

### Self-attention head
Now we will create one self-attention layer also called as one Head as it represents a single attention mechanism in a multi-attention mechanism where multiple such heads will be present. It is called as **self-attention** because the keys and values are produced from the same source as queries. For this purpose we will create a class which will transform our input embeddings into Key, Query and Value vectors and then carry out the computation of attention scores and find the value vectors using attention weights.

```
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
```

The class `Head` inherits from `nn.module` which is the pytorch neural network module. Using this module we create 3 linear layers (`self.key`, `self.query`, `self.value` ) for transforming the input embeddings into Key, Query and Value vectors respectively. `tril` represents the lower triangular matrix which ensures that the model cannot see the future tokens when predicting the current tokens. As this block here is a **decoder** attention block which is used in autoregressive settings we can convert it to an **encoder** attention block by just deleting this single line that does the masking with `tril` allowing all tokens to communicate. It does this by masking the future positions in the sequence. A dropout layer is used to prevent the overfitting by randomly setting some attention weights to zero during training. The input `x` has the shape (B,T,C) where B is the batch size, C is the sequence length and C is the number of features i.e. embedding dimension. Using the previously created linear layers we transform the input `x` into query, key and value vectors. Attention scores are computed by taking dot product and scaling it. This results in a matrix of shape (B,T,T) where each element represents the attention score between 2 positions in the sequence. Then we mask it by setting attention scores to `-inf`, which ensures that the model only attends to previous and current positions. The masked attention score is then passed through a softmax function to obtain weights and dropout is applied. This is then used to compute the final output by multiplying with value vectors.

### Multi-Head Attention
As we have already created a single attention layer head now let's create multiple such heads that will be working in parallel for better performance.

![multi-head_attention_layer](https://github.com/user-attachments/assets/f4a9e428-8142-41a0-9959-8f5b5d65d6a8)

*Multi-Head Attention layer*

For this we create a class which again inherits from the pytorch `nn.module` neural network module representing the multi-head mechanism used in Transformer models.

```
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

`self.heads` is a list of `Head` instances, each representing a single head of self-attention. `self.proj` is a linear layer that projects the concatenated outputs of all attention heads back to the original embedding dimension (`n_embd`). The input `x` is passed through each attention head in `self.heads` using a list comprehension. The outputs of all attention heads are concatenated along the last dimension `dim=-1` which is then passed through the projection layer `self.proj` to map it to original embedding dimension `n_embd`.

```
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

The above defined classes are just a simple feedforward network and a single transformer block that contains a muli-head attention followed by the feedforward network. `self.ln1` and `self.ln2` are layer normalization layers for stabilizing and accelerating the training. In this the `forward` function is used to create residual connections as this helps in training deeper networks and solving the problem of vanishing gradients. This is done again for feedforward network. 

### Bigram Language model
This is be the last class we have to make and this would be a simple language model that predicts the next token in a sequence based on current context. This class will also  be an inheritance from Pytorch neural network module. 

```
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

It maps each token in the vocabulary to an embedding vector and provides positional information to the model by mapping each position in the sequence to an embedding vector. `self.blocks` is a sequence of `Block` instances, each of which contains a multi-head self-attention mechanism and feedforward network. Then applies final layer normalization and maps the final embeddings to vocabulary size producing logits for each token in the vocabulary forward function takes an input of shape (B,T) where B is batch size and T is sequence length. Initialize token embeddings and psoitional embeddings as `tok_emb` and `pos_emb` respectively, which are then summed to form the input and passed through transformer blocks. This are then normalized and passed through the linear layer to produce logits (logits are direct outputs from final layer that haven't been transformed into probabilities). Then we define a function `generate` that generates new tokens based on the current context by iteratively predicting and sampling the next token.

```
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step
```

The above code snippet initializes and trains the `BigramLanguageModel` using a standard training loop and also evaluates the loss. It samples data, computes gradients and updates the model parameters. 

Finally we generate the model, calling the `generate` method of the model and `decode` function.

```
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
```

This will print the output i.e. the generated tokens according to number of tokens specified by `max_new_tokens`. 

The following is the output obtained after training for 5000 steps.

**Output**
```
SECTION LIII

Vaipaudeva said, "Indeed, then, knew by What, O Duryodhana, plaughtere foes with both nisted injuge and profit of the whole objects. In what the Suta's son of the earth whetter is he is Citrana.'"

SECTION XVI

"Hearing many wifal unto one that abound with irresible at these the gods. O bull thou canting interrofal and be incovered by me. What be what what I know are got and the grant and complexion. Griet shill, that bulls thou shall ever constly, Opinerce one, forwarth, when now blossed by the means of my advancing Ashtavathama, Thas are very rend roars of gold closse, immovable proceeds, the king to Dusasana have (been by the rays of the bodies), and other, with sharp be combated and righteousness of arrows. Armed with strength and effects in retror fire in the track race. And these excellent cannot fear from his enembled its wrongs and children sended thou shouts me he influence with even he hard. Those deity to be, heard these weapons do now, in one limps of husbulilary. The mighty bestowing those thus kingdom from onten the fiery delight of his deer-imper disragrection. Do thou a tattribute to doing addressed with weapons forsament so behold and swalled by the man-bodies and monkey and homage, of her snake, (ex-remeply) dog. The all about of duni, let thy slaughter me. Give, have cut off hard bour performing savat, tell me!'
```

We can match the output with original file and see that the way of writing and also mentioning of the section numbers is similar to the original file which means the model has learnt the semantics from the document but we can see that the output is not meaningful which means to understand the general grammar, complexities of language and form meaningful sentences it needs more training and even more data. So we have achieved our goal of building a transformer decoder architecture, trained it on our own data and also successfully predicted the learnt output which follows the semantics of original epic. 














