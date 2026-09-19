import torch
import torch.nn as nn
from torch.nn import functional as F
import keyboard
import time
with open('Shakespeare.txt') as inp:
  text = inp.read()

#hyperparameters
batch_size = 64 #of samples being trained on each time
block_size = 256 #number of characters in each sample
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' #trains on gpu instead of cpu if available
print(device)
eval_iters = 200 #used to get the average loss in loss evaluation
n_embd = 384 #the number of dimensions of the vector space we embed our tokens into
n_head = 8 #number of heads in each block
n_layer = 6 #number of blocks
dropout = 0.2 #percentage of nodes that are switched off

#creates a list that contains all characters used in the entire text
#set() finds all the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)

#converts text into tokens by indexing the letters in the chars list
encode = lambda s: [chars.index(c) for c in s]
#converts the list of index values back into a string of text
decode = lambda i: ''.join([chars[c] for c in i])



#Data setup
#converts all of the shakespeare text into a tokenized torch vector
data = torch.tensor(encode(text), dtype = torch.long)

#splits shakespeare file into train and val (90% of data for train)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#creates the system for collecting batches of data
torch.manual_seed(1337) #allows me to obtain the same random number generation as karpathy


#collects a batch of data and the targets from either train or val data
def get_batch(split):
  #determines if data is train or val
  data = train_data if split == 'train' else val_data

  #creates a random number for where the blocks should be sampled (1 random number for each block since the batch creates several blocks)
  ix = torch.randint(len(data) - block_size, (batch_size,)) # (1, Batch_size)

  x = torch.stack([data[i:i+block_size] for i in ix]) #collects the corresponding blocks for each random position in ix
  y = torch.stack([data[i+1:i+1+block_size] for i in ix]) #collects the targets for each corresponding block
  x, y = x.to(device), y.to(device) #x, y == (block_size, Batch_size)
  return x,y

@torch.no_grad() #this tells torch to exclude this segment of code from gradient descent as torch reads all of the code for parameters
def estimate_loss(): #gets the average loss of #eval_iters samples in order to report a more accurate loss since individual samples are more or less lucky
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




#creating the AI model
#One head of attention
class head(nn.Module):
    def __init__(self, head_size):
        #obtain Keys, Queries, and Values
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x) #B,T,C
        q = self.query(x) #B,T,C
        #compute the attention affinity (how interesting each token is for other tokens)
        wei = q @ k.transpose(-2,-1) * C**-0.5  #(B,T, C) x (B,16,C)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #B,T,T
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform the weighted transfer of information
        v = self.value(x) #B, T, C
        out = wei @ v #(B, T, T) x (B, T, C)
        return out
    

#multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
       super().__init__()
       self.heads = nn.ModuleList([head(head_size) for _ in range(num_heads)])
       self.proj = nn.Linear(head_size * num_heads, n_embd)
       self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       out = torch.cat([h(x) for h in self.heads], dim = -1)
       out = self.proj(out)
       return out


#Feed forward
class FeedForward(nn.Module):
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


#one block of multiheaded attention calculation
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



#does the computation after self attention has been utilized for communication
class BigramLanguageModel(nn.Module):
  #firstly, the tokens will be embedded into a vector space where each token becomes a vector of size vocab_size
  #defines the embedding table
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #n_embd determines the value of C
    self.position_embedding_table = nn.Embedding(block_size, n_embd)#^ each token will find its row (token 3 goes to 3rd row etc) and grabs its corresponding embedding vector of n_embd size
    self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head,) for _ in range(n_layer)])
    self.ln = nn.LayerNorm(n_embd) #the last layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

    #performs the embedding
  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx) #B, T, C
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #T,C
    x = tok_emb + pos_emb #B,T,C
    x = self.blocks(x) #runs the 3 blocks of multiheaded attention
    x = self.ln(x)
    logits = self.lm_head(x) #this converts the outputs into vectors that are a length that can be decoded
    #^shape is Batch, T(#Of trains per block since we start with one context size and work up to block size, Channels (the length of the vectors that are being grabbed)
    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      #converting the B,T,C to a BT,C because that is what the cross,entropy function is designed to use
      logits = logits.view(B*T, C)
      #converting targets to one dimension
      targets = targets.view(B*T)

      loss = F.cross_entropy(logits, targets)
      #^ measures the quality of the logits w.r.t. the targets

    return logits, loss #logits is the term for tokens that have been converted to vectors (tokens that have been embedded)

  def generate(self, idx, max_new_tokens):
    #idx is the B,T array of indices in the current context
    for _ in range(max_new_tokens):
      #crop idx to obtain the last block_size tokens
      idx_crop = idx[:, -block_size:]
      #get predictions
      logits, loss = self(idx_crop)
      #look only at the last time step
      logits = logits[:,-1,:] #becomes a B,C array
      #convert predictions into probabilities with softmax
      probs = F.softmax(logits, dim=-1) #B,C shape
      #get a sample from the probabilities
      idx_next = torch.multinomial(probs, num_samples = 1) #B,1
      #add these next values to the indices
      idx = torch.cat((idx, idx_next), dim = 1) #B, T+1
    return idx


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
#creating an optimizer (Gradient Descent is one version but AdamW is another)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(max_iters):
  #collect tokens and targets
  xb, yb = get_batch('train')
  #evaluate loss
  logits, loss = model(xb,yb)
  #zero out gradient from previous step
  optimizer.zero_grad(set_to_none=True)
  #collect gradients for all parameters
  loss.backward()
  #nudge the parameters through backprop
  optimizer.step()
  if steps % eval_interval ==0:
    losses = estimate_loss()
    print(losses, steps)

#print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_new_tokens=1000)[0].tolist()))

while True:
    start = input('\nStart the story!\n')
    working_text = torch.tensor([encode(start)], dtype = torch.long, device = device)
    print(start, end = '')
    while keyboard.is_pressed('n') == False:
      working_text = (m.generate(idx = working_text, max_new_tokens=1))
      print(decode(working_text[-1].tolist())[-1], end = '')
      time.sleep(0.01)
      if keyboard.is_pressed('q'):
        quit()
    else:
      continue





