import torch
import torch.nn.functional as F

# open dataset
words = open('names.txt', 'r').read().splitlines()

# creating itos and stoi dictionaries
itos = sorted(list(set(''.join(words))))
itos = {i+1:s for i,s in enumerate(itos)}
itos[0] = '.'
stoi = {i:s for s,i in itos.items()}

# generator
g = torch.Generator().manual_seed(7)

# training set of trigrams: for (a, b) the label is c
xs, ys = [], []
for word in words:
  nw = ['.'] + list(word) + ['.'] # . is special character for starting and ending a word
  for a, b, c in zip(nw, nw[1:], nw[2:]):
    xs.append([stoi[a], stoi[b]])
    ys.append(stoi[c])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = len(xs)
print(f'Number of examples: {num_examples}')

# encode xs: for this, we encode first character and second character separately
# and then concatenate those encodings to form one.
def encode (ch1, ch2):
  ch1 = ch1 if isinstance(ch1, torch.Tensor) else torch.tensor(ch1)
  ch2 = ch2 if isinstance(ch2, torch.Tensor) else torch.tensor(ch2)
  hc1 = F.one_hot(ch1, num_classes=27) # 1x27 vector
  hc2 = F.one_hot(ch2, num_classes=27) # 1x27 vector
  return torch.cat((hc1, hc2), dim=0) # 1x54 vector

xenc = []
for ch1, ch2 in xs:
  xenc.append(encode(ch1, ch2))

xenc = torch.stack(xenc).float() # (num_examples, 54)-matrix

# initialize neural network
W = torch.randn((54, 27), requires_grad=True, generator=g)

# forward pass
logits = xenc @ W     # log counts
counts = logits.exp() # counts
probs = counts / counts.sum(1, keepdims=True)
loss = -probs[torch.arange(num_examples), ys].log().mean()

print("Initial loss = ", loss.item())

# train
for i in range(10000):
  # forward pass
  logits = xenc @ W     # log counts
  counts = logits.exp()  # counts
  probs = counts / counts.sum(1, keepdims=True)
  loss = -probs[torch.arange(num_examples), ys].log().mean()
  loss += 0.01 * (W**2).mean()

  # backward pass
  W.grad = None
  loss.backward()

  # update weights
  W.data += (-10 * W.grad)

  if i % 50 == 0:
    print(f'Epoch {i}, Loss: {loss.item()}')

print("Final loss:", loss.item())

# sample words
def sample(generator, W, itos, stoi, num_samples=10, max_length=20):
  for _ in range(num_samples):
    out = []
    ixs = [0, 0]            # start with two special start characters (..)
    predpred = 0

    while True:
      # one-hot encode the current bigram
      xenc = encode(ixs[-2], ixs[-1]).float()
      logits = xenc @ W       # logits for the next character
      counts = logits.exp()   # convert logits to counts
      probs = counts / counts.sum()  # get probability distribution for the next char

      ixpred = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item()
      out.append(itos[ixpred])
      if ixpred == 0:
        break

      ixs.append(ixpred)

      if len(out) > max_length:
        break

    print(''.join(out))

# example usage
sample(g, W, itos, stoi, num_samples=30)
