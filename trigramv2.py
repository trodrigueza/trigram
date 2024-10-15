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

xenc = xs[:, 0] * 27 + xs[:, 1] # (num_examples, 1)-matrix

# initialize neural network
W = torch.randn((27*27, 27), requires_grad=True, generator=g)

# train / gradient descent
for i in range(10000):
  # forward pass
  logits = W[xenc]       # log counts
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
      xenc = ixs[-2]*27 + ixs[-1]
      logits = W[xenc]       # logits for the next character
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
