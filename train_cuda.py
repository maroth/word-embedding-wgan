import time
import datetime
import unittest
import re
import collections

import numpy as np
from tensorboard_logger import configure, log_value, Logger

from torchtext import data
import spacy
from tqdm import tqdm

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DIM = 25
SEQ_LEN = 10
BATCH_SIZE = 64
ITERS = 200000 # How many iterations to train for
CRITIC_ITERS = 10
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
G_LEARNING_RATE = 1e-4
D_LEARNING_RATE = 1e-4
NOISE_SIZE = 2 * DIM
D_HIDDEN = 15

#load tokenizer data
spacy_en = spacy.load('en')


#tokenize, save all lines 
lines = []
identifier_re = re.compile("@([0-9]){6}")
def tokenizer(text):
    text = text.replace('!', ' ').replace('-', ' ').replace(',', ' ').replace("'", '')
    text = text.replace('  ', ' ').replace('   ', ' ')
    text = re.sub(identifier_re, '', text)
    lines.append(text)
    return [tok.text for tok in spacy_en.tokenizer(text) if not tok.text.isspace()]

TWEET = data.ReversibleField(sequential=True, tokenize=tokenizer, lower=True,
include_lengths=True)

data_set = data.TabularDataset(
    path = 'twcs_cleaned_airasia.csv',
    format = 'csv',
    fields = [
        ('tweet_id', None),
        ('author_id', None),
        ('inbound', None),
        ('created_at', None),
        ('text', TWEET),
        ('in_response_to_tweet_id', None)
    ])

#load vocabulary. use GloVe embeddings, only consider words that appear 15 times at least
TWEET.build_vocab(data_set, vectors='glove.twitter.27B.25d', min_freq=15)

vocab = TWEET.vocab
print('vocab size:', len(vocab))

#these are words that are in the tweets but not in the embedding dataset
print('unknown embeddings:', len(set(
    [word for word in vocab.itos if vocab.vectors[vocab.stoi[word]].equal(torch.FloatTensor(25).fill_(0))]
)))
        

vocab = TWEET.vocab
embed = nn.Embedding(len(vocab), DIM).cuda()
embed.weight.data.copy_(vocab.vectors)

(iterator,) = data.Iterator.splits((data_set, ), (BATCH_SIZE,), shuffle=True, repeat = True, sort = False)

#from: https://github.com/caogang/wgan-gp/blob/master/language_helpers.py
class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)


#from: https://github.com/caogang/wgan-gp/blob/master/gan_language.py
class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(DIM, DIM, 3, padding=1),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

#adapted from: https://github.com/caogang/wgan-gp/blob/master/gan_language.py
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(NOISE_SIZE, DIM*SEQ_LEN)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1 = nn.Conv1d(DIM, DIM, 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        #shape = output.size()
        output = output.contiguous()
        return output
        #output = output.view(BATCH_SIZE*SEQ_LEN, -1)
        #output = self.softmax(output)
        #return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

#adapted from: https://github.com/caogang/wgan-gp/blob/master/gan_language.py
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1d = nn.Conv1d(DIM, DIM, 1)
        self.linear = nn.Linear(SEQ_LEN*DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, SEQ_LEN*DIM)
        output = self.linear(output)
        return output


netG = Generator().cuda()
netD = Discriminator().cuda()

optimizerD = optim.Adam(netD.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.9))

def normalize_length(tensor):
    padded = torch.cat([tensor, tensor.new(SEQ_LEN - tensor.size(0), *tensor.size()[1:]).zero_()])
    narrowed = padded.narrow(0, 0, SEQ_LEN)
    return narrowed


# expects DIM
def get_closest_word(a_word):
    minim = []
    for word in vocab.itos:
        embedding = (vocab.vectors[vocab.stoi[word]])
        a = a_word.unsqueeze(0).cpu()
        b = Variable(embedding.unsqueeze(0))
        dist = 1 - F.cosine_similarity(a, b)
        minim.append((dist.data[0], word))
    minim = sorted(minim, key=lambda v: v[0])
    closest_word = minim[0][1]
    return closest_word


#returns tensor BATCH_SIZE, SEQ_LEN, DIM
def get_next():
    for data in iterator:
        (x, x_lengths) = data.text
        input = Variable(normalize_length(x.data))
        embed_input = embed(input)
        embed_input = embed_input.permute(1, 0, 2) 
        embed_input.cuda().contiguous()
        yield embed_input
        

#expects BATCH_SIZE, SEQ_LEN, DIM
def de_embed(embedding):
    samples = []
    for sequence in embedding[:10]:
        seq = ""
        for word in sequence:
            seq += " " + get_closest_word(word)
        samples.append(seq)
    return samples


def generate_samples(netG):
    noise = torch.randn(10, NOISE_SIZE).cuda()
    noisev = Variable(noise, volatile=True) 
    fake = Variable(netG(noisev).data) 
    return de_embed(fake)


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1, 1)
    alpha = alpha.expand(real_data.size()).cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


one = torch.FloatTensor([1]).cuda()
mone = (one * -1).cuda()

#from: https://github.com/caogang/wgan-gp/blob/master/gan_language.py
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

#from: https://github.com/caogang/wgan-gp/blob/master/gan_language.py
strnow = str(datetime.datetime.now())
logger = Logger("runs/run-"+strnow, flush_secs=5)
def train():
    print('EXAMPLE REAL:', de_embed(next(get_next()))[0])
    print('EXAMPLE GENERATED:', generate_samples(netG)[0])
    for iteration in tqdm(range(ITERS)):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(CRITIC_ITERS):
            real_data_v = next(get_next())
                        
            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            
            # TODO: Waiting for the bug fix from pytorch
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BATCH_SIZE, NOISE_SIZE).cuda()
            noisev = Variable(noise, volatile=True)  # totally freeze netG
            fake = netG(noisev)
            fake = Variable(fake.data)
                        
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            # TODO: Waiting for the bug fix from pytorch
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, NOISE_SIZE).cuda()
        noisev = Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # Write logs and save samples
        
        #if disc_cost is positive, fake tweets are more likely believed that real tweets
        logger.log_value('disc_cost', D_cost.cpu().data.numpy(), iteration)
        
        #low gen_cost means discriminator is often fooled by fake tweets
        logger.log_value('gen_cost', G_cost.cpu().data.numpy(), iteration)
        
        #if wasserstein_dist is positive, real tweets are more likely to be believed than fake tweets
        logger.log_value('wasserstein_dist', Wasserstein_D.cpu().data.numpy(), iteration)

        if iteration % 100 == 99:
            samples = []
            samples.extend(generate_samples(netG))

            with open('samples_{}.txt'.format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")
                    print(s)
                    
            for i in range(4):
                lm = NgramLanguageModel(i+1, samples, tokenize=False)                
                logger.log_value('Jensen–Shannon divergence {}'.format(i+1), lm.js_with(true_char_ngram_lms[i]), iteration)
                
train()
