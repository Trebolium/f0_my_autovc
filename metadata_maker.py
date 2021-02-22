"""
Generate speaker embeddings and metadata for training (metadata consists of tuples(speaker directory path, speaker embeddings, spectrogram file paths )
"""
import os, pdb
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

# C is the speaker encoder. The config values match with the paper
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
# Speaker encoder checkpoint things. Load up the pretrained checkpoint info
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, fileList = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# speakers contains list of utterance paths/embeddings
speakers = []
# each speaker is a folder path to that speaker
for melspec_path in fileList:
    print('Processing speaker: %s' % melspec_path)
    # utterances is a list of speaker paths, speaker embeddings, filepath1 for that speaker, filepath2 for speaker, etc.
    utterances = []
    utterances.append(os.path.join(melspec_path))
    # fileList is list of paths for this speaker
    # make speaker embedding
    embs = []
    # tmp is just a single numpy melspec
    tmp = np.load(os.path.join(dirName, melspec_path))
    # left = random window offset
    melsp = torch.from_numpy(tmp[np.newaxis, 0:128, :]).cuda()
    # put mels through the speaker encoder to get their embeddings
    emb = C(melsp)
    embs.append(emb.detach().squeeze().cpu().numpy())     
    # Get mean of embs across rows, and add this to utterances list
    utterances.append(np.mean(embs, axis=0))
    utterances.append(tmp)
    speakers.append(utterances)

#save speaker utterance enbeddings and path info
with open(os.path.join('Brendan_train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

