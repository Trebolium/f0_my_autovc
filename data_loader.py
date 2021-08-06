from torch.utils import data
import torch
import numpy as np
import os, pdb, pickle, random
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    # this object will contain both melspecs and speaker embeddings taken from the train.pkl
    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.spmel_dir = config.data_dir
        self.pitch_dir = './pitch'
        self.len_crop = config.len_crop
        self.step = 10
        self.file_name = config.file_name
        self.one_hot = config.one_hot

        # metaname = os.path.join(self.spmel_dir, "all_meta_data.pkl")
        meta_all_data = pickle.load(open('./all_meta_data.pkl', "rb"))
        # split into training data
        num_training_speakers=config.train_size
        random.seed(1)
        training_indices =  random.sample(range(0, len(meta_all_data)), num_training_speakers)
        training_set = []

        meta_training_speaker_all_uttrs = []
        # make list of training speakers
        for idx in training_indices:
            meta_training_speaker_all_uttrs.append(meta_all_data[idx])
        # get training files
        for speaker_info in meta_training_speaker_all_uttrs:
            speaker_id_emb = speaker_info[:2]
            speaker_uttrs = speaker_info[2:]
            num_files = len(speaker_uttrs) # first 2 entries are speaker ID and speaker_emb)
            training_file_num = round(num_files*0.9)
            training_file_indices = random.sample(range(0, num_files), training_file_num)

            training_file_names = []
            for index in training_file_indices:
                fileName = speaker_uttrs[index]
                training_file_names.append(fileName)
            training_set.append(speaker_id_emb+training_file_names)
            # training_file_names_array = np.asarray(training_file_names)
            # training_file_indices_array = np.asarray(training_file_indices)
            # test_file_indices = np.setdiff1d(np.arange(num_files_in_subdir), training_file_indices_array)
        meta = training_set
        # pdb.set_trace()
        with open('./model_data/' +self.file_name +'/training_meta_data.pkl', 'wb') as train_pack:
            pickle.dump(training_set, train_pack)
        # pdb.set_trace()

        training_info = pickle.load(open('./model_data/' +self.file_name +'/training_meta_data.pkl', 'rb'))
        num_speakers_seq = np.arange(len(training_info))
        self.one_hot_array = np.eye(len(training_info))[num_speakers_seq]
        self.spkr_id_list = [spkr[0] for spkr in training_info]

        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        # uses a different process thread for every self.steps of the meta content
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
    # this function is called within the class init (after self.data_loader its the arguments) 
    def load_data(self, submeta, dataset, idx_offset):  
        # go through each entry in meta
        which_pitch_info=0
        for k, sbmt in enumerate(submeta):    
            num_uttr_files = len(sbmt)-2
            # metadata consists of id,emb,uttr filenames (but we will add pitch info here, so the uttrs info will have to be doubled (minus the offset of id and emb)
            uttrs = (len(sbmt)+num_uttr_files)*[None]
            # go through each element in a single meta entry
            for j, tmp in enumerate(sbmt):
                #print(tmp)
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrogram
                    uttrs[j] = np.load(os.path.join(self.spmel_dir, tmp))
                    # load pitch info to the pitch section of the uttrs list (ie somewhere in the last num_uttrs_files indices)
                    pitch_path = os.path.join(self.pitch_dir, tmp[:-4] +'.pkl')
                    #print(pitch_path)
                    uttrs[j+num_uttr_files] = pickle.load(open(pitch_path, 'rb'))[which_pitch_info]
            dataset[idx_offset+k] = uttrs
         
    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file""" 
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
        spkr_data = dataset[index]
        num_uttr_files = int((len(spkr_data)-2)/2)
        emb_org = spkr_data[1]
        speaker_name = spkr_data[0]
        # pick random uttr with random crop
        a = np.random.randint(2, num_uttr_files+2)
        uttr_info = spkr_data[a]
       # if (a+num_uttr_files) > len(spkr_data):
       #     pdb.set_trace()
        pitch_info = spkr_data[a+num_uttr_files]

        if uttr_info.shape[0] < self.len_crop:
            len_pad = self.len_crop - uttr_info.shape[0]
            uttr = np.pad(uttr_info, ((0,len_pad),(0,0)), 'constant')
            pitch = np.pad(pitch_info, ((0,len_pad),(0,0)), 'constant')
        elif uttr_info.shape[0] > self.len_crop:
            left = np.random.randint(uttr_info.shape[0]-self.len_crop)
            uttr = uttr_info[left:left+self.len_crop, :]
            pitch = pitch_info[left:left+self.len_crop, :]
        else:
            uttr = uttr_info
            pitch = pitch_info    
        
#        if pitch.shape[0] != 128:
#            pdb.set_trace()
        
        # find out where speaker is in the order of the training list for one-hot
        for i, spkr_id in enumerate(self.spkr_id_list):
            if speaker_name == spkr_id:
                spkr_label = i
                break
        one_hot_spkr_label = self.one_hot_array[spkr_label]
        if self.one_hot==False:
            #print(uttr.shape, emb_org.shape, pitch.shape)
            return uttr, emb_org, speaker_name, pitch
        else:
            #print(uttr.shape, one_hot_spkr_label.shape, pitch.shape)
            return uttr, one_hot_spkr_label, speaker_name, pitch

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(config, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(config)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






