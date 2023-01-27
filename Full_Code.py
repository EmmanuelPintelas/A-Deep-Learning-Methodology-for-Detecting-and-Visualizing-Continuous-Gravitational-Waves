# Use timm pretrained image model
#! pip3 install timm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
import timm
import torch
import torch.nn as nn
import sklearn

from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

import cv2
import os

device = torch.device('cuda')
criterion = nn.BCEWithLogitsLoss()

# Train metadata
di = '/kaggle/input/train-labels'
df = pd.read_csv(di + '/train_labels.csv')
df = df[df.target >= 0]  # Remove 3 unknowns (target = -1)




# ----------------------- Datasets Functions ---------------------------

                
class Dataset_4ch(torch.utils.data.Dataset):

    def __init__(self, df, time_size):
        self.df = df
        self.time_size = time_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        with h5py.File('/kaggle/input/g2net-detecting-continuous-gravitational-waves/train/'+file_id+'.hdf5', 'r') as f:
            g = f[file_id]
            H = (g['H1']['SFTs'][:]) * 1e22
            L = (g['L1']['SFTs'][:]) * 1e22
        
        img_4ch = np.empty((4, 360, self.time_size), dtype=np.float32)
        
        img_4ch[0] = cv2.resize(H.real, (self.time_size, 360))
        img_4ch[1] = cv2.resize(H.imag, (self.time_size, 360))
        img_4ch[2] = cv2.resize(L.real, (self.time_size, 360))
        img_4ch[3] = cv2.resize(L.imag, (self.time_size, 360))
                       
        return np.array(img_4ch, dtype=np.float16), file_id


class Dataset_Power_Spect_Time_Window_Mean(torch.utils.data.Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        with h5py.File('/kaggle/input/g2net-detecting-continuous-gravitational-waves/train/'+file_id+'.hdf5', 'r') as f:
            g = f[file_id]
            H = (g['H1']['SFTs'][:]) * 1e22
            L = (g['L1']['SFTs'][:]) * 1e22


            img_720 = np.empty((2, 360, 720), dtype=np.float32)
            img_360 = np.empty((2, 360, 360), dtype=np.float32)
            img_180 = np.empty((2, 360, 180), dtype=np.float32)

            Hr = cv2.resize(H.real, (4320, 360))
            Him = cv2.resize(H.imag, (4320, 360))
            Lr = cv2.resize(L.real, (4320, 360))
            Lim = cv2.resize(L.imag, (4320, 360))

            Hamp, Lamp = np.sqrt(Hr**2 + Him**2), np.sqrt(Lr**2 + Lim**2) # Power Spectrum
            Hamp, Lamp = Hamp/np.mean(Hamp), Lamp/np.mean(Lamp)

            # Time Window Mean
            H_720 = np.copy(np.mean(Hamp.reshape(360, 720, 6), axis=2))
            H_360 = np.copy(np.mean(Hamp.reshape(360, 360, 12), axis=2))
            H_180 = np.copy(np.mean(Hamp.reshape(360, 180, 24), axis=2))

            L_720 = np.copy(np.mean(Lamp.reshape(360, 720, 6), axis=2))
            L_360 = np.copy(np.mean(Lamp.reshape(360, 360, 12), axis=2))
            L_180 = np.copy(np.mean(Lamp.reshape(360, 180, 24), axis=2))

            img_720[0], img_720[1] = H_720, L_720
            img_360[0], img_360[1] = H_360, L_360
            img_180[0], img_180[1] = H_180, L_180

        return np.array(img_720, dtype=np.float16), np.array(img_360, dtype=np.float16), np.array(img_180, dtype=np.float16), file_id   


# dataset_4ch = Dataset_4ch(df, 720)
# try:
#     os.mkdir('4ch_720')
# except:
#     print('folder already build')
# for i in range(600):
#     im, id = dataset_4ch[i]
#     np.save('4ch_720/'+id,im)
#     #break
    
# dataset_4ch = Dataset_4ch(df, 360)
# try:
#     os.mkdir('4ch_360')
# except:
#     print('folder already build')
# for i in range(600):
#     im, id = dataset_4ch[i]
#     np.save('4ch_360/'+id,im)
#     #break
    
# dataset_4ch = Dataset_4ch(df, 180)
# try:
#     os.mkdir('4ch_180')
# except:
#     print('folder already build')
# for i in range(600):
#     im, id = dataset_4ch[i]
#     np.save('4ch_180/'+id,im)
#     #break

# # ----------------------------------------------------------------------------------------------------------------

# dataset_power = Dataset_Power_Spect_Time_Window_Mean(df)
# try:
#     os.mkdir('P_720')
#     os.mkdir('P_360')
#     os.mkdir('P_180')
# except:
#     print('folders already build')

# for i in range(600):
#     im_720, im_360, im_180, id = dataset_power[i]
#     np.save('P_720/'+id,im_720)
#     np.save('P_360/'+id,im_360)
#     np.save('P_180/'+id,im_180)
#     #break



# #----------------------------------------------------------------------------------------------------------------












import torchaudio

# SpecAugmen 
transforms_time_mask = nn.Sequential(
                torchaudio.transforms.TimeMasking(time_mask_param=10),
            )
transforms_freq_mask = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            )
## ---- setting of audio data augmentation (spec_augm) ------
augm_rate = 0.80
flip_rate = 0.55 # probability of applying the horizontal flip and vertical flip 
fre_shift_rate = 0.8 # probability of applying the vertical shift
mask_rate = 0.7 # probability of applying the masking
time_mask_num = 1 # number of time masking
freq_mask_num = 2 # number of frequency masking
# -------------------------------------------------

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, image_path, spec_augm):
        self.df = df
        self.image_path = image_path
        self.spec_augm = spec_augm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        img = np.load(self.image_path+'/'+file_id+'.npy')
        
        img = img-np.min(img)
        img = img/np.max(img)

        img = np.array(img, dtype=np.float32)
        if self.spec_augm == True:
            if np.random.rand() <= augm_rate:
                if np.random.rand() <= flip_rate: # horizontal flip
                    img = np.flip(img, axis=1).copy()
                if np.random.rand() <= flip_rate: # vertical flip
                    img = np.flip(img, axis=2).copy()
                if np.random.rand() <= fre_shift_rate: # vertical shift
                    img = np.roll(img, np.random.randint(low=0, high=img.shape[1]), axis=1)

                if np.random.rand() <= mask_rate:

                    img = torch.from_numpy(img)
                    for _ in range(time_mask_num): # tima masking
                        img = transforms_time_mask(img)
                    for _ in range(freq_mask_num): # frequency masking
                        img = transforms_freq_mask(img)
                else:
                    img = torch.from_numpy(img)
            else:
                img = torch.from_numpy(img)
        else:
            img = torch.from_numpy(img)
        
        
        return np.array(img, dtype=np.float32), y







# # ---------- Some Vizualizations ------------------------------------------------------

# # load as it is in init form (creation of a 4 ch image constituted of: 
# #                                   H(interferometer) Real part, H(interferometer) Imaginery part, 
# #                                   L(interferometer) Real part, L(interferometer) Imaginery part 
# #                                   of spectograph of DFT Amplitude)
dataset_INIT = Dataset(df, '/kaggle/input/gw-datasets/4ch_180', spec_augm=False)
img, y = dataset_INIT[10]
print(y)
plt.figure(figsize=(9, 9))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(np.mean(img,axis=0))  # zooming in for dataset[10]
plt.colorbar()
plt.show()

# # load as proposed Power_Time_Window_Mean Spectrum
dataset_PSTM = Dataset(df, '/kaggle/input/gw-datasets/P_180', spec_augm=False)
img, y = dataset_PSTM[10]
print(y)
plt.figure(figsize=(9, 9))
plt.title('Spectrogram')
plt.xlabel('time')
plt.ylabel('frequency')
plt.imshow(np.mean(img,axis=0))  # zooming in for dataset[10]
plt.colorbar()
plt.show()



class Model(nn.Module):
    def __init__(self, name, *, in_chans = 2, pretrained=True):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        # Use timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)

        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x



def evaluate(model, loader_val, *, compute_score=True, pbar=None):
    global val, best_val
    """
    Predict and compute loss and score
    """
    tb = time.time()
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    n_sum = 0
    y_all = []
    y_pred_all = []

    if pbar is not None:
        pbar = tqdm(desc='Predict', nrows=78, total=pbar)

    for img, y in loader_val:
        n = y.size(0)
        img = img.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(img.to(device))
        loss = criterion(y_pred.view(-1), y)

        n_sum += n
        loss_sum += n * loss.item()

        y_all.append(y.cpu().detach().numpy())
        y_pred_all.append(y_pred.sigmoid().squeeze().cpu().detach().numpy())

        if pbar is not None:
            pbar.update(len(img))
        
        del loss, y_pred, img, y

    loss_val = loss_sum / n_sum

    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)
    
    if compute_score:
        ACC = sklearn.metrics.accuracy_score(y, (y_pred+0.5).astype(int))
        F1 = sklearn.metrics.f1_score(y, (y_pred+0.5).astype(int))
        PRE = sklearn.metrics.precision_score(y, (y_pred+0.5).astype(int))
        CM = sklearn.metrics.confusion_matrix(y, (y_pred+0.5).astype(int))
        SEN = CM[0,0]/(CM[0,0]+CM[0,1])
        SPE = CM[1,1]/(CM[1,0]+CM[1,1])
        AUC = roc_auc_score(y, y_pred)
    else: None
    
#     if AUC>best_val:
#         print('ACC',ACC)
#         print('F1',F1)
#         print('PRE',PRE)
#         print('SEN',SEN) 
#         print('SPE',SPE)
#         print('AUC',AUC)

    ret = {'loss': loss_val,
           'score': AUC,
           'ACC':ACC,'F1':F1,'PRE':PRE,'SEN':SEN, 'SPE':SPE, 'AUC':AUC,
           'y': y,
           'y_pred': y_pred,
           'time': time.time() - tb}
    
    model.train(was_training)  # back to train from eval if necessary

    return ret


#timm.list_models()









for model_name in ['inception_v4', 'resnest14d', 'densenet201', 'efficientnet_b0', 'tf_efficientnet_b8']:#
    nfold = 5
    kfold = KFold(n_splits=nfold, random_state=42, shuffle=True)
    #'tf_efficientnet_b7_ns',
     #'tf_efficientnet_b8'
    epochs = 15
    batch_size = 8
    num_workers = 2
    weight_decay = 1e-6
    max_grad_norm = 1000

    lr_max = 4e-4
    lr_min=1e-6
    epochs_warmup = 1.0

    Best_Scores = []
    for ifold, (idx_train, idx_test) in enumerate(kfold.split(df)):
        print('Fold %d/%d' % (ifold, nfold))
        torch.manual_seed(42 + ifold + 1)

        # Train - val split
        dataset_train = Dataset(df.iloc[idx_train], '/kaggle/input/gw-datasets/P_180', spec_augm = True)
        dataset_val =Dataset(df.iloc[idx_test], '/kaggle/input/gw-datasets/P_180', spec_augm = False)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        # Model and optimizer
        model = Model(model_name, pretrained=True, in_chans = 2)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

        # Learning-rate schedule
        nbatch = len(loader_train)
        warmup = epochs_warmup * nbatch  # number of warmup steps
        nsteps = epochs * nbatch        # number of total steps

        scheduler = CosineLRScheduler(optimizer,
                      warmup_t=warmup, warmup_lr_init=0.0, warmup_prefix=True, # 1 epoch of warmup
                      t_initial=(nsteps - warmup), lr_min=lr_min)                # 3 epochs of cosine

        time_val = 0.0
        lrs = []

        tb = time.time()
        print('Epoch   loss          score   lr')
        best_val = 0
        for iepoch in range(epochs):
            loss_sum = 0.0
            n_sum = 0

            # Train
            for ibatch, (img, y) in enumerate(tqdm(loader_train)):
                n = y.size(0)
                img = img.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_pred = model(img)
                loss = criterion(y_pred.view(-1), y)

                loss_train = loss.item()
                loss_sum += n * loss_train
                n_sum += n

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
                optimizer.step()

                scheduler.step(iepoch * nbatch + ibatch + 1)
                lrs.append(optimizer.param_groups[0]['lr'])            

            # Evaluate
            val = evaluate(model, loader_val)
            print(val['score'])
            if val['score']>best_val:
                best_val = val['score']

                best_scores = [val['ACC'],val['F1'],val['PRE'],val['SEN'],val['SPE'],val['AUC']]

                time_val += val['time']
                loss_train = loss_sum / n_sum
                lr_now = optimizer.param_groups[0]['lr']
                dt = (time.time() - tb) / 60
            print('Epoch %d %.4f %.4f %.4f  %.2e  %.2f min' %(iepoch + 1, loss_train, val['loss'], val['score'], lr_now, dt))

        dt = time.time() - tb
        print('Training done %.2f min total, %.2f min val' % (dt / 60, time_val / 60))

        # Save model
        ofilename = 'model%d.pytorch' % ifold
        torch.save(model.state_dict(), ofilename)
        print(ofilename, 'written')

        plt.title('LR Schedule: Cosine with linear warmup')
        plt.xlabel('steps')
        plt.ylabel('learning rate')
        plt.plot(lrs)
        plt.show()

        #break  # 1 fold only

        Best_Scores.append(best_scores)

    Best_Scores = np.array(Best_Scores)
    Aver_Scores = np.mean(Best_Scores,axis=0)
    print('Scores for ', model_name,':')
    print('ACC: ',np.round(Aver_Scores[0],3),
          'F1: ',np.round(Aver_Scores[1],3),
          'PRE: ',np.round(Aver_Scores[2],3),
          'SEN: ',np.round(Aver_Scores[3],3),
          'SPE: ',np.round(Aver_Scores[4],3),
          'AUC: ',np.round(Aver_Scores[5],3))
