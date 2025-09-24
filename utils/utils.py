import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import torch
#from utils.distributed import is_main_gpu
import matplotlib.pyplot as plt

def filter_dates(csvfile, datestart, datestop):
    if datestart and datestop :
        csvfile['Date'] = pd.to_datetime(csvfile['Date'])
        csvfile = csvfile.loc[(csvfile['Date'] >= datestart)
                    & (csvfile['Date'] <= datestop)]
    return csvfile

def filter_lead_times(csvfile, leadtimes):
    if leadtimes :
            csvfile = csvfile[csvfile['LeadTime'].isin(leadtimes)]
    return csvfile

def batch_output_sample_files(data_dir, Shape=(4, 256, 256), conditioned=False, csv_file=[], ensemble_index=0, config=None):
    if is_main_gpu():
        parent_directory = os.path.abspath(os.path.join(data_dir, os.pardir))
        # Define the output directory
        save_folder = f'batched_samples_{str(ensemble_index)}'
        save_directory = os.path.join(parent_directory, save_folder)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")
        else:
            print(f"Directory already exists: {save_directory}")

        # List all .npy files in the directory
        npy_files = [f for f in sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[2])) if f.endswith('.npy')]
        batch_file_size = 128
        # Initialize a list to hold the batched images
        batch = []
        batch_index = 0

        if not conditioned:
            sample_local_index = 0
            batch = []
            for filename in tqdm(npy_files, desc="Processing files"):
                img = np.load(os.path.join(data_dir, filename))
                batch.append(img)
                if len(batch) == batch_file_size:
                    batch_array = np.stack(batch)  # Convert list to a numpy array of shape (128, 4, 256, 256)
                    batch_filename = f'4var_fake_sample_{batch_index}.npy'
                    np.save(os.path.join(save_directory, batch_filename), batch_array)
                    batch = []  # Reset the batch
                    batch_index += 1
            # Save any remaining images in the last batch if it’s not empty
            if batch:
                batch_array = np.stack(batch)
                batch_filename = f'_sample_batch_{batch_index}.npy'
                np.save(os.path.join(save_directory, batch_filename), batch_array)

        if conditioned:
            batch_file_size = 16
            batched_samples = np.zeros((batch_file_size,) + tuple(Shape))
            sample_local_index = 0

            # Read the dataset csv
            df = pd.read_csv(csv_file, index_col=False)
            df = filter_dates(df, config.date_start, config.date_stop)
            df = filter_lead_times(df, config.leadtimes)
            dates = df["Date"].to_list()
            leadtimes = df["LeadTime"].to_list()
            for filename in tqdm(npy_files, desc="Processing files"):
                # Load the .npy file
                img = np.load(os.path.join(data_dir, filename))
                batched_samples[sample_local_index] = img

                sample_local_index += 1
                if sample_local_index == batch_file_size:
                    date_time = dates[batch_index * batch_file_size]
                    if type(date_time) is str:
                        date = date_time.replace("T21:00:00Z", "")
                    else:
                        date = date_time.date()
                    lt = leadtimes[batch_index * batch_file_size] + 1
                    file_name = f'4var_fake_ensemble_{date}_{lt}.npy'
                    # file_name = f'4var_fake_ensemble_{date}_{batch_index}.npy'
                    # file_name = f'4var_fake_ensemble_{date}_{batch_index}_{str(ensemble_index)}.npy'
                    # file_name = f'4var_fake_ensemble_{date}_{lt}_{str(batch_index)}_{str(ensemble_index)}.npy'
                    np.save(os.path.join(save_directory, file_name), batched_samples)

                    batch_index += 1
                    sample_local_index = 0
                    batched_samples = np.zeros((batch_file_size,) + tuple(Shape))


def mirror_fill(img, mask):
    """
    Fills an img with invalid datas with valid datas, keeping a continuous physical aspect. The goal of this function is to define
    indexes pointing to invalid datas and their corresponding valid datas for mirror filling.
    Args:
        img (torch.tensor) : an image of shape (batch_size,variables,latitude,longitude)
        mask (torch.tensor) : a tensor of img's shape containing True (valid datas of img) and False (invalid datas of img)
    """
    device, dtype = img.device, img.dtype
    img_np = img[0].cpu().numpy()      
    mask_no_batch = mask[0].cpu().numpy() 
    var, H, W = img_np.shape
    filled = img_np.copy()
    #because of the AROME domain's shape, it is needed to fill vertically then horizontally 
    #indexes for vertical filling
    valid_x_v = []
    invalid_x_v=[]
    valid_y_v = []
    invalid_y_v=[]
    #indexes for horizontal filling
    valid_x_h = []
    invalid_x_h=[]
    valid_y_h = []
    invalid_y_h=[]
    # vertical filling
    for x in range(W):
        
        rows = np.where(mask_no_batch[0,:,x])[0] #for a given longitude x, find every invalid datas on each latitude 
        if rows.size == 0:
            continue
        r_min, r_max = rows.min(), rows.max() #looking for the extremes valid datas to fill symetrically with respect to theses extremes
        
        for i in range(r_min): #computing every pairs of indexes (pair for valid and invalid datas (lat,lon)) for invalid datas below the domain 
            i_ref = min(r_min + (r_min - i), H-1)
            valid_x_v.append(x)
            invalid_x_v.append(x)
            valid_y_v.append(i_ref)
            invalid_y_v.append(i)
            
        for i in range(r_max+1, H): #computing every pairs of indexes (pair for valid and invalid datas (lat,lon)) for invalid datas above the domain 
            i_ref = max(r_max - (i - r_max), 0)
            valid_x_v.append(x)
            invalid_x_v.append(x)
            valid_y_v.append(i_ref)
            invalid_y_v.append(i)
            
    # horizontal filling
    for y in range(H):
        cols = np.where(mask_no_batch[0,y,:])[0]#for a given latitude y, find every invalid datas on each longitude 
        if cols.size == 0:
            continue
        c_min, c_max = cols.min(), cols.max()

        for j in range(c_min): #computing every pairs of indexes (pair for valid and invalid datas (lat,lon)) for invalid datas to the left of the domain

            j_ref = min(c_min + (c_min - j), W-1)          
            valid_x_h.append(j_ref)
            invalid_x_h.append(j)
            valid_y_h.append(y)
            invalid_y_h.append(y)

        for j in range(c_max+1, W):#computing every pairs of indexes (pair for valid and invalid datas (lat,lon)) for invalid datas to the right of the domain
            j_ref = max(c_max - (j - c_max), 0)            
            valid_x_h.append(j_ref)
            invalid_x_h.append(j)
            valid_y_h.append(y)
            invalid_y_h.append(y)

    return torch.IntTensor(valid_x_v), torch.IntTensor(invalid_x_v), torch.IntTensor(valid_y_v), torch.IntTensor(invalid_y_v), torch.IntTensor(valid_x_h), torch.IntTensor(invalid_x_h), torch.IntTensor(valid_y_h), torch.IntTensor(invalid_y_h)


def plot_loss_during_training(losses : list, save_path : str, title : str):
    plt.figure()
    x = np.arange(len(losses))
    plt.plot(x, losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
