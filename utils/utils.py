import numpy as np
import os
from tqdm import tqdm
import pandas as pd

from utils.distributed import is_main_gpu

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
