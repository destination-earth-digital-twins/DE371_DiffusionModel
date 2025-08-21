import pandas as pd

leadtime_list = [ 2, 5, 8, 11, 14, 17, 20, 23, 29, 35, 41, 44 ]
N_draw = 50
data = {
    'idx': list(range(len(leadtime_list)*N_draw)),
    'filename_parent': [f'_grand_sample_{lt+1}_875.npy' for lt in leadtime_list]*N_draw,
    'filename_member': [f'mb_{draw_idx}.npy' for draw_idx in range(N_draw) for _ in range(len(leadtime_list))],
    'date': ['2021-10-01']*len(leadtime_list)*N_draw,
    'leadtime': [lt+1 for lt in leadtime_list]*N_draw,
    'draw_idx': [draw_idx for draw_idx in range(N_draw) for _ in range(len(leadtime_list))]
}

df = pd.DataFrame(data)

csv_file_path = '/project/home/p200177/DE_371/datasets/dataset_Meteo_France/grandEnsemble/AROME/data.csv'

df.to_csv(csv_file_path, index=False)

print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')

