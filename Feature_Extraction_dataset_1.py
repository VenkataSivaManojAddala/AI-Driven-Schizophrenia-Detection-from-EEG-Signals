import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import os
import antropy as ant
import neurokit2 as nkt


def process_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    ch = 128
    n_col = []

    for i in range(1, 17):
        n_col.append(f'v{i}')
    for i in range(1,17):
        n_col.append(f'std{i}')
    for i in range(1, 17):
        n_col.append(f'sk{i}')
    for i in range(1, 17):
        n_col.append(f'k{i}')
    for i in range(1,17):
        n_col.append(f'HFD{i}')
    for i in range(1,17):
        n_col.append(f'KFD{i}')
    for i in range(1,17):
        n_col.append(f'SFD{i}')     
    n_col.append('target')  # Adding the target variable column

    # Initialize an empty DataFrame
    df1 = pd.DataFrame(columns=n_col)

    inr = 0

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        m, n = df.shape
        chunk_size = ch
        num_chunks = m // chunk_size
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunk_data = df.iloc[start_idx:end_idx, :]

            # Dynamically increase the size of df1 if needed
            while inr >= len(df1):
                df1 = df1.append(pd.Series(), ignore_index=True)

            for j in range(1, 17):
                var = np.var(chunk_data[str(j)], ddof=1)
                df1.loc[inr, f'v{j}'] = var

            for j in range(1, 17):
                std = np.std(chunk_data[str(j)], ddof=1)
                df1.loc[inr, f'std{j}'] = std

            for j in range(1, 17):
                skew = chunk_data[str(j)].skew()
                df1.loc[inr, f'sk{j}'] = skew

            for j in range(1, 17):
                kurt = chunk_data[str(j)].kurtosis()
                df1.loc[inr, f'k{j}'] = kurt
            
            for j in range(1, 17):
                hfd = ant.higuchi_fd(chunk_data[str(j)])
                df1.loc[inr, f'HFD{j}'] = hfd
        
            for j in range(1, 17):
                kfd = ant.katz_fd(chunk_data[str(j)])
                df1.loc[inr, f'KFD{j}'] = kfd
        
            for j in range(1, 17):
                sfd = nkt.fractal_sevcik(chunk_data[str(j)])
                df1.loc[inr, f'SFD{j}'] = sfd[0]

            inr += 1

    df1.to_csv(os.path.join(folder_path, "trail.csv"), index=False)

# Example usage:
folder_path = "norm"
process_csv_files(folder_path)