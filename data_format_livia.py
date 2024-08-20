import pandas as pd
import glob
import os


import yaml
from data_formatter.base import DataFormatter

import matplotlib.pyplot as plt
import statsmodels.api as sm


# Path to the directory containing your CSV files
path = './raw_data/livia_unmerged'

# Use glob to get all the CSV files in the specified directory
all_files = glob.glob(os.path.join(path, "*.csv"))

# List to store the DataFrames
df_list = []

# Read each CSV file into a DataFrame, remove rows 2-11, and append to the list
for filename in all_files:
    #print (filename)
    df = pd.read_csv(filename,low_memory=False)
    #datatypes = df.dtypes
    #print(datatypes)
    df_list.append(df)

print("out of for")

# Concatenate all DataFrames in the list
combined_df = pd.concat(df_list, ignore_index=True)

combined_df = combined_df[combined_df['Event Type'].str.lower() == 'egv']
combined_df = combined_df[combined_df['Event Subtype'].isna()]

print("after removing calibrations, injection and food")

# List of columns to keep (you can adjust this list as needed)
columns_to_keep = [
    'Index',
    'Timestamp (YYYY-MM-DDThh:mm:ss)',
    'Glucose Value (mg/dL)',
]

# Keep only the specified columns
combined_df = combined_df[columns_to_keep]



# Rename columns (adjust as needed)
column_rename = {
    'Index':'id',
    'Timestamp (YYYY-MM-DDThh:mm:ss)': 'time',
    'Glucose Value (mg/dL)': 'gl'
}


combined_df = combined_df.rename(columns=column_rename)

# Sort the combined DataFrame by timestamp
combined_df = combined_df.sort_values('time')

# Set all values in the 'Index' column to 1 where 'Timestamp' is not null
combined_df.loc[combined_df['time'].notna(), 'id'] = 1
combined_df = combined_df.drop_duplicates(subset=['time'], keep='first')
lenght_of_df=len(combined_df)
newarray=[]
i=0
j=0
for s in range(lenght_of_df):
    newarray.append(i)
    j=j+1
    if j==1000:
        i=i+1
        j=0

combined_df['id']=newarray




# Convert timestamp to datetime
combined_df['time'] = pd.to_datetime(combined_df['time'])

# Calculate time difference and keep rows with at least 1 minute difference
combined_df['time_diff'] = combined_df['time'].diff()
combined_df = combined_df[combined_df['time_diff'].isna() | (combined_df['time_diff'] >= pd.Timedelta(minutes=1))]

# Drop the temporary time_diff column
combined_df = combined_df.drop(columns=['time_diff'])

#combined_df['time'] = combined_df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
combined_df['gl']=combined_df['gl'].astype('float64')

#print("+++++++++++++++++++++++++")
#print(combined_df)
#datatypes = combined_df.dtypes
#print(datatypes)

# Write the modified dataframe to a new CSV file
combined_df.to_csv('./raw_data/livia_unmerged/livia_mini.csv', index=True)

print("CSV files have been successfully merged, modified, and saved.")

'''
dataset="livia"
with open(f'./config/{dataset}.yaml', 'r') as f:
    config = yaml.safe_load(f)
formatter = DataFormatter(config)

# set interpolation params for no interpolation
new_config = config.copy()
new_config['interpolation_params']['gap_threshold'] = 30
new_config['interpolation_params']['min_drop_length'] = 0
# set split params for no splitting
new_config['split_params']['test_percent_subjects'] = 0
new_config['split_params']['length_segment'] = 0
# set scaling params for no scaling
new_config['scaling_params']['scaler'] = 'None'

formatter = DataFormatter(new_config)

##capture

# Need: Tradeoff between interpolation and segment length
# Problem: Manually tuning is slow and potentially imprecise
# Idea: have automated function that can help determine what the gap threshold should be
# Proof of concept below

import numpy as np

def calc_percent(a, b):
    return a*100/b

gap_threshold = np.arange(5, 70, 1)
percent_valid = []
for i in gap_threshold:
    new_config['interpolation_params']['gap_threshold'] = i
    df = DataFormatter(new_config).train_data
    
    segment_lens = []
    for group, data in df.groupby('id_segment'):
        segment_lens.append(len(data))
    
    threshold = 240
    valid_ids = df.groupby('id_segment')['time'].count().loc[lambda x : x>threshold].reset_index()['id_segment']
    
    percent_valid.append((len(valid_ids)*100/len(segment_lens)))



# Plot results
plt.plot(gap_threshold, percent_valid)
plt.title("Gap Threshold affect on % Segments > 240 Length")
plt.ylabel("% Above Threshhold")
plt.xlabel("Gap Threshold (min)")


# print min, max, median, mean, std of segment lengths
df = formatter.train_data
segment_lens = []
for group, data in df.groupby('id_segment'):
    segment_lens.append(len(data))

print('Train segment lengths:')
print('\tMin: ', min(segment_lens))
print('\tMax: ', max(segment_lens))
print('\tMedian: ', np.median(segment_lens))
print('\tMean: ', np.mean(segment_lens))
print('\tStd: ', np.std(segment_lens))

# Visualize segment lengths to see approx # of valid ones (>240)
plt.title("Segment Lengths (Line at 240)")
plt.hist(segment_lens)
plt.axvline(240, color='r', linestyle='dashed', linewidth=1)

# filter to get valid indices
threshold = 240
valid_ids = df.groupby('id_segment')['time'].count().loc[lambda x : x>threshold].reset_index()['id_segment']
df_filtered = df.loc[df['id_segment'].isin(valid_ids)]

# plot each segment
num_segments = df_filtered['id_segment'].nunique()

fig, axs = plt.subplots(1, num_segments, figsize=(30, 5))
for i, (group, data) in enumerate(df_filtered.groupby('id_segment')):
    data.plot(x='time', y='gl', ax=axs[i], title='Segment {}'.format(group))


# plot acf of random samples from segments
fig, ax = plt.subplots(2, 5, figsize=(30, 5))
lags = 240
for i, (group, data) in enumerate(df_filtered.groupby('id_segment')):
    # only view top 5
    if i < 5:
        data = data['gl']
        if len(data) < lags: # TODO: Could probably do filtering in pandas which would be faster
            print('Segment {} is too short'.format(group))
            continue
        # select 10 random samples from index of data
        sample = np.random.choice(range(len(data))[:-lags], 10, replace=False)
        # plot acf / pacf of each sample
        for j in sample:
            acf, acf_ci = sm.tsa.stattools.acf(data[j:j+lags], nlags=lags, alpha=0.05)
            pacf, pacf_ci = sm.tsa.stattools.pacf(data[j:j+lags], method='ols-adjusted', alpha=0.05)
            ax[0, i].plot(acf)
            ax[1, i].plot(pacf)


with open(f'./config/{dataset}.yaml', 'w') as file:
    yaml.dump(new_config, file, default_flow_style=False)
'''