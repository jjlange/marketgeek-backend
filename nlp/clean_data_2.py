import pandas as pd


# remove short sentences
# calculate mean of the lengths of all the bodies, then subtract the standard deviation
# then filter the text that is less than the mean - standard deviation

data_frame = pd.read_csv('data/full_data_1.csv')

# length of data_frame, length of characters in the bodies and the mean and standard deviation variables
length = len(data_frame.index)
body_lengths = data_frame['body'].str.len()
mean = body_lengths.sum() / length
sd = body_lengths.std()

# delete when condition is met
data_frame = data_frame.drop(data_frame[body_lengths < (mean - sd)].index)
data_frame = data_frame.dropna(how='any', axis=0)


# cleaning certain author names
sub_df = data_frame[data_frame['author'].str.match('^\[')== True]
for i in sub_df.index:
    data_frame['author'][i] = data_frame['author'][i][19:-3]


# export -----------------------
# data_frame.to_csv('data/full_data.csv', index=False)

# print(len(data_frame.index))
# latest size of dataset is 422,101
