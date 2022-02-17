import pandas as pd

df = pd.read_csv('full_data_1.csv')
df2 = pd.read_csv('data_1.csv')

print(df.head(5))
print(df.tail(5))

print(df2.head(5))
print(df2.tail(5))

# df_sub = df.head(20)
# df_sub2 = df.tail(20)
#
# df_sub.to_csv('small_data_1.csv', index=False)
# df_sub2.to_csv('small_data_2.csv', index=False)
