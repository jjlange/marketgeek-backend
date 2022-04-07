import os
import re
import pandas as pd
import nltk
import numpy as np
from tqdm import tqdm


folder = "/home/benedict/Documents/comp_sci_repository/project_2/bloomberg_df/bloomberg_dataset"

folder_mac = "/Users/BenedictGrey/Documents/GitHub/comp_sci/project_2/bloomberg_df"

folder2 = "/home/benedict/Documents/comp_sci_repository/project_2/bloomberg_df/bloomberg_dataset/2006-10-20/inco-s-net-soars-on-higher-metal-prices-breakup-fee-update4-"

folder3 = "/home/benedict/Documents/comp_sci_repository/project_2/data_scrape/text.txt"

current = folder_mac

dfs = []


def get_structure(filepath):
    file_text = ""
    try:
        struct_dict = {'title': '', 'author': '', 'datetime': '', 'source': '', 'body': ''}
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                line = re.sub("\n", " ", line)
                file_text += line
            file_text = file_text.replace('--', '&')
            file_text = file_text.replace('.html', '&')
            file_text = file_text.lower()
            file_text = file_text.encode('ascii', 'ignore').decode()

            array = file_text.split('&')
            struct_dict["title"] = [array[1].strip()]
            author_array = [array[2].strip()][0]

            if author_array.find('b y ') != -1:
                author_array = re.sub('(?<!\s)\s{1}(?!\s)', '', author_array)
                author_array = re.sub('\s\s+', ' ', author_array)
                author_array = author_array.replace('by ', '')

            struct_dict["author"] = [author_array]
            struct_dict["datetime"] = [array[3].strip()]
            struct_dict["source"] = [array[4].strip() + '.html']
            struct_dict["body"] = [array[5].strip()]
    except:
        return
    return struct_dict


def walk_files(filepath):
    for subdir, dirs, files in sorted(os.walk(filepath)):
        i = 0
        for file in sorted(files):
            if i < 2:
                path = os.path.join(subdir, file)
                if get_structure(path) is None:
                    continue
                else:
                    dfs.append(pd.DataFrame(get_structure(path)))
                i += 1


# ------------export----------------#

# call walk_files to insert all the articles into the dfs list

walk_files(current)

# merge dfs into the data_frame
data_frame = pd.concat(dfs).reset_index(drop=True)
# print(data_frame)

# chunks = np.array_split(data_frame.index, 100) # chunks of 100 rows


# for chunck, subset in enumerate(tqdm(chunks)):
#     if chunck == 0: # first row
#         data_frame.loc[subset].to_csv('full_data_1.csv', mode='w', index=False)
#     else:
#         data_frame.loc[subset].to_csv('full_data_1.csv', header=None, mode='a', index=False)


# export data to csv
# data_frame.to_csv('full_data_1.csv', index=False)
# data_frame.to_csv('small_data_1.csv', index=False)


