import os
import re
import pandas as pd

folder = "/home/benedict/Documents/comp_sci_repository/project_2/bloomberg_df/bloomberg_dataset"
folder3 = "/home/benedict/Documents/comp_sci_repository/project_2/data_scrape/text.txt"

current = folder

dfs = []

def get_structure(filepath):
    file_text = ""
    try:
        struct_dict = {'title': '', 'author': '', 'datetime': '', 'source': '', 'body': ''}
        with open(filepath, 'r') as f:
            # struct_dict["body"] = ""
            for idx, line in enumerate(f):
                #get everything on one line
                line = re.sub("\n", " ", line)
                file_text += line
                #now we need to put a dollar sign after each --
            file_text = file_text.replace('--', '&')
            file_text = file_text.replace('.html', '&')
            # filetext = filetext.replace('&', '\n')
            array = file_text.split('&')
            struct_dict["title"] = [array[1]]
            struct_dict["author"] = [array[2]]
            struct_dict["datetime"] = [array[3]]
            struct_dict["source"] = [array[4] + '.html']
            struct_dict["body"] = [array[5]]

    except:
        return
    return struct_dict


def walk_files(filepath):
    for subdir, dirs, files in sorted(os.walk(filepath)):
        # i = 0
        for file in sorted(files):
            # if i < 2:
                path = os.path.join(subdir, file)
                # print(path)
                if get_structure(path) is None:
                    continue
                else:
                    dfs.append(pd.DataFrame(get_structure(path)))
                # i += 1
            # print(i)


# call walk_files to insert all the articles into the dfs list
walk_files(current)
# merge dfs into the data_frame
data_frame = pd.concat(dfs).reset_index(drop=True)
data_frame.to_csv('data_1.csv', index=False)


