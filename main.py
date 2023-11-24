import pandas as pd
from shutil import copy


def create_new_copy(src_path, destination_path):
    print("Copying Recalls_Data_Original.csv")
    copy(src_path, destination_path)
    print("Created Recalls_Data.csv")


def preprocess_csv(filepath):
    print("Preprocessing " + filepath)
    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    data = pd.read_csv(filepath)

    # define columns to be removed from the dataframe
    columns_to_remove = ['NHTSA ID', 'Recall Link', 'Mfr Campaign Number', 'Recall Description', 'Consequence Summary',
                         'Corrective Action', 'Completion Rate % (Blank - Not Reported)']

    # remove columns that are no longer needed
    # labels is the array of headers for the colums to remove
    # inplace works on the dataframe itself as opposed to returning a copy
    # axis specifies to remove columns, not rows
    data.drop(columns=columns_to_remove, inplace=True, axis=1)

    # remove any rows where 'Recall Type' is not vehicle
    data = data.drop(data[data['Recall Type'] != 'Vehicle'].index)

    # save the dataframe as a CSV and overwrite the original file
    data.to_csv(path_or_buf=filepath, index=False)


def get_unique(header, filepath):
    print("Getting Unique Values for " + header)
    src_path = filepath

    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    data = pd.read_csv(src_path)

    # get number of unique entries
    unique = data[header].unique()

    # return unique entries
    return unique


if __name__ == '__main__':
    filepath = "./data/Recalls_Data.csv"
    print("Started Program")
    create_new_copy('./data/Recalls_Data_Original.csv', filepath)
    preprocess_csv(filepath)


''' 

all_columns = ['Report Received Date', 'NHTSA ID', 'Recall Link', 'Manufacturer', 'Subject', 'Component', 'Mfr Campaign Number',
            'Recall Type', 'Potentially Affected', 'Recall Description', 'Consequence Summary', 'Corrective Action',
            'Park Outside Advisory ', 'Do Not Drive Advisory', 'Completion Rate % (Blank - Not Reported)']

[
'Report Received Date', 
'NHTSA ID', 
'Recall Link', 
'Manufacturer', 
'Subject', 
'Component', 
'Mfr Campaign Number',
'Recall Type', 
'Potentially Affected', 
'Recall Description', 
'Consequence Summary', 
'Corrective Action',
'Park Outside Advisory', 
'Do Not Drive Advisory', 
'Completion Rate % (Blank - Not Reported)'
]

'''
