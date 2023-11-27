import pandas as pd
from shutil import copy
from sklearn.model_selection import train_test_split


def create_new_copy(src_path, destination_path):
    print("Copying " + src_path)
    copy(src_path, destination_path)
    print("Created " + destination_path)


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

    print(filepath + " has been preprocessed")


def split_data(filepath):
    print("Splitting into Training and Testing data")
    dataset = pd.read_csv(filepath)
    training_data, testing_data = train_test_split(dataset, test_size=0.2)
    return training_data, testing_data


def get_unique(header, filepath):
    print("Getting Unique Values for " + header)
    src_path = filepath

    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    data = pd.read_csv(src_path)

    # get number of unique entries and return them
    return data[header].unique()

