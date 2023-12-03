import pandas as pd
from shutil import copy
from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def create_new_copy(src_path, destination_path):
    copy(src_path, destination_path)
    print("Created " + destination_path)


def encode_data(dataframe):
    print("Encoding Data ....")
    le = LabelEncoder()
    headers = ['Manufacturer', 'Recall Type', 'Component', 'Park Outside Advisory ', 'Do Not Drive Advisory', 'Report Received Date', 'Subject']
    for column in headers:
        dataframe[column] = le.fit_transform(dataframe[column])

    return dataframe


def split_result_column(x_dataset, header):
    print("Separating the results column ....")
    # create new dataframe y_dataset to hold the column specified by the 'header' variable
    y_dataset = pd.DataFrame(x_dataset[header])

    # remove the column name 'header' from the x_dataset
    x_dataset.drop(columns=header, inplace=True, axis=1)

    # return the two datasets
    return x_dataset, y_dataset


def resample_data(x_dataset, y_dataset, option):
    print("Rescaling Data ....")
    samplers = []

    if option == 'nm':
        samplers.append(NearMiss(sampling_strategy=1))
    elif option == 'cc':
        samplers.append(ClusterCentroids())
    elif option == 'rus(not minority)':
        samplers.append(RandomUnderSampler(sampling_strategy='not minority'))
    elif option == 'ros(0.1)rus(0.5)':
        samplers.append(RandomOverSampler(sampling_strategy=0.1))
        samplers.append(RandomUnderSampler(sampling_strategy=0.5))
    elif option == 'smoteenn':
        samplers.append(SMOTEENN())
    else:
        pass

    for sampler in samplers:
        x_dataset, y_dataset = sampler.fit_resample(x_dataset, y_dataset)

    return x_dataset, y_dataset


def clean_dataset(dataset):
    print("Cleaning Dataset ....")
    # define columns to be removed from the dataframe
    columns_to_remove = ['NHTSA ID', 'Recall Link', 'Mfr Campaign Number', 'Recall Description', 'Consequence Summary',
                         'Corrective Action', 'Completion Rate % (Blank - Not Reported)']

    # remove columns that are no longer needed
    # labels is the array of headers for the colums to remove
    # inplace works on the dataframe itself as opposed to returning a copy
    # axis specifies to remove columns, not rows
    dataset.drop(columns=columns_to_remove, inplace=True, axis=1)

    # remove any rows where 'Recall Type' is not vehicle - COMMENTED THIS OUT FOR NOW - THIS IS NOT WORKING
    dataset.drop(dataset[dataset['Recall Type'] != 'Vehicle'].index, axis=0, inplace=True)

    # remove any rows which contains NaN or no value
    dataset.dropna(axis=0, inplace=True)


def split_rescale_data(x_dataset, option):
    print("Splitting into Training and Testing data ....")

    # remove the 'Do Not Drive Advisory' column from x_dataset and store it in y_dataset
    x_dataset, y_dataset = split_result_column(x_dataset, 'Do Not Drive Advisory')

    # rescale dataset to remove the imbalance between 1 and 0 values in 'Do Not Drive'
    x_dataset, y_dataset = resample_data(x_dataset, y_dataset, option)

    return x_dataset, y_dataset


def preprocess_csv(filepath, option):
    print("Preprocessing " + filepath)

    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    dataset = pd.read_csv(filepath)

    # clean data by removing unwanted rows and columns
    clean_dataset(dataset)

    # save the dataframe as a CSV and overwrite the original file (this produces a readable csv)
    dataset.to_csv(path_or_buf=filepath, index=False)

    # encode dataframe and save the encoded values to a csv file (this produces a csv of encoded data)
    dataset = encode_data(dataset)
    dataset.to_csv(path_or_buf='./data/encoded_data.csv', index=False)

    # split data into x and y datasets (column to predict separately)
    x_dataset, y_dataset = split_rescale_data(dataset, option)

    # split the datasets into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2)

    # return the split and preprocessed data
    return x_train, x_test, y_train, y_test


def get_unique(header, filepath):
    print("Getting Unique Values for " + header)
    src_path = filepath

    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    data = pd.read_csv(src_path)

    # get number of unique entries and return them
    return data[header].unique()

