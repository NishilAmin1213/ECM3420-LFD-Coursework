import pandas as pd
from shutil import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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

    # remove any rows where 'Recall Type' is not vehicle - COMMENTED THIS OUT FOR NOW
    # data = data.drop(data[data['Recall Type'] != 'Vehicle'].index)

    # remove any rows which contains NaN or no value
    data.dropna(axis=0, inplace=True)

    # save the dataframe as a CSV and overwrite the original file
    data.to_csv(path_or_buf=filepath, index=False)

    print(filepath + " has been preprocessed")


def remove_result_column(x_dataset, header):
    print("Separating the result column - " + str(header))
    # create new dataframe y_dataset to hold the column specified by the 'header' variable
    y_dataset = pd.DataFrame(x_dataset[header])

    # remove the column name 'header' from the x_dataset
    x_dataset.drop(columns=header, inplace=True, axis=1)

    # return the two datasets
    return x_dataset, y_dataset


def encode_data(dataframe):
    print("Encoding Data...")
    le = LabelEncoder()
    dataframe['Manufacturer'] = le.fit_transform(dataframe['Manufacturer'])
    dataframe['Recall Type'] = le.fit_transform(dataframe['Recall Type'])
    dataframe['Component'] = le.fit_transform(dataframe['Component'])
    dataframe['Park Outside Advisory '] = le.fit_transform(dataframe['Park Outside Advisory '])
    dataframe['Do Not Drive Advisory'] = le.fit_transform(dataframe['Do Not Drive Advisory'])

    dataframe['Report Received Date'] = le.fit_transform(dataframe['Report Received Date'])
    dataframe['Subject'] = le.fit_transform(dataframe['Subject'])

    return dataframe



def split_data(filepath):
    print("Splitting into Training and Testing data....")
    x_dataset = pd.read_csv(filepath)
    x_dataset = encode_data(x_dataset)
    x_dataset.to_csv(path_or_buf='./data/encoded_data.csv', index=False)

    # remove the 'Do Not Drive Advisory' column from x_dataset and store it in y_dataset
    x_dataset, y_dataset = remove_result_column(x_dataset, 'Do Not Drive Advisory')

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2)
    return x_train, x_test, y_train, y_test


def get_unique(header, filepath):
    print("Getting Unique Values for " + header)
    src_path = filepath

    # use pandas read_csv function to read the 'Recalls_Data.csv' file into a dataframe
    data = pd.read_csv(src_path)

    # get number of unique entries and return them
    return data[header].unique()

