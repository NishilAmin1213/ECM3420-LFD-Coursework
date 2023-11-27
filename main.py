from preprocessing import *

if __name__ == '__main__':
    # specify the path for the dataset
    dataset_path = "./data/Recalls_Data.csv"
    print("Started Program")
    # copy the original dataset and then preprocess the copy
    create_new_copy('./data/Recalls_Data_Original.csv', dataset_path)
    preprocess_csv(dataset_path)
    # at this point, the copied dataset at the specified path is the dataset to work on

    # split data into train and test data
    training_data, testing_data = split_data(dataset_path)
