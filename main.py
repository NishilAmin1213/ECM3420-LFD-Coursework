from preprocessing import *

if __name__ == '__main__':
    dataset = "./data/Recalls_Data.csv"
    print("Started Program")
    create_new_copy('./data/Recalls_Data_Original.csv', dataset)
    preprocess_csv(dataset)
