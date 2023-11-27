from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

if __name__ == '__main__':
    # specify the path for the dataset
    dataset_path = "./data/Recalls_Data.csv"
    print("Started Program")
    # copy the original dataset and then preprocess the copy
    create_new_copy('./data/Recalls_Data_Original.csv', dataset_path)
    preprocess_csv(dataset_path)
    # at this point, the copied dataset at the specified path is the dataset to work on

    # split data into train and test data
    x_train, x_test, y_train, y_test = split_data(dataset_path)

    print("Training Model")
    n = 5
    knn_model = KNeighborsClassifier(n)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
