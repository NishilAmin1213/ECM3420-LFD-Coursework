from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def print_stats(y_test, y_pred):
    print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:\n" + str(metrics.confusion_matrix(y_test, y_pred)))

if __name__ == '__main__':
    # specify the path for the dataset
    dataset_path = "./data/Recalls_Data.csv"
    print("Started Program")
    # copy the original dataset and then preprocess the copy
    create_new_copy('./data/Recalls_Data_Original.csv', dataset_path)

    # preprocess the dataset
    x_train, x_test, y_train, y_test = preprocess_csv(dataset_path)

    print("Training KNN Model")
    n = 5
    knn_model = KNeighborsClassifier(n)
    knn_model.fit(x_train, y_train)
    print("Testing KNN Model")
    y_pred = knn_model.predict(x_test)
    print_stats(y_test, y_pred)


    print("\n\nTraining Random Forest Classifier")
    rf_model = RandomForestClassifier(max_depth=1, random_state=0)
    rf_model.fit(x_train, y_train)
    print("Testing Random Forest Classifier")
    y_pred = rf_model.predict(x_test)
    print_stats(y_test, y_pred)

