import pandas as pd
import os
import argparse
from joblib import dump, load
from datetime import datetime


from sklearn.svm import SVC
from sklearn.metrics import classification_report


def main(arg=None):
    parser = argparse.ArgumentParser(
        description="Pipeline for SVM classifier"
    )
    parser.add_argument(
        '--train', action="store_true"
    )
    parser.add_argument(
        '--test', action="store_true"
    )
    parser.add_argument(
        '--save', help='Whether to save the model', action="store_true"
    )
    parser.add_argument(
        '--model_dir', help="Directory for saving the model for training mode",
        default="./model/SVM"
    )
    parser.add_argument(
        '--model_path', help="Path to model.joblib file for testing mode or predicting mode."
    )
    parser.add_argument(
        '--dataset_dir', required=True, help='Directory of dataset. Should contain train.csv and test.csv'
    )
    parser.add_argument(
        '--output_dir', help="Directory to output the prediction file and report.", default=None
    )
    parser.add_argument(
        '--exp_name', help="Experiment name for logging.", default=""
    )
    parser.add_argument(
        '--pred', action='store_true', help='Enable prediction for the unsupervised.csv under dataset_dir.'
    )

    parser = parser.parse_args()

    dataset_dir = parser.dataset_dir
    train_mode = parser.train
    test_mode = parser.test
    pred_mode = parser.pred

    if pred_mode:
        print("Predicting...")
        model_path = parser.model_path
        classifier = load(model_path)
        time_stamp = os.path.split(model_path)[-1].split(".")[
            0].split("_")[-1]

        X_unsupervised = pd.read_csv(os.path.join(dataset_dir, "unsupervised.csv")).values
        y_pred = classifier.predict(X_unsupervised)

        output_dir = parser.output_dir
        if output_dir == None:
            output_dir = os.path.dirname(model_path)

        pred_path = "prediction_" + time_stamp + ".csv"
        pred_path = os.path.join(output_dir, pred_path)

        y_pred_df = pd.DataFrame(y_pred, columns=['pred'])
        y_pred_df.to_csv(pred_path, index=False)
        print("Prediction for unsupervised dataset saved as:", pred_path)
        
        return
    
    
    
    if not train_mode and not test_mode:
        train_mode = True
        test_mode = True
    
    model_dir = parser.model_dir
    exp_name = parser.exp_name
    model_dir = os.path.join(model_dir, exp_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    time_stamp = datetime.now().strftime("%b%d-%H-%M-%S")
    
    train_data = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    y_train = y_train.astype(int)
    
    if train_mode:
        print("Training...")
        classifier = SVC(kernel="rbf", gamma="auto")
        classifier.fit(X_train, y_train)

        model_path = "SVM_"+time_stamp+".joblib"
        model_path = os.path.join(model_dir, model_path)
        dump(classifier, model_path)
        print("Model saved as: ", model_path)

    if test_mode:
        print("Testing...")
        if not train_mode:
            model_path = parser.model_path
            classifier = load(model_path)
            time_stamp = os.path.split(model_path)[-1].split(".")[
                0].split("_")[-1]

        test_data = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
        X_test = test_data.iloc[:, 1:].values
        y_test = test_data.iloc[:, 0].values
        y_pred = classifier.predict(X_test)

        report = classification_report(y_test, y_pred)
        print(report)

        output_dir = parser.output_dir
        if output_dir == None:
            output_dir = os.path.dirname(model_path)

        report_path = "report_" + time_stamp + ".txt"
        report_path = os.path.join(output_dir, report_path)
        with open(report_path, "w") as out:
            out.write(report)
        print("Report saved as:", report_path)

        pred_path = "test_prediction_" + time_stamp + ".csv"
        pred_path = os.path.join(output_dir, pred_path)

        y_pred_df = pd.DataFrame(y_pred, columns=['pred'])
        y_pred_df.to_csv(pred_path, index=False)
        print("Prediction for test dataset saved as:", pred_path)
        

        
        
    


if __name__ == "__main__":
    main()
