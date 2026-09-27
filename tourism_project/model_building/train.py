"""
Model Training Script (with Experimentation Tracking)
--------------------------------------------------------
Loads the train/test splits produced by the data-preparation step, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV, logs every
tuned parameter combination (and the winning one) to MLflow, evaluates the
best model on the held-out test set, and saves the best model so the
GitHub Actions workflow can commit it to the repository.
"""

import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import joblib
import mlflow

MODEL_OUT_PATH = "tourism_project/deployment/best_model.joblib"

CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
NUMERIC_COLS = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


def load_splits():
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()
    y_test = pd.read_csv("ytest.csv").squeeze()
    return X_train, X_test, y_train, y_test


def build_pipeline():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        (StandardScaler(), NUMERIC_COLS),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    return make_pipeline(preprocessor, model)


def main():
    X_train, X_test, y_train, y_test = load_splits()
    pipeline = build_pipeline()

    mlflow.set_experiment("Tourism_Wellness_Package")

    with mlflow.start_run(run_name="xgb_gridsearch") as parent_run:
        grid_search = GridSearchCV(
            pipeline,
            param_grid=PARAM_GRID,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Log every parameter combination that was tried as its own nested run.
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

        best_model = grid_search.best_estimator_

        # Log the winning configuration and test-set metrics on the parent run.
        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)

        print("Best parameters:", grid_search.best_params_)
        print(f"Best CV F1 score : {grid_search.best_score_:.4f}")
        print(f"Test accuracy    : {test_accuracy:.4f}")
        print(f"Test F1 score    : {test_f1:.4f}")
        print("\nClassification report on test set:")
        print(classification_report(y_test, y_pred))

        # Save the best model so the pipeline can commit it to the repository.
        os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_OUT_PATH)
        mlflow.log_artifact(MODEL_OUT_PATH)
        print(f"\nBest model saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
