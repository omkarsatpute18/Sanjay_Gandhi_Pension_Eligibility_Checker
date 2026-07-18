# main.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_FILE = "sanjay_gandhi_pension_eligibility.csv"
MODEL_FILE = "model.pkl"
ENCODER_FILE = "encoder.pkl"
CAT_COLUMNS_FILE = "cat_columns.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"

TARGET = "Eligible"
DROP_COLUMNS = ["Applicant_ID","Applicant_Name","Rejection_Reason"]

if not os.path.exists(MODEL_FILE):

    print("Training model...")

    df = pd.read_csv(DATASET_FILE)

    train_set, test_set = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET]
    )

    X_train = train_set.drop(TARGET, axis=1)
    Y_train = train_set[TARGET]

    X_test = test_set.drop(TARGET, axis=1)
    Y_test = test_set[TARGET]

    for c in DROP_COLUMNS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)

        if c in X_test.columns:
            X_test = X_test.drop(columns=c)

    cat_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
    
    
    encoder = OneHotEncoder(handle_unknown="ignore")

    train_encoded = encoder.fit_transform(X_train[cat_columns])
    train_encoded = pd.DataFrame(
        train_encoded.toarray(),
        columns=encoder.get_feature_names_out(cat_columns),
        index=X_train.index
    )

    X_train = X_train.drop(columns=cat_columns)

    for c in DROP_COLUMNS:
        if c in X_train.columns:
            X_train = X_train.drop(columns=c)

    concatenated_train = pd.concat([X_train, train_encoded], axis=1)

    test_encoded = encoder.transform(X_test[cat_columns])
    test_encoded = pd.DataFrame(
        test_encoded.toarray(),
        columns=encoder.get_feature_names_out(cat_columns),
        index=X_test.index
    )

    X_test = X_test.drop(columns=cat_columns)

    for c in DROP_COLUMNS:
        if c in X_test.columns:
            X_test = X_test.drop(columns=c)

    concatenated_test = pd.concat([X_test, test_encoded], axis=1)

    model = RandomForestClassifier(n_estimators=30,max_depth=10,min_samples_leaf=5,random_state=42)
    model.fit(concatenated_train, Y_train)

    y_pred = model.predict(concatenated_test)

    # print("Accuracy :", accuracy_score(Y_test,y_pred)*100)
    # print(classification_report(Y_test,y_pred))
    # print(confusion_matrix(Y_test,y_pred))

    joblib.dump(model,MODEL_FILE,compress=4)
    joblib.dump(encoder,ENCODER_FILE)
    joblib.dump(cat_columns,CAT_COLUMNS_FILE)
    joblib.dump(concatenated_train.columns.tolist(),FEATURE_COLUMNS_FILE)

    test_set.to_csv("input.csv",index=False)

    print("Training Complete.")
    print("input.csv generated.")

else:

    print("Loading trained model...")

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    cat_columns = joblib.load(CAT_COLUMNS_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

    input_data1 = pd.read_csv("input.csv")

    original = input_data1.copy()

    if TARGET in input_data1.columns:
        input_data = input_data1.drop(columns=[TARGET])

    encoded = encoder.transform(input_data[cat_columns])

    encoded_df = pd.DataFrame(
        encoded.toarray(),
        columns=encoder.get_feature_names_out(cat_columns),
        index=input_data.index
    )

    input_data = input_data.drop(columns=cat_columns)

    for c in DROP_COLUMNS:
        if c in input_data.columns:
            input_data = input_data.drop(columns=c)

    final_input = pd.concat([input_data,encoded_df],axis=1)

    final_input = final_input.reindex(columns=feature_columns,fill_value=0)

    pred = model.predict(final_input)
    prob = model.predict_proba(final_input)

    predictions = []
    confidences = []
    reasons_list = []

    for i, row in input_data1.iterrows():

        if pred[i] == 1:
            prediction = "Eligible for Sanjay Gandhi Pension Yojana"
            reason = "Applicant satisfies the eligibility criteria."

        else:
            prediction = "Not Eligible"

            reasons = []

            if row["Age"] < 65:
                reasons.append("Applicant is below the minimum eligible age.")

            if row["Annual_Family_Income"] > 100000:
                reasons.append("Family income exceeds the scheme limit.")

            if row["BPL_Status"] == "No":
                reasons.append("Applicant is not listed under the BPL category.")

            if row["Bank_Account"] == "No":
                reasons.append("Bank account is not available.")

            if row["Income_Certificate"] == "No":
                reasons.append("Income certificate is missing.")

            if len(reasons) == 0:
                reasons.append("Applicant does not satisfy one or more eligibility conditions.")

            reason = "; ".join(reasons)

        predictions.append(prediction)
        confidences.append(round(prob[i].max(), 4))
        reasons_list.append(reason)

    input_data1["Prediction"] = predictions
    input_data1["Confidence"] = confidences
    input_data1["Reason"] = reasons_list

    input_data1.to_csv("output.csv", index=False)