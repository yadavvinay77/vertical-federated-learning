import pandas as pd

def load_adult_dataset():
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]
    df_train = pd.read_csv("data/adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    df_train.dropna(inplace=True)
    df_train["income"] = df_train["income"].apply(lambda x: 1 if x.strip().strip('.') == ">50K" else 0)
    return df_train

def save_vertical_splits():
    df = load_adult_dataset()

    client1_features = ["age", "workclass", "education", "marital-status", "occupation", "relationship"]
    client2_features = ["race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    label = "income"

    categorical_c1 = ["workclass", "education", "marital-status", "occupation", "relationship"]
    categorical_c2 = ["race", "sex", "native-country"]

    df_client1 = df[client1_features + [label]].copy()
    df_client1 = pd.get_dummies(df_client1, columns=categorical_c1)
    df_client1.to_csv("data/client1_data.csv", index=False)

    df_client2 = df[client2_features + [label]].copy()
    df_client2 = pd.get_dummies(df_client2, columns=categorical_c2)
    df_client2.to_csv("data/client2_data.csv", index=False)

    # Save labels separately without header or index
    df[[label]].to_csv("data/labels.csv", index=False, header=False)

    print("Saved vertical splits and labels as CSVs in the data/ folder.")

if __name__ == "__main__":
    save_vertical_splits()
