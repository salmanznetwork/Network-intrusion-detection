import numpy as np
import pandas as pd
import pickle

import warnings

warnings.filterwarnings("ignore")

model = pickle.load(open('rf_model.pkl', 'rb'))

def predict(csv_path):
    try:
        trainB = pd.read_csv(csv_path)
        # 'num_outbound_cmds' is a redundant column so remove it from both train & test datasets
        trainB.drop(['num_outbound_cmds'], axis=1, inplace=True)
        trainB.drop(['dst_host_srv_count'], axis=1, inplace=True)
        trainB.drop(['src_bytes'], axis=1, inplace=True)
        trainB.drop(['flag'], axis=1, inplace=True)
        trainB.drop(['dst_bytes'], axis=1, inplace=True)
        trainB.drop(['same_srv_rate'], axis=1, inplace=True)
        trainB.drop(['dst_host_same_srv_rate'], axis=1, inplace=True)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # extract numerical attributes and scale it to have zero mean and unit variance
        cols = trainB.select_dtypes(include=['float64', 'int64']).columns
        sc_trainB = scaler.fit_transform(trainB.select_dtypes(include=['float64', 'int64']))

        # turn the result back to a dataframe
        sc_trainBdf = pd.DataFrame(sc_trainB, columns=cols)

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()

        # extract categorical attributes from both training and test sets
        cattrainB = trainB.select_dtypes(include=['object']).copy()

        # encode the categorical attributes
        trainBcat = cattrainB.apply(encoder.fit_transform)

        trainB_df = pd.concat([sc_trainBdf, trainBcat], axis=1)

        result = model.predict(trainB_df)
        return result[0]

    except Exception as e:
        print("Something went wrong", e)
        return "Something went wrong"

if __name__ == "__main__":
    res = predict("test.csv")
    print(res)