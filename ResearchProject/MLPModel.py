from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
import os

base_dir = os.getcwd()
current_dir = os.path.join(base_dir)
model_path = os.path.join(current_dir, 'model\\trained_model.pkl')
input_data_path = os.path.join(current_dir, 'data\\datapointwithMatchMarks.csv')
new_data_path = os.path.join(current_dir, 'data\\new.csv')


def tune_model_parameters(x_train, x_validate, y_train, y_validate):
    hidden_layer_configuration = [(4),(5),(6)]
    learning_rates = ["constant", "adaptive"]
    algorithms =["adam", "sgd"]
    no_iterations =[4000]
    is_early_stopping = [True, False]
    rmse_list = []
    params_list = []
    parameters = []
    for hidden_layer_sizes in hidden_layer_configuration:
        for learning_rate in learning_rates:
            for solver in algorithms:
                for max_iter in no_iterations:
                    for early_stopping in is_early_stopping:
                        parameters.append((hidden_layer_sizes, learning_rate, solver, max_iter, early_stopping))


    for param in parameters:
        try:
            mlp_model = MLPRegressor(hidden_layer_sizes=param[0], learning_rate=param[1], solver=param[2],
                                      max_iter=param[3], early_stopping=param[4], random_state=9, shuffle=False)
            mlp_model.fit(x_train, y_train)
            preds = mlp_model.predict(x_validate)
            model_rmse = np.sqrt(mean_squared_error(y_validate, preds))
            rmse_list.append(model_rmse)
            params_list.append(param)
        except Exception as e:
            print("An exception occurred:", str(e))
            continue

    return pd.DataFrame({'parameter_set': params_list, 'rmse': rmse_list}), len(parameters)

def fit_neural_network(x_train, x_validate, y_train, y_validate, model_path):
    try:
        params_set, params_len = tune_model_parameters(x_train, x_validate, y_train, y_validate)
        params_set = params_set.sort_values(by='rmse', ascending=True)
        print('best_params: ')
        print(params_set)
        params_set = params_set.reset_index(drop=True)
        best_params = params_set['parameter_set'][0]

        params = {"hidden_layer_sizes": best_params[0],
                    "learning_rate": best_params[1],
                    "solver": best_params[2],
                    "max_iter": best_params[3],
                    "early_stopping": best_params[4],
                    "shuffle": False,
                    "random_state": 9}

        train_set = pd.concat([y_train, x_train], axis=1)
        validation_set = pd.concat([y_validate, x_validate], axis=1)
        full_dataset = train_set._append(validation_set)
        x = full_dataset[list(full_dataset)[1:]]
        y = full_dataset[list(full_dataset)[:1]]

        model = MLPRegressor(**params).fit(x,y)
        joblib.dump(model, model_path)
        return model
    except ValueError:
        raise ValueError('Neural network model is not fitted successfully')

def predict(model, predictors):
    prediction = pd.Series(model.predict(predictors))
    prediction[prediction < 0] = 0
    return prediction

def load_model(model_path):
    return joblib.load(model_path)

if not os.path.exists(model_path):
    dataset = pd.read_csv(input_data_path)

    train, test_set = train_test_split(dataset, test_size=0.1, shuffle=True)

    test_x = test_set.iloc[:,1:63]
    test_y = test_set.iloc[:,63]

    train_set, validation_set = train_test_split(train, test_size=0.2, shuffle=True)

    train_x = train_set.iloc[:,1:63]
    train_y = train_set.iloc[:,63]
    validation_x = validation_set.iloc[:,1:63]
    validation_y = validation_set.iloc[:,63]

    model = fit_neural_network(train_x, validation_x, train_y, validation_y, model_path)
    predicted = predict(model, test_x)
    pd.DataFrame({'Actual': test_y.to_list(), 'Predicted': predicted.to_list()}).to_csv('predicted.csv')
else:
    model = load_model(model_path)
    newdata = pd.read_csv(new_data_path)
    newdata_x = newdata.iloc[:,0:63]
    predicted = predict(model, newdata_x)
    pd.DataFrame({'Predicted': predicted.to_list()}).to_csv('predicted_unseendata.csv')
