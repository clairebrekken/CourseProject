import os
import pandas as pd
import tensorflow as tf
import warnings
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import space_eval
from hyperopt import tpe
from hyperopt.fmin import fmin
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import precision_recall_curve

plt.ion()

# Deep Learning Global Variables
OPTIMIZERS = dict(
    adamax=tf.keras.optimizers.Adamax,
    adam=tf.keras.optimizers.Adam,
    rmsprop=tf.keras.optimizers.RMSprop,
    nadam=tf.keras.optimizers.Nadam,
)

ACTIVATION_FUNCTIONS = dict(
    relu=tf.keras.activations.relu,
    sigmoid=tf.keras.activations.sigmoid,
    tanh=tf.keras.activations.tanh,
    swish=tf.keras.activations.swish,
)

SPACE = {
    "opt": hp.choice("opt", ["adamax", "adam", "rmsprop", "nadam"]),
    "learning_rate": hp.lognormal(
        "learning_rate", -6.2146, 0.75
    ),  # log(0.002) = -6.2146, .75 std
    "num_nodes": hp.choice("num_nodes", [8, 16, 32, 64, 128]),
    "num_layers": hp.choice("num_layers", [1, 2, 4, 8, 16]),
    "dropout_rate": hp.choice("dropout_rate", [0.0, 0.05]),
    "activation": hp.choice("activation", ["relu", "tanh", "swish"]),
}


def make_model(
    length,
    opt,
    learning_rate,
    num_nodes,
    num_layers,
    dropout_rate,
    activation
):
    losses = list()
    metrics = dict()
    input_layer = layers.Input(shape=[length])
    hidden = layers.Dense(num_nodes, activation=ACTIVATION_FUNCTIONS[activation])(input_layer)
    hidden = layers.Dropout(rate=dropout_rate)(hidden)
    for _ in range(num_layers-1):
        hidden = layers.Dense(num_nodes, activation=ACTIVATION_FUNCTIONS[activation])(hidden)
        hidden = layers.Dropout(rate=dropout_rate)(hidden)
    output_layer = layers.Dense(1, activation="sigmoid", name="sarcasm")(hidden)
    losses.append("binary_crossentropy")
    metrics["sarcasm"] = ["Precision", "Recall"]

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=losses, optimizer=OPTIMIZERS[opt](learning_rate), metrics=metrics)

    return model

def run():
    data_dir = os.path.join(os.getcwd(), 'data')
    results_dir = os.path.join(os.getcwd(), 'results')

    # training data
    # file_train = 'train.jsonl'

    # training data
    file_train = 'train_feature_engineering.csv';
    file_test = 'test_feature_engineering.csv'

    # file_train = os.path.join(data_dir, file_train)
    file_train = os.path.join(data_dir, file_train)
    file_test = os.path.join(data_dir, file_test)

    # df_train = utils.parse_json(file_train)
    df_train = pd.read_csv(file_train)
    # feats
    x_train = df_train.drop(columns=['label'])
    # labels
    y_train = df_train.label
    # convert labels to binary (1 - sarcasm)
    y_train = (y_train == 'SARCASM').astype(int)

    # Get train data (80% of data)
    x_train, x_test, y_train, y_test = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # split test data into test and validation for final model
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=.5,
        random_state=42,
        stratify=y_test
    )
    # Split train into train and val for hyperparameter tuning
    x_train_sample, x_val_params, y_train_sample, y_val_params = train_test_split(
        x_train,
        y_train,
        test_size=.2,
        random_state=42,
        stratify=y_train
    )


    def objective(params):
        model = make_model(
            len(x_train_sample.keys()),
            **params
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            x_train_sample,
            y_train_sample,
            epochs=100,
            batch_size=256,
            validation_data=(x_val_params, y_val_params),
            callbacks=[early_stop],
            verbose=0,
        )
        early_stop_loss = model.evaluate(
            x_val, y_val, verbose=0, return_dict=True
        )
        print(f"Early stop loss: {early_stop_loss}")
        loss = model.evaluate(x_val, y_val, verbose=2, return_dict=True)["loss"]
        print(f"Model with params {params} has loss of {loss}")
        results = {"loss": loss, "status": STATUS_OK}

        tf.keras.backend.clear_session()
        return results

    TRIALS = Trials()
    print("starting hyperparameter optimization")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(fn=objective, space=SPACE, algo=tpe.suggest, max_evals=50, trials=TRIALS)
    print("done")

    best_params = space_eval(SPACE, best)
    print(f"The best params after hyperopt are {best_params}")

    model = make_model(len(x_train.keys()),
                       **best_params)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    print("Fitting final model")
    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=256,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=0,
    )
    print("Finished fitting model")
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print(results)
    predictions = model.predict(x_test)
    # print(predictions)
    unpacked_predictions = [row[0] for row in predictions]
    test_preds = pd.DataFrame(unpacked_predictions)
    print(y_test)
    print(test_preds)

    precision, recall, threshold = precision_recall_curve(
        y_test, test_preds
    )
    precision_filtered_lst = [(x, y) for x, y in enumerate(precision) if y >= 0.723]
    recall_filtered_lst = [(x, y) for x, y in enumerate(recall) if y >= 0.723]
    print(f"min precision index: {min(precision_filtered_lst)}")
    print(f"max recall index: {max(recall_filtered_lst)}")
    print(threshold[min(precision_filtered_lst)[0]:max(recall_filtered_lst)[0]])
    print(threshold[min(precision_filtered_lst)[0]])
    print(threshold[max(recall_filtered_lst)[0]])
    precision = precision[:-1]
    recall = recall[:-1]
    plt.plot(threshold, precision, color="blue", label="Precision")
    plt.plot(threshold, recall, color="red", label="Recall")
    plt.xlabel("Probability of Sarcasm Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.show()
    fig_file = os.path.join(results_dir, "precision_recall_curve.png")
    plt.savefig(fig_file)
    plt.clf()

    # Prediction model:
    x_train = df_train.drop(columns=['label'])
    # labels
    y_train = df_train.label
    # convert labels to binary (1 - sarcasm)
    y_train = (y_train == 'SARCASM').astype(int)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    model = make_model(len(x_train.keys()),
                       **best_params)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    print("Fitting final model")
    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=256,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=0,
    )
    print("Finished fitting model")

    df_test = pd.read_csv(file_test)
    x_test = df_test.drop(columns=["id"])
    y_test = df_test.id

    predictions = model.predict(x_test)
    unpacked_predictions = [row[0] for row in predictions]
    test_preds = pd.DataFrame(unpacked_predictions)
    test_preds.insert(loc=0, column='id', value=y_test)
    test_preds["answer"] = test_preds[0].apply(lambda x: "SARCASM" if x > 0.6 else "NOT_SARCASM")

    test_preds[["id", "answer"]].to_csv("results/answer.txt", index=False, header=False)


if __name__ == "__main__":
    run()
