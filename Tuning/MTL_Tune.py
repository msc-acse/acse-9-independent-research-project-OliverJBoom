"""
This file performs hyperparameter tuning for an MTL framework

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from Foresight.deeplearning import set_seed, DeepLearning, model_save
from Foresight.preprocessing import universe_select, truncate_window_length, \
clean_dict_gen, generate_dataset, price_rename, dimension_reduce,\
    dimension_selector, feature_spawn, slice_series
from Foresight.eval_inspect import evaluate
from Foresight.models import LSTM

torch.nn.Module.dump_patches = True

def main():

    set_seed(42)

    device = 'cpu'

    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = 'cuda'

        if torch.cuda.device_count() > 1:
            print("Parallel Available")
            parallel = True

        else:
            parallel = False

    else:
        print("No GPU available!")
        parallel = False

    # Loading data and setting run details
    path = "../Data/"
    universe_dict = universe_select(path, "MTL")
    use_lg_returns = False
    auto_regressive = False
    use_pca = True
    feat_spawn = False
    saving = False

    print("MTL\nAutoregressive = ", auto_regressive)

    if auto_regressive:
        assert auto_regressive != use_pca

    # Renaming the columns to price
    universe_dict = price_rename(universe_dict)

    # Cleaning the dataset of any erroneous datapoints
    universe_dict = clean_dict_gen(universe_dict, verbose=False)

    # Making sure that all the points in the window have consistent length
    universe_dict = truncate_window_length(universe_dict)

    # # Generating the dataset
    if use_lg_returns:
        # Lg Returns Only
        df_full = generate_dataset(universe_dict, lg_only=True,
                                   price_only=False)
        target_col = ["cu_lme", "al_lme", "sn_lme", "pb_lme", "ni_lme"]

    else:
        # Price Only
        df_full = generate_dataset(universe_dict, lg_only=False,
                                   price_only=True)
        target_col = ["price_cu_lme", "price_al_lme", "price_sn_lme",
                      "price_pb_lme", "price_ni_lme"]

    if auto_regressive:
        df_full = df_full[target_col]

    if feat_spawn:
        df_full = feature_spawn(df_full)

    for forecast_length in [5, 22, 66, 132]:
        print("\nForecast Length:", forecast_length)

        model_name = "MTL_Full_F" + str(forecast_length)

        # Data scaling
        scaler_data_X = MinMaxScaler()
        scaler_data_y = MinMaxScaler()

        df_target = df_full[target_col]

        data_X = scaler_data_X.fit_transform(df_full)[:-forecast_length, :]
        # Need to have an independent scaler for inverse_transforming later
        data_y = scaler_data_y.fit_transform(df_target)

        # Offset target one forecast length
        data_y = data_y[forecast_length:, :]

        # Reducing the dataset to containing a threshold amount of variance
        if use_pca:
            n_dim = dimension_selector(data_X, thresh=0.95, verbose=False)
            data_X = dimension_reduce(data_X, n_dim, verbose=False)

        # The input size of each time series window
        series_length = 132

        data_X, data_y = slice_series(data_X, data_y, series_length)

        # Hyperparameters
        learning_rate = 1e-2
        weight_decay = 0

        # Batch Parameters
        batch_size = 32

        # Training Parameters
        n_epochs = 1000
        patience = 40
        disp_freq = 50
        fig_disp_freq = 0

        # Model Parameters
        num_features = data_X.shape[2]
        hidden_dim = 8
        dense_hidden = 32
        num_layers = 1
        output_dim = data_y.shape[1]
        dropout = 0.0

        model = LSTM(num_features=num_features,
                     hidden_dim=hidden_dim,
                     dense_hidden=dense_hidden,
                     series_length=series_length,
                     batch_size=batch_size,
                     output_dim=output_dim,
                     num_layers=num_layers,
                     device=device,
                     dropout=dropout)

        if parallel:
            model = nn.DataParallel(model)
            print("Parallel Workflow\n")

        model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), learning_rate,
                                     weight_decay=weight_decay)

        learning = DeepLearning(model=model,
                                data_X=data_X,
                                data_y=data_y,
                                n_epochs=n_epochs,
                                optimiser=optimiser,
                                batch_size=batch_size,
                                debug=False,
                                disp_freq=disp_freq,
                                fig_disp_freq=fig_disp_freq,
                                device=device,
                                patience=patience,
                                scaler_data_X=scaler_data_X,
                                scaler_data_y=scaler_data_y)

        # Splitting the data into the train, validation and test sets
        learning.train_val_test()
        learning.training_wrapper()
        learning.evaluate(learning.best_model, learning.test_loader)

        # Observed
        train_true = learning.scaler_data_y.inverse_transform(
            learning.y_train.numpy())
        val_true = learning.scaler_data_y.inverse_transform(
            learning.y_val.numpy())
        test_true = learning.scaler_data_y.inverse_transform(
            learning.y_test.numpy())

        # Predicted
        train_pred = learning.scaler_data_y.inverse_transform(
            np.array(learning.train_predictions))
        val_pred = learning.scaler_data_y.inverse_transform(
            np.array(learning.val_predictions))
        test_pred = learning.scaler_data_y.inverse_transform(
            np.array(learning.test_predictions))

        mse, mae, mda = evaluate(test_pred[:, :1], test_true[:, :1],
                                 log_ret=False)
        print("Copper Price Metrics: %i %i %.1f" % (mse, mae, 100 * mda))

        test_naive = test_true[:-forecast_length, :]
        mse_naive, mae_naive, mda_naive = evaluate(test_naive,
                                                   test_true[forecast_length:],
                                                   log_ret=False)
        print("Naive:  %i %i %.1f" % (mse_naive, mae_naive, 100 * mda_naive))

        # Saving Plots
        if saving:
            _, ax = plt.subplots(2, 3, figsize=(16, 12))

            ax[0, 0].set_title("Training Predictions")
            ax[0, 0].plot(train_true, label="Observed")
            ax[0, 0].plot(train_pred, '--', label="Predicted")
            ax[0, 0].grid()
            ax[0, 0].legend()

            ax[0, 1].grid()
            ax[0, 1].set_title("Validation Predictions")
            ax[0, 1].plot(val_true, label="Observed")
            ax[0, 1].plot(val_pred, '--', label="Predictions")
            ax[0, 1].legend()

            ax[1, 0].grid()
            ax[1, 0].set_title("Test Predictions")
            ax[1, 0].plot(test_true, label="Observed")
            ax[1, 0].plot(test_pred, '--', label="Predictions")
            ax[1, 0].legend()

            ax[1, 1].grid()
            ax[1, 1].set_title("Loss Plots")
            ax[1, 1].plot(learning.logs['Training Loss'],
                          label="Training Loss")
            ax[1, 1].plot(learning.logs['Validation Loss'],
                          label="Validation Loss")
            ax[1, 1].legend()

            ax[0, 2].grid()
            ax[0, 2].set_title("Train Predictions")
            ax[0, 2].plot(train_true[:, 0], label="Observed")
            ax[0, 2].plot(train_pred[:, 0], '--', label="Predictions")
            ax[0, 2].legend()

            ax[1, 2].grid()
            ax[1, 2].set_title("Test Predictions")
            ax[1, 2].plot(test_true[:, 0], label="Observed")
            ax[1, 2].plot(test_pred[:, 0], '--', label="Predictions")
            ax[1, 2].legend()

            print("Saving", model_name)

            # Saving Plots
            plt.savefig(
                "../Results/Plots/MTL/Multivariate/" + model_name + ".png")
            # Saving Pickle
            pickle.dump(learning,
                        open(
                            "../Results/Pickles/MTL/Multivariate/" + model_name,
                            'wb'))
            # Saving Pth
            model_save(model, model_name,
                       path="../Results/Pths/MTL/Multivariate/")

if __name__ == "__main__":
    main()
