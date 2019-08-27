"""
This file performs hyper-parameter tuning for an univariate or
multivariate framework

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from Foresight.deeplearning import set_seed, DeepLearning
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
    universe_dict = universe_select(path, "Cu", custom_list=None)
    use_lg_returns = False
    use_pca = False
    autoregressive = False
    feat_spawn = False

    # Renaming the columns to price
    universe_dict = price_rename(universe_dict)

    # Cleaning the dataset of any erroneous datapoints
    universe_dict = clean_dict_gen(universe_dict, verbose=False)

    # Making sure that all the points in the window have consistent length
    universe_dict = truncate_window_length(universe_dict)

    # Generating the dataset
    if use_lg_returns:
        # Lg Returns Only
        df_full = generate_dataset(universe_dict, lg_only=True,
                                   price_only=False)
        target_col = "cu_lme"

    else:
        # Price Only
        df_full = generate_dataset(universe_dict, lg_only=False,
                                   price_only=True)
        target_col = "price_cu_lme"

    if autoregressive:
        df_full = df_full[[target_col]]

    if feat_spawn:
        df_full = feature_spawn(df_full)

    for forecast_length in [5, 22, 66, 132]:
        print("\nForecast Length:", forecast_length)

        # Data scaling
        scaler_data_X = MinMaxScaler()
        scaler_data_y = MinMaxScaler()

        df_target = df_full[[target_col]]

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
        weight_decay = 0.0

        # Batch Parameters
        batch_size = 32

        # Training Parameters
        n_epochs = 1000
        patience = 50
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
        test_true = learning.scaler_data_y.inverse_transform(
            learning.y_test.numpy())

        # Predicted
        test_pred = learning.scaler_data_y.inverse_transform(
            np.array(learning.test_predictions))

        mse, mae, mda = evaluate(test_pred, test_true, log_ret=False)
        print("Copper Price Metrics: %i %i %.1f" % (mse, mae, 100 * mda))

        test_naive = test_true[:-forecast_length]
        mse_naive, mae_naive, mda_naive = evaluate(test_naive,
                                                   test_true[forecast_length:],
                                                   log_ret=False)
        print("Naive:  %i %i %.1f" % (mse_naive, mae_naive, 100 * mda_naive))


if __name__ == "__main__":
    main()
