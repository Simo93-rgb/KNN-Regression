import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

from .knn_parallel import KNN_Parallel
from .valutazione import mean_squared_error, validate_predictions


def _get_results_dir(assets_dir: str = "") -> str:
    if assets_dir:
        return os.path.join(assets_dir, "results")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    return os.path.join(project_dir, "Assets", "results")


def plot_predictions(y_true, y_pred, model_name: str = "", assets_dir: str = "") -> None:
    y_true, y_pred = validate_predictions(y_true, y_pred)
    results_dir = _get_results_dir(assets_dir)

    plt.figure(figsize=(16, 9))
    plt.scatter(y_true, y_pred, alpha=0.5, color="green")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", lw=2)
    plt.title(f"Confronto Valori Predetti e Reali - {model_name}", fontsize=24)
    plt.xlabel("Valori Reali", fontsize=18)
    plt.ylabel("Valori Predetti", fontsize=18)
    plt.savefig(f"{results_dir}/predictions_{model_name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, model_name: str = "", assets_dir: str = "") -> None:
    y_true, y_pred = validate_predictions(y_true, y_pred)
    results_dir = _get_results_dir(assets_dir)

    residuals = y_true - y_pred
    plt.figure(figsize=(16, 9))
    plt.hist(residuals, bins=60, color="green")
    plt.title(f"Distribuzione dei Residui - {model_name}", fontsize=24)
    plt.xlabel("Errore di Predizione", fontsize=18)
    plt.ylabel("Frequenza", fontsize=18)
    plt.savefig(f"{results_dir}/residuals_{model_name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_corr_matrix(X, assets_dir: str = "") -> None:
    x_dim, y_dim = [20, 14]
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)
    plt.savefig(f"{assets_dir}/correlation_matrix.png", format="png", dpi=600, bbox_inches="tight")
    plt.close()


def plot_learning_curve(model, X_train, y_train, X_test, y_test, assets_dir: str = "", file_name: str = "learning_curve") -> None:
    y_train = y_train.values if hasattr(y_train, "values") else y_train
    y_test = y_test.values if hasattr(y_test, "values") else y_test
    train_errors = []
    val_errors = []

    training_sizes = np.linspace(0.1, 1.0, 10)

    for size in training_sizes:
        current_size = int(size * len(X_train))
        X_train_subset = X_train[:current_size]
        y_train_subset = y_train[:current_size]

        model.fit(X_train_subset, y_train_subset)

        y_train_pred = model.predict(X_train_subset)
        train_mse = root_mean_squared_error(y_train_subset, y_train_pred)
        train_errors.append(train_mse)

        y_val_pred = model.predict(X_test)
        val_mse = root_mean_squared_error(y_test, y_val_pred)
        val_errors.append(val_mse)

    plt.figure(figsize=(16, 9))
    plt.plot(training_sizes * len(X_train), train_errors, label="Errore Training Set", marker="o")
    plt.plot(training_sizes * len(X_train), val_errors, label="Errore Validation Set", marker="o")
    plt.xlabel("Dimensione Training Set")
    plt.ylabel("RMSE")
    plt.title("Curva di Apprendimento - KNN")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{assets_dir}/{file_name}.png", format="png", dpi=600, bbox_inches="tight")


def plot_rmse_vs_n_neighbors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors_range: range = range(1, 51),
) -> None:
    rmse_train_list = []
    rmse_test_list = []

    for k in n_neighbors_range:
        knn = KNN_Parallel(k=k)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_train_list.append(rmse_train)

        y_test_pred = knn.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmse_test_list.append(rmse_test)

    plt.figure(figsize=(10, 6))
    plt.plot(n_neighbors_range, rmse_train_list, label="Train RMSE", marker="o")
    plt.plot(n_neighbors_range, rmse_test_list, label="Test RMSE", marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Number of Neighbors in KNN")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_comparison(knn_metrics: dict, knn_sklearn_metrics: dict, title: str, assets_dir: str = "") -> None:
    metrics = list(knn_metrics.keys())
    knn_values = list(knn_metrics.values())
    knn_sklearn_values = list(knn_sklearn_metrics.values())
    results_dir = _get_results_dir(assets_dir)

    x = range(len(metrics))

    fig, ax = plt.subplots()
    bar_width = 0.35
    ax.bar(x, knn_values, width=bar_width, label="KNN")
    ax.bar([p + bar_width for p in x], knn_sklearn_values, width=bar_width, label="KNN_Sklearn")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_title(title)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.savefig(f"{results_dir}/comparison_{title}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
