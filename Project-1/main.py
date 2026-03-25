import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data, split_data, standardize, one_hot_encode, CLASS_NAMES
from utils.metrics import evaluate, plot_combined_loss, plot_combined_cm
from models.numpy_mlp import NumpyMLP
from models.sklearn_mlp import SklearnMLP
from models.pytorch_mlp import PytorchMLP
from sklearn.metrics import accuracy_score


def eda(path="data/Dry_Bean_Dataset.xlsx"):
    """Exploratory Data Analysis"""
    print("=" * 60)
    print("0. Exploratory Data Analysis (EDA)")
    print("=" * 60)

    df = pd.read_excel(path)
    print(f"\nOrnek sayisi: {df.shape[0]}, Ozellik sayisi: {df.shape[1] - 1}")
    print(f"\n--- Istatistikler ---")
    print(df.describe().round(2))
    print(f"\nNull deger: {df.isnull().sum().sum()}")

    print(f"\n--- Sinif Dagilimi ---")
    print(df["Class"].value_counts())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df["Class"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Sinif Dagilimi")
    axes[0].set_xlabel("Sinif")
    axes[0].set_ylabel("Ornek Sayisi")
    axes[0].tick_params(axis="x", rotation=45)

    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, ax=axes[1], cmap="coolwarm", annot=False,
                xticklabels=True, yticklabels=True)
    axes[1].set_title("Korelasyon Matrisi")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig("plots/eda_plots.png", dpi=150)
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 4][i % 4]
        df.boxplot(column=col, by="Class", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Ozellik Dagilimlari (Sinifa Gore)", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=150)
    plt.close()
    print("\nEDA grafikleri plots/ klasorune kaydedildi.\n")


def accuracy_vs_nsteps(X_train, y_train_oh, X_test, y_test,
                       n_features, n_classes, lr, seed):
    """Accuracy vs n_steps analizi"""
    print("=" * 60)
    print("6. Accuracy vs n_steps Analizi")
    print("=" * 60)

    epoch_values = [100, 200, 300, 500, 750, 1000]
    configs = [
        ("1 Hidden [16,64,7]", [n_features, 64, n_classes], 0.0),
        ("2 Hidden [16,64,32,7]", [n_features, 64, 32, n_classes], 0.0),
        ("3 Hidden [16,64,32,16,7]", [n_features, 64, 32, 16, n_classes], 0.0),
        ("3 Hidden+L2 [16,64,32,16,7]", [n_features, 64, 32, 16, n_classes], 0.01),
    ]

    step_results = {name: [] for name, _, _ in configs}

    for name, layers, l2 in configs:
        for ep in epoch_values:
            model = NumpyMLP(layer_sizes=layers, learning_rate=lr, seed=seed, l2_lambda=l2)
            model.fit(X_train, y_train_oh, epochs=ep, print_every=0)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            step_results[name].append((ep, acc))

    # Tablo
    print(f"\n{'Model':<35}", end="")
    for ep in epoch_values:
        print(f"{'n=' + str(ep):>10}", end="")
    print()
    print("-" * (35 + 10 * len(epoch_values)))

    for name in step_results:
        print(f"{name:<35}", end="")
        for _, acc in step_results[name]:
            print(f"{acc:>10.4f}", end="")
        print()

    # Model secimi
    print("\n--- Model Secimi ---")
    print("Kriter: %90 accuracy'i gecen modellerde en dusuk n_steps\n")
    best_model = None
    best_epochs = float("inf")
    for name in step_results:
        for ep, acc in step_results[name]:
            if acc >= 0.90 and ep < best_epochs:
                best_epochs = ep
                best_model = (name, ep, acc)

    if best_model:
        print(f"Secilen model: {best_model[0]}")
        print(f"n_steps: {best_model[1]}, Accuracy: {best_model[2]:.4f}")

    # Grafik
    plt.figure(figsize=(12, 6))
    styles = [
        {"marker": "o", "linestyle": "-"},
        {"marker": "s", "linestyle": "-"},
        {"marker": "^", "linestyle": "--"},
        {"marker": "D", "linestyle": ":"},
    ]
    for i, name in enumerate(step_results):
        epochs_list = [x[0] for x in step_results[name]]
        accs = [x[1] for x in step_results[name]]
        s = styles[i % len(styles)]
        plt.plot(epochs_list, accs, marker=s["marker"], linestyle=s["linestyle"],
                 label=name, markersize=8, linewidth=2)
    plt.axhline(y=0.90, color="r", linestyle="--", label="Threshold (%90)")
    plt.xlabel("n_steps (Epoch)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs n_steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/accuracy_vs_nsteps.png", dpi=150)
    plt.close()
    print("Grafik plots/accuracy_vs_nsteps.png olarak kaydedildi.\n")


def main():
    # ======== 0. EDA ========
    eda()

    # ======== 1. Data Loading & Preprocessing ========
    print("=" * 60)
    print("1. Loading and preprocessing data...")
    print("=" * 60)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test, mean, std = standardize(X_train, X_val, X_test)
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(CLASS_NAMES)}\n")

    LR = 0.01
    EPOCHS = 1000
    SEED = 42
    N_FEATURES = X_train.shape[1]
    N_CLASSES = len(CLASS_NAMES)

    # ======== 2. NumPy Models ========
    models_config = [
        ("Model 1: 1 Hidden Layer", [N_FEATURES, 64, N_CLASSES], 0.0),
        ("Model 2: 2 Hidden Layers", [N_FEATURES, 64, 32, N_CLASSES], 0.0),
        ("Model 3: 3 Hidden Layers", [N_FEATURES, 64, 32, 16, N_CLASSES], 0.0),
        ("Model 4: 3 Hidden + L2", [N_FEATURES, 64, 32, 16, N_CLASSES], 0.01),
    ]

    results = {}
    numpy_loss_data = []
    numpy_cm_data = []

    for name, layers, l2 in models_config:
        print("=" * 60)
        print(f"Training: {name}")
        print(f"Architecture: {layers}, L2={l2}")
        print("=" * 60)

        model = NumpyMLP(layer_sizes=layers, learning_rate=LR, seed=SEED, l2_lambda=l2)
        train_losses, val_losses = model.fit(
            X_train, y_train_oh, X_val, y_val_oh, epochs=EPOCHS, print_every=200
        )
        preds = model.predict(X_test)

        print(f"\n--- {name} Test Results ---")
        metrics = evaluate(y_test, preds, class_names=CLASS_NAMES)
        results[name] = metrics

        numpy_loss_data.append((name, train_losses, val_losses))
        numpy_cm_data.append((name, y_test, preds))
        print()

    # Birlesik NumPy grafikleri
    plot_combined_loss(numpy_loss_data,
                       title="NumPy Models - Loss Curves",
                       save_path="plots/numpy_loss_comparison.png")
    plot_combined_cm(numpy_cm_data, class_names=CLASS_NAMES,
                     title="NumPy Models - Confusion Matrices",
                     save_path="plots/numpy_confusion_matrices.png", ncols=2)

    # ======== 3. Scikit-learn Model ========
    print("=" * 60)
    print("Training: Scikit-learn MLPClassifier (2 Hidden Layers)")
    print("=" * 60)

    sklearn_model = SklearnMLP(
        hidden_layer_sizes=(64, 32), learning_rate=LR, seed=SEED, l2_lambda=0.0
    )
    sk_train_losses, _ = sklearn_model.fit(X_train, y_train_oh, epochs=EPOCHS)
    sk_preds = sklearn_model.predict(X_test)

    print("\n--- Scikit-learn Test Results ---")
    sk_metrics = evaluate(y_test, sk_preds, class_names=CLASS_NAMES)
    results["Scikit-learn MLP"] = sk_metrics

    # ======== 4. PyTorch Model ========
    print("=" * 60)
    print("Training: PyTorch MLP (2 Hidden Layers)")
    print("=" * 60)

    pytorch_model = PytorchMLP(
        layer_sizes=[N_FEATURES, 64, 32, N_CLASSES], learning_rate=LR, seed=SEED, l2_lambda=0.0
    )
    pt_train_losses, pt_val_losses = pytorch_model.fit(
        X_train, y_train_oh, X_val, y_val_oh, epochs=EPOCHS, print_every=200
    )
    pt_preds = pytorch_model.predict(X_test)

    print("\n--- PyTorch Test Results ---")
    pt_metrics = evaluate(y_test, pt_preds, class_names=CLASS_NAMES)
    results["PyTorch MLP"] = pt_metrics

    # Birlesik kutuphane grafikleri
    # Loss: NumPy best (Model 2) vs Sklearn vs PyTorch
    best_numpy = numpy_loss_data[1]  # Model 2
    lib_loss_data = [
        ("NumPy (Model 2)", best_numpy[1], best_numpy[2]),
        ("Scikit-learn", sk_train_losses, []),
        ("PyTorch", pt_train_losses, pt_val_losses),
    ]
    plot_combined_loss(lib_loss_data,
                       title="Library Comparison - Loss Curves",
                       save_path="plots/library_loss_comparison.png")

    lib_cm_data = [
        ("NumPy (Model 2)", y_test, numpy_cm_data[1][2]),
        ("Scikit-learn", y_test, sk_preds),
        ("PyTorch", y_test, pt_preds),
    ]
    plot_combined_cm(lib_cm_data, class_names=CLASS_NAMES,
                     title="Library Comparison - Confusion Matrices",
                     save_path="plots/library_confusion_matrices.png", ncols=3)

    # ======== 5. Model Comparison ========
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<40} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<40} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # ======== 6. Accuracy vs n_steps ========
    accuracy_vs_nsteps(X_train, y_train_oh, X_test, y_test,
                       N_FEATURES, N_CLASSES, LR, SEED)


if __name__ == "__main__":
    main()
