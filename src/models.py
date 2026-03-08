"""Model definitions: scikit-learn pipelines and the Keras neural network."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_NAMES, build_features, signed_log1p

SEED = 42


def _feature_names(transformer, input_features):
    return FEATURE_NAMES


def feature_step():
    return FunctionTransformer(build_features, feature_names_out=_feature_names)


def scaled_preprocessor():
    """Features -> median imputation -> log compression -> standard scaling (for LR and the NN)."""
    return make_pipeline(
        feature_step(),
        SimpleImputer(strategy="median"),
        FunctionTransformer(signed_log1p),
        StandardScaler(),
    )


def make_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            scaled_preprocessor(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Decision Tree": make_pipeline(
            feature_step(),
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED),
        ),
        "Random Forest": make_pipeline(
            feature_step(),
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=50, max_features="sqrt",
                n_jobs=-1, random_state=SEED,
            ),
        ),
        # Handles missing values natively, so no imputer.
        "Gradient Boosting": make_pipeline(
            feature_step(),
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=50,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, random_state=SEED,
            ),
        ),
    }


def build_nn(n_features: int):
    import keras

    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def fit_nn(X, y, epochs: int = 60, verbose: int = 0):
    """Fit the scaled preprocessor + NN. Returns (preprocessor, keras_model, history)."""
    import keras

    pre = scaled_preprocessor().fit(X)
    Xt = pre.transform(X).astype("float32")
    model = build_nn(Xt.shape[1])
    stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=6, restore_best_weights=True
    )
    history = model.fit(
        Xt, y.to_numpy(), validation_split=0.15, epochs=epochs, batch_size=512,
        callbacks=[stop], verbose=verbose,
    )
    return pre, model, history


def predict_nn(pre, model, X):
    return model.predict(pre.transform(X).astype("float32"), verbose=0).ravel()
