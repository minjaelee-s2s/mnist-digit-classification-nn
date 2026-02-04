import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from math import sqrt
import xgboost as xgb
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)





# 1. Load training data
X = pd.read_csv("train_features.txt", header=None, sep="\t")
y = pd.read_csv("train_labels.txt", header=None, sep="\t")[0]

# Convert to numpy
X = X.to_numpy()
y = y.to_numpy()


# 2. Remove monochrome predictors
#    (columns with < 3 unique values)
# The treshold is 3 becuase due to rounding, when converting into vectorforms, noises might split into 2.

def count_unique(col):
    return len(np.unique(col))

unique_counts = np.apply_along_axis(count_unique, 0, X)
mono_cols = np.where(unique_counts < 3)[0]

X = np.delete(X, mono_cols, axis=1)

# 3. Same thing for the test data

testX = pd.read_csv("test_features.txt", header=None, sep="\t").values
testy = pd.read_csv("test_labels.txt", header=None, sep="\t")[0].values

testX = np.delete(testX, mono_cols, axis=1)

# 4. Try two mtry values. (number of random values)

p = X.shape[1]
mtry1 = int(round(0.5 * p))
mtry2 = int(round(sqrt(p)))

print("Number of predictors =", p)
print("mtry1 (0.5p):", mtry1)
print("mtry2 (sqrt(p)):", mtry2)

# Helper: train & get OOB error

def train_rf(mtry):
    rf = RandomForestClassifier(
        n_estimators=5000,
        max_features=mtry,
        oob_score=True,
        n_jobs=-1,
        random_state=123
    )
    rf.fit(X, y)
    oob_error = 1 - rf.oob_score_
    return rf, oob_error

# 5. Train two RF models
print("\n--- Timing Random Forest (mtry1) ---")
start = time.time()
rf1, oob1 = train_rf(mtry1)
print("Time:", time.time() - start, "seconds")

print("\n--- Timing Random Forest (mtry2) ---")
start = time.time()
rf2, oob2 = train_rf(mtry2)
print("Time:", time.time() - start, "seconds")


print("OOB Error (0.5p):", oob1)
print("OOB Error (sqrt(p)):", oob2)

# 6. Pick best mtry
best_mtry = mtry1 if oob1 < oob2 else mtry2
print("Best mtry selected =", best_mtry)

# 7. Final model with best mtry 
final_rf = RandomForestClassifier(
    n_estimators=5000,
    max_features=best_mtry,
    oob_score=True,
    n_jobs=-1,
    random_state=123
)
print("\n--- Timing Final Random Forest ---")
start = time.time()
final_rf.fit(X, y)
print("Time:", time.time() - start, "seconds")

# 8. Variable importance plot
importances = final_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title(f"Variable Importance (mtry={best_mtry})")
plt.plot(importances[indices])
plt.xlabel("Feature index (sorted)")
plt.ylabel("Importance")
plt.show()

# 9. Predict on test set
pred = final_rf.predict(testX)
test_error = np.mean(pred != testy)

print("Test Misclassification Rate:", round(test_error, 7))

# Optional confusion matrix
from sklearn.metrics import confusion_matrix
print("\nConfusion Matrix:")
print(confusion_matrix(testy, pred))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------
# Boosted Trees (XGBoost) — Multiclass 0–9
# --------------------------------------------------

# Ensure labels are integers 0–9
y_numeric = y.astype(int)
testy_numeric = testy.astype(int)

np.random.seed(82)
total_rows = len(y_numeric)
all_indices = np.arange(total_rows)

# 20% validation split
val_size = int(np.floor(0.2 * total_rows))
rows_for_validation = np.random.choice(all_indices, size=val_size, replace=False)
rows_for_training = np.setdiff1d(all_indices, rows_for_validation)

X_train_boost = X[rows_for_training]
y_train_boost = y_numeric[rows_for_training]
X_vali_boost  = X[rows_for_validation]
y_vali_boost  = y_numeric[rows_for_validation]

dtrain_boosted = xgb.DMatrix(X_train_boost, label=y_train_boost)
dvali_boosted  = xgb.DMatrix(X_vali_boost,  label=y_vali_boost)
dtest_boosted  = xgb.DMatrix(testX,         label=testy_numeric)

tree_amounts  = [10000, 20000, 30000]
depth_amounts = [1, 2, 3, 4, 5, 6]
learning_value = 0.001

lowest_wrong = 1.0
best_combo = (None, None)

print("\n--- XGBoost Grid Search (with timing) ---")
print("Trees\tDepth\tValError\tTrainTime_sec")

for trees_try in tree_amounts:
    for depth_try in depth_amounts:
        params = {
            "max_depth": depth_try,
            "eta": learning_value,
            "objective": "multi:softmax",
            "num_class": 10,
            "verbosity": 0
        }

        start = time.time()
        boosting_model_try = xgb.train(
            params=params,
            dtrain=dtrain_boosted,
            num_boost_round=trees_try
        )
        train_time = time.time() - start

        pred_vali_try = boosting_model_try.predict(dvali_boosted).astype(int)
        error_vali_try = np.mean(pred_vali_try != y_vali_boost)

        print(f"{trees_try}\t{depth_try}\t{error_vali_try:.4f}\t{train_time:.2f}")

        if error_vali_try < lowest_wrong:
            lowest_wrong = error_vali_try
            best_combo = (trees_try, depth_try)

print(f"\nBest Parameters → Trees: {best_combo[0]} | Depth: {best_combo[1]} | "
      f"Validation Error: {lowest_wrong:.4f}")

# Final model on full training data
dfull = xgb.DMatrix(X, label=y_numeric)

final_params = {
    "max_depth": best_combo[1],
    "eta": learning_value,
    "objective": "multi:softmax",
    "num_class": 10,
    "verbosity": 0
}

print("\n--- Timing Final XGBoost Model ---")
start = time.time()
final_model_boosting = xgb.train(
    params=final_params,
    dtrain=dfull,
    num_boost_round=best_combo[0]
)
final_train_time = time.time() - start
print("Final XGBoost Train Time (sec):", f"{final_train_time:.2f}")

# Test predictions
prediction_test_boost = final_model_boosting.predict(dtest_boosted).astype(int)
final_error_test_boost = np.mean(prediction_test_boost != testy_numeric)

print(f"\nTest Misclassification Rate (Boosted Tree): {final_error_test_boost:.4f}")


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


np.random.seed(349)
tf.random.set_seed(349)


# 1. Prepare labels and splits
weird_labels_nn = y.astype(int)
y_test_tensor   = testy.astype(int)

how_many_rows = weird_labels_nn.shape[0]
all_indices    = np.arange(how_many_rows)

val_size = int(0.2 * how_many_rows)

rng = np.random.default_rng(349)
validation_indices = rng.choice(all_indices, size=val_size, replace=False)
training_indices   = np.setdiff1d(all_indices, validation_indices)

tiny_X_train = X[training_indices, :]
tiny_y_train = weird_labels_nn[training_indices]
tiny_X_valid = X[validation_indices, :]
tiny_y_valid = weird_labels_nn[validation_indices]

X_train_tensor = tiny_X_train.astype("float32")
y_train_tensor = tiny_y_train.astype("int32")
X_valid_tensor = tiny_X_valid.astype("float32")
y_valid_tensor = tiny_y_valid.astype("int32")

X_test_tensor = testX.astype("float32")

# 2. Hyperparameter
amount_of_neurons   = [50, 100, 150, 200]
prob_of_dropout     = [0.2, 0.3, 0.4, 0.5]
chunk_sizes         = [100, 200, 300]
fixed_learning_step = 0.001

best_valid_error_so_far = 1.0
remembers_best_config   = None

n_features = X.shape[1]

# 3. Grid search (timed)
print("\n===== Starting GRID SEARCH =====")
grid_start = time.time()

for neurons_now in amount_of_neurons:
    for drop_now in prob_of_dropout:
        for batch_try in chunk_sizes:

            net_model = keras.Sequential([
                layers.Dense(neurons_now, activation="relu", input_shape=(n_features,)),
                layers.Dropout(drop_now),
                layers.Dense(10, activation="softmax")
            ])

            net_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=fixed_learning_step),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            stopping_check = keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            )

            net_model.fit(
                X_train_tensor,
                y_train_tensor,
                epochs=5000,
                batch_size=batch_try,
                validation_data=(X_valid_tensor, y_valid_tensor),
                callbacks=[stopping_check],
                verbose=0
            )

            final_valid_results = net_model.evaluate(
                X_valid_tensor,
                y_valid_tensor,
                verbose=0
            )
            acc_val_now        = float(final_valid_results[1])
            valid_mistake_rate = 1.0 - acc_val_now

            print(
                "Neurons:", neurons_now,
                "| Dropout:", drop_now,
                "| Batch Size:", batch_try,
                "| Validation Error Rate:", round(valid_mistake_rate, 4)
            )

            if valid_mistake_rate < best_valid_error_so_far:
                best_valid_error_so_far = valid_mistake_rate
                remembers_best_config   = (neurons_now, drop_now, batch_try)

grid_end = time.time()
grid_time = grid_end - grid_start
print("\nGrid Search Completed in {:.2f} seconds ({:.2f} minutes)\n".format(grid_time, grid_time/60))

print("Best Parameters → Neurons: {} | Dropout: {} | Batch Size: {} | Validation Error: {:.4f}".format(
    remembers_best_config[0],
    remembers_best_config[1],
    remembers_best_config[2],
    best_valid_error_so_far
))

# 4. Final model training 
print("\n===== Starting FINAL TRAINING =====")
final_train_start = time.time()

X_full_tensor = X.astype("float32")
y_full_tensor = weird_labels_nn.astype("int32")

best_neurons, best_dropout, best_batch = remembers_best_config

final_model_nn = keras.Sequential([
    layers.Dense(best_neurons, activation="relu", input_shape=(n_features,)),
    layers.Dropout(best_dropout),
    layers.Dense(10, activation="softmax")
])

final_model_nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=fixed_learning_step),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

stopping_final = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

final_model_nn.fit(
    X_full_tensor,
    y_full_tensor,
    epochs=5000,
    batch_size=best_batch,
    validation_split=0.2,
    callbacks=[stopping_final],
    verbose=0
)

final_train_end = time.time()
final_train_time = final_train_end - final_train_start

print("\nFinal Training Completed in {:.2f} seconds ({:.2f} minutes)\n".format(
    final_train_time, final_train_time/60))

# 5. Test accuracy
predicted_test_probs   = final_model_nn.predict(X_test_tensor)
predicted_test_classes = np.argmax(predicted_test_probs, axis=1)

nn_test_error = np.mean(predicted_test_classes != y_test_tensor)

print("\nTest Misclassification Rate (Neural Network, 0–9): {:.4f}".format(nn_test_error))

# 6. Total wall time
total_time = (grid_time + final_train_time)
print("\n===== TOTAL RUNTIME: {:.2f} seconds ({:.2f} minutes) =====".format(
    total_time, total_time/60))


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Make sure labels are integers 0–9
labels_mlp = y.astype(int)

# Reproducibility
np.random.seed(888)
tf.random.set_seed(888)

# Train/validation split (20% validation, like R)
all_id_set = np.arange(len(labels_mlp))
val_size = int(0.2 * len(all_id_set))

rng = np.random.default_rng(888)
validation_id_mlp = rng.choice(all_id_set, size=val_size, replace=False)
training_id_mlp = np.setdiff1d(all_id_set, validation_id_mlp)

X_mlp_train = X[training_id_mlp, :].astype("float32")
y_mlp_train = labels_mlp[training_id_mlp].astype("int32")

X_mlp_valid = X[validation_id_mlp, :].astype("float32")
y_mlp_valid = labels_mlp[validation_id_mlp].astype("int32")

X_mlp_test = testX.astype("float32")
y_mlp_test = testy.astype(int)

# Hyperparameter grids
first_layer_sizes = [50, 100, 150, 200]
second_layer_sizes = [50, 100, 150, 200]
first_drops = [0.2, 0.3, 0.4, 0.5]
second_drops = [0.2, 0.3, 0.4, 0.5]
mini_batch_options = [100, 200, 300]
fixed_rate_mlp = 0.001

lowest_valid_mistake = 1.0
best_mlp_combo = None

n_features = X.shape[1]

# -----------------------------
# Grid search over architecture
# -----------------------------
for neurons1 in first_layer_sizes:
    for neurons2 in second_layer_sizes:
        for drop1 in first_drops:
            for drop2 in second_drops:
                for batch_mlp in mini_batch_options:

                    # Model setup: Dense -> Dropout -> Dense -> Dropout -> Dense(10, softmax)
                    model_mlp = keras.Sequential([
                        layers.Dense(neurons1, activation="relu", input_shape=(n_features,)),
                        layers.Dropout(drop1),
                        layers.Dense(neurons2, activation="relu"),
                        layers.Dropout(drop2),
                        layers.Dense(10, activation="softmax")  # 10 classes (0–9)
                    ])

                    model_mlp.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=fixed_rate_mlp),
                        loss="sparse_categorical_crossentropy",  # integer labels 0..9
                        metrics=["accuracy"]
                    )

                    stopping_gate = keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )

                    model_mlp.fit(
                        X_mlp_train,
                        y_mlp_train,
                        validation_data=(X_mlp_valid, y_mlp_valid),
                        epochs=5000,
                        batch_size=batch_mlp,
                        callbacks=[stopping_gate],
                        verbose=0
                    )

                    score_now = model_mlp.evaluate(X_mlp_valid, y_mlp_valid, verbose=0)
                    acc_now = float(score_now[1])  # accuracy
                    error_now = 1.0 - acc_now

                    print(
                        "First Layer:", neurons1,
                        "| Second Layer:", neurons2,
                        "| Drop1:", drop1,
                        "| Drop2:", drop2,
                        "| Batch:", batch_mlp,
                        "| Valid Error:", round(error_now, 4)
                    )

                    if error_now < lowest_valid_mistake:
                        lowest_valid_mistake = error_now
                        best_mlp_combo = (neurons1, neurons2, drop1, drop2, batch_mlp)

print(
    "\nBest → L1:", best_mlp_combo[0],
    "| L2:", best_mlp_combo[1],
    "| Drop1:", best_mlp_combo[2],
    "| Drop2:", best_mlp_combo[3],
    "| Batch:", best_mlp_combo[4],
    "| Validation Error:", round(lowest_valid_mistake, 4)
)

# -----------------------------
# Final model on all training data
# -----------------------------
X_mlp_all = X.astype("float32")
y_mlp_all = labels_mlp.astype("int32")

best_L1, best_L2, best_drop1, best_drop2, best_batch = best_mlp_combo

final_net = keras.Sequential([
    layers.Dense(best_L1, activation="relu", input_shape=(n_features,)),
    layers.Dropout(best_drop1),
    layers.Dense(best_L2, activation="relu"),
    layers.Dropout(best_drop2),
    layers.Dense(10, activation="softmax")
])

final_net.compile(
    optimizer=keras.optimizers.Adam(learning_rate=fixed_rate_mlp),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

final_net.fit(
    X_mlp_all,
    y_mlp_all,
    epochs=5000,
    batch_size=best_batch,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

# -----------------------------
# Test prediction and error
# -----------------------------
predictions_mlp = final_net.predict(X_mlp_test)
classes_mlp = np.argmax(predictions_mlp, axis=1)  # 0..9

test_error_mlp = np.mean(classes_mlp != y_mlp_test)

print("\nTest Misclassification Rate (Multi-layer Neural Net, 0–9):",
      round(test_error_mlp, 4))