# ============================================================
#  Bank Marketing Dataset – Full Analysis
#  Student : Kashish Shrestha  |  ID : 26140480
#  Pure Python – NO external libraries used
# ============================================================

import math
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────
# 1.  CSV READER  (Data Structure: List of Dicts)
# ─────────────────────────────────────────────────────────────

def read_csv(filepath):
    """Read CSV file and return header list and list-of-dict rows."""
    dataset = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    header = [col.strip() for col in lines[0].strip().split(",")]

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        values = line.split(",")
        row = {}
        for col, val in zip(header, values):
            row[col] = val.strip()
        dataset.append(row)

    print(f"[CSV] Loaded {len(dataset)} rows | {len(header)} columns")
    return header, dataset


# ─────────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────

def describe_column(dataset, col, numeric=True):
    """Return basic stats for a column."""
    values = []
    for row in dataset:
        try:
            values.append(float(row[col]))
        except ValueError:
            pass

    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    mn = min(values)
    mx = max(values)
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2

    return {"count": n, "mean": round(mean, 4), "std": round(std, 4),
            "min": mn, "median": median, "max": mx}


def value_counts(dataset, col):
    """Count occurrences of each unique value in a column."""
    counts = {}
    for row in dataset:
        v = row[col]
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def run_eda(dataset):
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    numeric_cols = ["age", "duration", "campaign", "pdays", "previous",
                    "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                    "euribor3m", "nr_employed"]
    categorical_cols = ["job", "marital", "education", "default",
                        "housing", "loan", "contact", "month",
                        "day_of_week", "poutcome"]

    print("\n--- Numeric Column Statistics ---")
    for col in numeric_cols:
        stats = describe_column(dataset, col)
        print(f"  {col:20s} | mean={stats['mean']:>10.4f}  std={stats['std']:>10.4f}"
              f"  min={stats['min']:>8}  max={stats['max']:>8}")

    print("\n--- Categorical Column Value Counts ---")
    for col in categorical_cols:
        vc = value_counts(dataset, col)
        top = list(vc.items())[:5]
        top_str = "  |  ".join(f"{k}: {v}" for k, v in top)
        print(f"  {col:15s} → {top_str}")

    # Target distribution
    vc_y = value_counts(dataset, "y")
    total = sum(vc_y.values())
    print("\n--- Target Variable (y) Distribution ---")
    for k, v in vc_y.items():
        label = "Subscribed    (Yes)" if k == "1" else "Not Subscribed (No)"
        print(f"  {label}: {v}  ({100 * v / total:.2f}%)")


# ─────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month",
                    "day_of_week", "poutcome"]

NUMERIC_COLS = ["age", "duration", "campaign", "pdays", "previous",
                "emp_var_rate", "cons_price_idx", "cons_conf_idx",
                "euribor3m", "nr_employed"]


def build_label_encoders(dataset, cols):
    """Data Structure: Dictionary (Hash Map) mapping category → int index."""
    encoders = {}
    for col in cols:
        unique_vals = sorted(set(row[col] for row in dataset))
        encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
    return encoders


def compute_normalization_params(dataset, cols):
    """Compute mean and std for each numeric column (for standardization)."""
    params = {}
    for col in cols:
        values = [float(row[col]) for row in dataset]
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
        std = std if std != 0 else 1.0          # avoid division by zero
        params[col] = (mean, std)
    return params


def preprocess(dataset, encoders, norm_params):
    """
    Convert raw rows into numeric feature vectors.
    Returns: X (list of lists), y (list of ints)
    """
    X, y = [], []
    for row in dataset:
        features = []

        # Label-encoded categoricals
        for col in CATEGORICAL_COLS:
            features.append(float(encoders[col].get(row[col], 0)))

        # Standardised numerics
        for col in NUMERIC_COLS:
            mean, std = norm_params[col]
            features.append((float(row[col]) - mean) / std)

        X.append(features)
        y.append(int(row["y"]))

    return X, y


def train_test_split(X, y, test_ratio=0.2):
    """Shuffle and split into train / test sets."""
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────
# 4.  METRICS 
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Accuracy, Precision, Recall, F1 – computed from scratch."""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    total     = len(y_true)
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return {"accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def print_metrics(name, metrics):
    print(f"\n  [{name}] Results")
    print(f"    Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"    Precision : {metrics['precision'] * 100:.2f}%")
    print(f"    Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"    F1 Score  : {metrics['f1']        * 100:.2f}%")
    print(f"    Confusion Matrix → TP:{metrics['tp']}  TN:{metrics['tn']}"
          f"  FP:{metrics['fp']}  FN:{metrics['fn']}")


# ─────────────────────────────────────────────────────────────
# 5.  LOGISTIC REGRESSION  (from scratch)
# ─────────────────────────────────────────────────────────────

class LogisticRegression:
    """
    Binary Logistic Regression trained via Gradient Descent.
    Data Structures used: Lists (weights), Matrices (dot products as loops)
    """

    def __init__(self, learning_rate=0.1, epochs=300):
        self.lr     = learning_rate
        self.epochs = epochs
        self.weights = []          # List acting as weight vector
        self.bias    = 0.0

    @staticmethod
    def _sigmoid(z):
        # Clamp to avoid overflow
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _dot(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias    = 0.0

        for epoch in range(self.epochs):
            # Forward pass – compute predictions
            predictions = [self._sigmoid(self._dot(x)) for x in X]

            # Compute gradients
            errors = [p - yi for p, yi in zip(predictions, y)]

            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                for j in range(n_features):
                    dw[j] += errors[i] * X[i][j]
                db += errors[i]

            # Update weights
            self.weights = [w - self.lr * (dw[j] / n_samples)
                            for j, w in enumerate(self.weights)]
            self.bias -= self.lr * (db / n_samples)

            # Log every 50 epochs
            if (epoch + 1) % 50 == 0:
                loss = -sum(
                    yi * math.log(p + 1e-9) + (1 - yi) * math.log(1 - p + 1e-9)
                    for p, yi in zip(predictions, y)
                ) / n_samples
                print(f"    Epoch {epoch + 1:>3}/{self.epochs}  |  Loss: {loss:.6f}")

    def predict(self, X, threshold=0.5):
        return [1 if self._sigmoid(self._dot(x)) >= threshold else 0 for x in X]

    def feature_importances(self, feature_names):
        """Return features ranked by absolute weight magnitude."""
        pairs = sorted(zip(feature_names, self.weights),
                       key=lambda x: abs(x[1]), reverse=True)
        return pairs



# ─────────────────────────────────────────────────────────────
# 6.  DECISION TREE  (from scratch – ID3 / Gini)
# ─────────────────────────────────────────────────────────────

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def best_split(X, y, max_features=None):
    """
    Find the best feature index and threshold that minimises weighted Gini.
    Data Structure: Dictionary used to track best split parameters.
    """
    n_samples  = len(y)
    n_features = len(X[0])
    feature_indices = list(range(n_features))

    if max_features and max_features < n_features:
        feature_indices = random.sample(feature_indices, max_features)

    best = {"gini": float("inf"), "feat": None, "thresh": None}

    for feat in feature_indices:
        # Collect candidate thresholds (midpoints of sorted unique values)
        values = sorted(set(row[feat] for row in X))
        thresholds = [(values[i] + values[i + 1]) / 2
                      for i in range(len(values) - 1)]

        for thresh in thresholds:
            left_y  = [y[i] for i in range(n_samples) if X[i][feat] <= thresh]
            right_y = [y[i] for i in range(n_samples) if X[i][feat] >  thresh]

            if not left_y or not right_y:
                continue

            weighted = (len(left_y)  * gini_impurity(left_y) +
                        len(right_y) * gini_impurity(right_y)) / n_samples

            if weighted < best["gini"]:
                best = {"gini": weighted, "feat": feat, "thresh": thresh}

    return best


class DecisionNode:
    """Tree node – either a split or a leaf."""
    def __init__(self, feat=None, thresh=None, left=None, right=None, value=None):
        self.feat   = feat
        self.thresh = thresh
        self.left   = left
        self.right  = right
        self.value  = value   # set only for leaf nodes


class DecisionTree:
    """
    Decision Tree Classifier built from scratch.
    Data Structure: Recursive tree of DecisionNode objects (Linked structure).
    """

    def __init__(self, max_depth=8, min_samples_split=10, max_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.root              = None
        self.feature_counts    = {}   # Dictionary tracking feature usage

    def _majority(self, y):
        counts = {}
        for lbl in y:
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get)

    def _build(self, X, y, depth):
        n = len(y)

        # Stopping conditions
        if depth >= self.max_depth or n < self.min_samples_split or len(set(y)) == 1:
            return DecisionNode(value=self._majority(y))

        split = best_split(X, y, self.max_features)
        if split["feat"] is None:
            return DecisionNode(value=self._majority(y))

        feat, thresh = split["feat"], split["thresh"]
        self.feature_counts[feat] = self.feature_counts.get(feat, 0) + 1

        left_mask  = [i for i in range(n) if X[i][feat] <= thresh]
        right_mask = [i for i in range(n) if X[i][feat] >  thresh]

        left_node  = self._build([X[i] for i in left_mask],
                                  [y[i] for i in left_mask], depth + 1)
        right_node = self._build([X[i] for i in right_mask],
                                  [y[i] for i in right_mask], depth + 1)

        return DecisionNode(feat=feat, thresh=thresh,
                            left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _predict_one(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feat] <= node.thresh:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X):
        return [self._predict_one(self.root, x) for x in X]

    def feature_importances(self, feature_names):
        """Return features ranked by split frequency (proxy for importance)."""
        total = sum(self.feature_counts.values()) or 1
        scores = [(feature_names[i], self.feature_counts.get(i, 0) / total)
                  for i in range(len(feature_names))]
        return sorted(scores, key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────
# 7.  RANDOM FOREST  (Optional Extension – from scratch)
# ─────────────────────────────────────────────────────────────

class RandomForest:
    """
    Ensemble of Decision Trees trained on bootstrap samples.
    Data Structure: List of DecisionTree objects.
    """

    def __init__(self, n_trees=10, max_depth=6, min_samples_split=15):
        self.n_trees          = n_trees
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.trees            = []   # List acting as ensemble container

    def _bootstrap(self, X, y):
        n = len(X)
        indices = [random.randint(0, n - 1) for _ in range(n)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X, y):
        n_features = len(X[0])
        max_f = max(1, int(math.sqrt(n_features)))   # sqrt(p) features per split

        for t in range(self.n_trees):
            Xb, yb = self._bootstrap(X, y)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=max_f)
            tree.fit(Xb, yb)
            self.trees.append(tree)
            print(f"    Tree {t + 1}/{self.n_trees} trained.")

    def predict(self, X):
        # Majority vote across all trees
        all_preds = [tree.predict(X) for tree in self.trees]
        final = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(self.n_trees)]
            final.append(1 if votes.count(1) > votes.count(0) else 0)
        return final

    def feature_importances(self, feature_names):
        """Aggregate feature counts from all trees."""
        combined = {}
        for tree in self.trees:
            for feat, score in tree.feature_importances(feature_names):
                combined[feat] = combined.get(feat, 0) + score
        total = sum(combined.values()) or 1
        sorted_feats = sorted(combined.items(), key=lambda x: -x[1])
        return [(f, round(s / total, 4)) for f, s in sorted_feats]


# ─────────────────────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE DISPLAY
# ─────────────────────────────────────────────────────────────

def print_feature_importance(importances, title, top_n=10):
    print(f"\n  Top {top_n} Features – {title}")
    print(f"  {'Feature':<25} {'Importance':>12}  {'Bar'}")
    print("  " + "-" * 60)
    for name, score in importances[:top_n]:
        bar = "█" * int(abs(score) * 40)
        print(f"  {name:<25} {score:>12.4f}  {bar}")


# ─────────────────────────────────────────────────────────────
# 9.  MAIN – PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BANK MARKETING ANALYSIS  –  Kashish Shrestha (26140480)")
    print("=" * 60)

    # --- Load data ---
    _, dataset = read_csv("banking.csv")

    # --- EDA ---
    run_eda(dataset)

    # --- Preprocessing ---
    print("\n" + "=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)

    encoders   = build_label_encoders(dataset, CATEGORICAL_COLS)
    norm_params = compute_normalization_params(dataset, NUMERIC_COLS)

    print("\n  Label Encoders (Dictionary / Hash Map):")
    for col, mapping in encoders.items():
        print(f"    {col:15s} → {dict(list(mapping.items())[:4])}{'...' if len(mapping) > 4 else ''}")

    X, y = preprocess(dataset, encoders, norm_params)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    print(f"\n  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")

    # Feature names list (matches column order in preprocess())
    feature_names = CATEGORICAL_COLS + NUMERIC_COLS

    # ── Logistic Regression ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL 1 : LOGISTIC REGRESSION")
    print("=" * 60)

    lr_model = LogisticRegression(learning_rate=0.1, epochs=300)
    lr_model.fit(X_train, y_train)

    lr_preds  = lr_model.predict(X_test)
    lr_metrics = compute_metrics(y_test, lr_preds)
    print_metrics("Logistic Regression", lr_metrics)

    lr_importance = lr_model.feature_importances(feature_names)
    print_feature_importance(lr_importance, "Logistic Regression (|weight|)")

    # ── Decision Tree ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL 2 : DECISION TREE")
    print("=" * 60)

    dt_model = DecisionTree(max_depth=8, min_samples_split=10)
    print("  Training decision tree …")
    dt_model.fit(X_train, y_train)
    print("  Training complete.")

    dt_preds   = dt_model.predict(X_test)
    dt_metrics = compute_metrics(y_test, dt_preds)
    print_metrics("Decision Tree", dt_metrics)

    dt_importance = dt_model.feature_importances(feature_names)
    print_feature_importance(dt_importance, "Decision Tree (split frequency)")

    # ── Random Forest (Optional Extension) ──────────────────
    print("\n" + "=" * 60)
    print("  MODEL 3 : RANDOM FOREST  (Optional Extension)")
    print("=" * 60)

    rf_model = RandomForest(n_trees=10, max_depth=6, min_samples_split=15)
    print("  Training random forest …")
    rf_model.fit(X_train, y_train)

    rf_preds   = rf_model.predict(X_test)
    rf_metrics = compute_metrics(y_test, rf_preds)
    print_metrics("Random Forest", rf_metrics)

    rf_importance = rf_model.feature_importances(feature_names)
    print_feature_importance(rf_importance, "Random Forest (aggregated splits)")

    # ── Model Comparison ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 65)
    for name, m in [("Logistic Regression", lr_metrics),
                    ("Decision Tree",       dt_metrics),
                    ("Random Forest",       rf_metrics)]:
        print(f"  {name:<22} {m['accuracy']*100:>9.2f}% {m['precision']*100:>9.2f}%"
              f" {m['recall']*100:>9.2f}% {m['f1']*100:>9.2f}%")

    # ── Key Insights (Problem Statement 2) ──────────────────
    print("\n" + "=" * 60)
    print("  KEY FACTORS INFLUENCING CUSTOMER DECISIONS")
    print("  (Consolidated from all three models)")
    print("=" * 60)

    # Aggregate normalised ranks across models
    all_feats = {}
    for rank, (feat, _) in enumerate(lr_importance):
        all_feats[feat] = all_feats.get(feat, 0) + rank
    for rank, (feat, _) in enumerate(dt_importance):
        all_feats[feat] = all_feats.get(feat, 0) + rank
    for rank, (feat, _) in enumerate(rf_importance):
        all_feats[feat] = all_feats.get(feat, 0) + rank

    final_ranking = sorted(all_feats.items(), key=lambda x: x[1])
    print(f"\n  {'Rank':<6} {'Feature':<25} Avg. Rank Position")
    print("  " + "-" * 50)
    for rank, (feat, score) in enumerate(final_ranking[:10], 1):
        bar = "★" * (11 - rank)
        print(f"  {rank:<6} {feat:<25} {bar}")

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()