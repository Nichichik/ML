import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator



def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
    """
    sorted_indices = np.argsort(feature_vector)
    feature_vector_sorted = feature_vector[sorted_indices]
    target_vector_sorted = target_vector[sorted_indices]
    unique_values, unique_indices = np.unique(feature_vector_sorted, return_index=True)

    if len(unique_values) < 2:
        return np.array([]), np.array([]), None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    n = len(target_vector)

    class_1_total = np.sum(target_vector)

    split_indices = unique_indices[1:] - 1

    left_size = split_indices + 1
    left_class_1_count = np.cumsum(target_vector_sorted)[split_indices]

    p1_left = left_class_1_count / left_size
    p0_left = 1 - p1_left
    gini_left = 1 - (p1_left**2 + p0_left**2)

    right_size = n - left_size
    right_class_1_count = class_1_total - left_class_1_count

    p1_right = right_class_1_count / right_size
    p0_right = 1 - p1_right
    gini_right = 1 - (p1_right**2 + p0_right**2)

    ginis = - (left_size / n) * gini_left - (right_size / n) * gini_right

    best_gini_index = np.argmax(ginis)
    threshold_best = thresholds[best_gini_index]
    gini_best = ginis[best_gini_index]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._tree = {}

    def get_params(self, deep=True):
        """
        Метод, который возвращает словарь с параметрами, заданными в __init__.
        Нужен для sklearn.
        """
        return {
            "feature_types": self.feature_types,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf
        }

    def set_params(self, **params):
        """
        Метод для установки новых параметров. Нужен для sklearn.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth):
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self.max_depth is not None and depth >= self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self.min_samples_split is not None and len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best = None, None, None
        best_categories_map = {}

        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}

                sorted_categories = sorted(ratio.keys(), key=lambda k: ratio[k])
                categories_map = {category: i for i, category in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map.get(x) for x in sub_X[:, feature]])

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                if feature_type == "categorical":
                    best_categories_map = categories_map

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_type = self.feature_types[feature_best]
        if feature_type == "real":
            split = sub_X[:, feature_best] < threshold_best
        elif feature_type == "categorical":
            left_categories = {k for k, v in best_categories_map.items() if v < threshold_best}
            split = np.isin(sub_X[:, feature_best], list(left_categories))

        right_split = np.logical_not(split)

        if self.min_samples_leaf is not None and (
                np.sum(split) < self.min_samples_leaf or np.sum(right_split) < self.min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            left_categories = {k for k, v in best_categories_map.items() if v < threshold_best}
            node["categories_split"] = list(left_categories)

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_split], sub_y[right_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # Реализация рекурсивного предсказания
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self.feature_types[feature_idx]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
