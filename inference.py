import json
import math
from collections import Counter

def make_features(str1, str2, equal, operation):
    """Feature extraction function - returns list instead of pandas DataFrame"""
    
    str1_len = len(str1)
    str2_len = len(str2)
    equal_len = len(equal)
    all_len = str1_len + str2_len + equal_len

    overallCount = Counter(str1 + str2 + equal)
    unique_char = len(overallCount)
    max_occurences = max(overallCount.values())
    min_occurences = min(overallCount.values())
    reuse_ratio = all_len / unique_char if unique_char != 0 else 0
    
    if operation == '+':
        operation_encoded = 1
    elif operation == '*':
        operation_encoded = 2
    elif operation == '/':
        operation_encoded = 3
    else:
        operation_encoded = 4
    
    str1Count = Counter(str1)
    str2Count = Counter(str2)
    equalCount = Counter(equal)
    sumCount = Counter(str1 + str2)

    unique_str1_char = len(str1Count)
    unique_str2_char = len(str2Count)
    unique_equal_char = len(equalCount)
    unique_1_2_char = len(sumCount)

    max_str1_occurences = max(str1Count.values())
    min_str1_occurences = min(str1Count.values())
    max_str2_occurences = max(str2Count.values())
    min_str2_occurences = min(str2Count.values())
    max_equal_occurences = max(equalCount.values())
    min_equal_occurences = min(equalCount.values())
    max_1_2_occurences = max(sumCount.values())
    min_1_2_occurences = min(sumCount.values())

    crypto_complexity = all_len * unique_char

    # Return list of feature values in exact order
    feature_values = [
        str1_len, str2_len, equal_len, all_len, unique_char,
        max_occurences, min_occurences, reuse_ratio, operation_encoded,
        unique_str1_char, unique_str2_char, unique_equal_char,
        unique_1_2_char, max_str1_occurences, min_str1_occurences,
        max_str2_occurences, min_str2_occurences, max_equal_occurences,
        min_equal_occurences, max_1_2_occurences, min_1_2_occurences,
        crypto_complexity
    ]
    
    return feature_values

class SklearnDecisionTree:
    """Extracted sklearn decision tree that works without sklearn"""
    
    def __init__(self, tree_structure):
        self.tree = tree_structure
    
    def predict_sample(self, sample):
        """Predict a single sample by traversing the tree"""
        node = self.tree
        
        while not node['is_leaf']:
            feature_value = sample[node['feature']]
            
            if feature_value <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        # For sklearn trees, value is [n_samples, n_classes]
        # Get the class with the highest count
        values = node['value'][0]  # Get the class counts
        return values.index(max(values))

class SklearnRandomForest:
    """Extracted sklearn Random Forest that works without sklearn"""
    
    def __init__(self, model_data=None, json_file=None):
        if json_file:
            with open(json_file, 'r') as f:
                model_data = json.load(f)
        
        if model_data['model_type'] != 'SklearnRandomForestClassifier':
            raise ValueError(f"Expected SklearnRandomForestClassifier, got {model_data['model_type']}")
        
        self.classes_ = model_data['classes_']
        self.n_estimators = model_data['n_estimators']
        self.n_features_ = model_data['n_features_']
        
        # Create individual tree objects
        self.trees = []
        for tree_data in model_data['trees']:
            self.trees.append(SklearnDecisionTree(tree_data))
        
        print(f"✅ Loaded sklearn Random Forest with {self.n_estimators} trees")
    
    def predict(self, X):
        """Predict classes for samples"""
        if isinstance(X[0], (int, float)):
            # Single sample
            return [self._predict_single(X)]
        else:
            # Multiple samples
            return [self._predict_single(sample) for sample in X]
    
    def _predict_single(self, sample):
        """Predict a single sample using all trees"""
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            pred = tree.predict_sample(sample)
            tree_predictions.append(pred)
        
        # Vote: return most common prediction
        vote_counts = Counter(tree_predictions)
        winning_class_idx = vote_counts.most_common(1)[0][0]
        return self.classes_[winning_class_idx]
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X[0], (int, float)):
            # Single sample
            return [self._predict_proba_single(X)]
        else:
            # Multiple samples
            return [self._predict_proba_single(sample) for sample in X]
    
    def _predict_proba_single(self, sample):
        """Get probability distribution for a single sample"""
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            pred = tree.predict_sample(sample)
            tree_predictions.append(pred)
        
        # Calculate probabilities based on votes
        vote_counts = Counter(tree_predictions)
        probabilities = []
        
        for i, class_label in enumerate(self.classes_):
            class_votes = vote_counts.get(i, 0)
            probability = class_votes / self.n_estimators
            probabilities.append(probability)
        
        return probabilities

class SklearnLogisticRegression:
    """Extracted sklearn Logistic Regression that works without sklearn"""
    
    def __init__(self, model_data=None, json_file=None):
        if json_file:
            with open(json_file, 'r') as f:
                model_data = json.load(f)
        
        if model_data['model_type'] != 'LogisticRegression':
            raise ValueError(f"Expected LogisticRegression, got {model_data['model_type']}")
        
        self.coef_ = model_data['coef_']
        self.intercept_ = model_data['intercept_']
        self.classes_ = model_data['classes_']
        self.n_features_ = model_data['n_features_']
        
        print(f"✅ Loaded sklearn Logistic Regression")
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X[0], (int, float)):
            # Single sample
            z = sum(coef * feature for coef, feature in zip(self.coef_[0], X)) + self.intercept_[0]
            prob_positive = self._sigmoid(z)
            return [[1 - prob_positive, prob_positive]]
        else:
            # Multiple samples
            probabilities = []
            for sample in X:
                z = sum(coef * feature for coef, feature in zip(self.coef_[0], sample)) + self.intercept_[0]
                prob_positive = self._sigmoid(z)
                probabilities.append([1 - prob_positive, prob_positive])
            return probabilities
    
    def predict(self, X):
        """Predict classes"""
        probabilities = self.predict_proba(X)
        if isinstance(X[0], (int, float)):
            return [self.classes_[1] if probabilities[0][1] > 0.5 else self.classes_[0]]
        else:
            return [self.classes_[1] if prob[1] > 0.5 else self.classes_[0] for prob in probabilities]

def load_sklearn_model(json_file):
    """Load any supported sklearn model from JSON"""
    with open(json_file, 'r') as f:
        model_data = json.load(f)
    
    model_type = model_data['model_type']
    
    if model_type == 'SklearnRandomForestClassifier':
        return SklearnRandomForest(model_data=model_data)
    elif model_type == 'LogisticRegression':
        return SklearnLogisticRegression(model_data=model_data)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Usage example and testing
if __name__ == "__main__":
    try:
        # Load the extracted sklearn model
        model = load_sklearn_model('sklearn_random_forest.json')
        
        # Test with known examples
        test_cases = [
            ("SEND", "MORE", "MONEY", "+"),
            ("TWO", "TWO", "FOUR", "+"),
            ("ABC", "DEF", "GHIJKL", "+"),  # Should be unsolvable
            ("A", "B", "C", "+"),
        ]
        
        print("\n🧪 Testing predictions:")
        for str1, str2, result, op in test_cases:
            features = make_features(str1, str2, result, op)
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            
            solvable = "Solvable" if prediction == 1 else "Not Solvable"
            confidence = max(probabilities)
            
            print(f"  {str1} {op} {str2} = {result}")
            print(f"    Prediction: {solvable} (confidence: {confidence:.3f})")
            print()
            
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("Run extract_sklearn_model.py first to extract your sklearn model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        