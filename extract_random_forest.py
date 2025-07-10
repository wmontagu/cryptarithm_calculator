import pickle
import json
import numpy as np

def extract_decision_tree(tree):
    """Extract a single sklearn decision tree structure"""
    def extract_node(node_id):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node
            return {
                'is_leaf': True,
                'value': tree.value[node_id].tolist(),
                'samples': int(tree.n_node_samples[node_id])
            }
        else:
            # Internal node
            return {
                'is_leaf': False,
                'feature': int(tree.feature[node_id]),
                'threshold': float(tree.threshold[node_id]),
                'left': extract_node(tree.children_left[node_id]),
                'right': extract_node(tree.children_right[node_id]),
                'samples': int(tree.n_node_samples[node_id])
            }
    
    return extract_node(0)

def extract_sklearn_random_forest(model_path='model.pkl', output_path='sklearn_random_forest.json'):
    """Extract sklearn Random Forest to JSON format"""
    
    print(f"Loading sklearn model from {model_path}...")
    
    # Load the sklearn model
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"Model type: {type(rf_model).__name__}")
    print(f"Number of estimators: {rf_model.n_estimators}")
    print(f"Number of features: {rf_model.n_features_in_}")
    print(f"Classes: {rf_model.classes_}")
    
    # Extract the forest structure
    forest_data = {
        'model_type': 'SklearnRandomForestClassifier',
        'n_estimators': rf_model.n_estimators,
        'classes_': rf_model.classes_.tolist(),
        'n_features_': rf_model.n_features_in_,
        'trees': []
    }
    
    print(f"\nExtracting {rf_model.n_estimators} trees...")
    
    for i, estimator in enumerate(rf_model.estimators_):
        if i % 10 == 0 or i == rf_model.n_estimators - 1:
            print(f"Processing tree {i+1}/{rf_model.n_estimators}")
        
        tree_structure = extract_decision_tree(estimator.tree_)
        forest_data['trees'].append(tree_structure)
    
    # Save to JSON
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(forest_data, f, indent=2)
    
    print(f"✅ Successfully extracted sklearn Random Forest!")
    print(f"📊 Trees: {forest_data['n_estimators']}")
    print(f"📊 Features: {forest_data['n_features_']}")
    print(f"📊 Classes: {forest_data['classes_']}")
    print(f"💾 Saved to: {output_path}")
    
    # Calculate file sizes
    import os
    pkl_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    json_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\n📏 File sizes:")
    print(f"  Original .pkl: {pkl_size:.2f} MB")
    print(f"  Extracted .json: {json_size:.2f} MB")
    
    return forest_data

def extract_other_sklearn_models(model_path='model.pkl', output_path='sklearn_model.json'):
    """Extract other sklearn models (LogisticRegression, SVM, etc.)"""
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_type = type(model).__name__
    print(f"Model type: {model_type}")
    
    if 'LogisticRegression' in model_type:
        model_data = {
            'model_type': 'LogisticRegression',
            'coef_': model.coef_.tolist(),
            'intercept_': model.intercept_.tolist(),
            'classes_': model.classes_.tolist(),
            'n_features_': model.n_features_in_
        }
    
    elif 'SVC' in model_type or 'SVM' in model_type:
        model_data = {
            'model_type': model_type,
            'support_vectors_': model.support_vectors_.tolist(),
            'support_': model.support_.tolist(),
            'n_support_': model.n_support_.tolist(),
            'dual_coef_': model.dual_coef_.tolist(),
            'intercept_': model.intercept_.tolist(),
            'classes_': model.classes_.tolist(),
            'gamma': model.gamma if hasattr(model, 'gamma') else None,
            'kernel': model.kernel if hasattr(model, 'kernel') else None
        }
    
    elif 'DecisionTree' in model_type:
        model_data = {
            'model_type': model_type,
            'tree': extract_decision_tree(model.tree_),
            'classes_': model.classes_.tolist(),
            'n_features_': model.n_features_in_
        }
    
    else:
        print(f"❌ Model type {model_type} not supported yet")
        print("Available attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and hasattr(model, attr):
                try:
                    value = getattr(model, attr)
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: shape {value.shape}")
                    elif isinstance(value, (int, float, str)):
                        print(f"  {attr}: {value}")
                    else:
                        print(f"  {attr}: {type(value)}")
                except:
                    print(f"  {attr}: (cannot access)")
        return None
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"✅ Successfully extracted {model_type}!")
    print(f"💾 Saved to: {output_path}")
    
    return model_data

if __name__ == "__main__":
    try:
        # Try to extract as Random Forest first
        extract_sklearn_random_forest()
    except Exception as e:
        print(f"Not a Random Forest model: {e}")
        print("Trying other model types...")
        
        try:
            extract_other_sklearn_models()
        except Exception as e2:
            print(f"Failed to extract model: {e2}")
            print("Please check that model.pkl exists and is a supported sklearn model.")