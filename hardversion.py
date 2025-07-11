import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import time
from collections import Counter, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def make_features(str1, str2, equal, operation):
    # Lots of features that may be important, think about adding more specific cases, same leading or tailing chars? 
    # These are the basic ones tho
    feature_order = [
        'str1 len', 'str2 len', 'equal len', 'all len', 'unique char',
        'max occurences', 'min occurences', 'reuse ratio', 'operation',
        'unique str1 char', 'unique str2 char', 'unique equal char',
        'unique 1+2 char', 'max str1 occurences', 'min str1 occurences',
        'max str2 occurences', 'min str2 occurences', 'max equal occurences',
        'min equal occurences', 'max 1+2 occurences', 'min 1+2 occurences',
        'crypto complexity'
    ]
    
    
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

    # Create DataFrame with values in the exact order we want
    feature_values = [
        str1_len, str2_len, equal_len, all_len, unique_char,
        max_occurences, min_occurences, reuse_ratio, operation_encoded,
        unique_str1_char, unique_str2_char, unique_equal_char,
        unique_1_2_char, max_str1_occurences, min_str1_occurences,
        max_str2_occurences, min_str2_occurences, max_equal_occurences,
        min_equal_occurences, max_1_2_occurences, min_1_2_occurences,
        crypto_complexity
    ]
    return pd.DataFrame([feature_values], columns=feature_order)

    
# Asked ChatGPT to generate a list of word patterns to use for learning the best type
word_patterns = [
    # Original patterns
    ("AB", "CD", "EF"),
    ("ABC", "DEF", "GHIJ"),
    ("AB", "CD", "EFG"),
    ("ABCD", "EFGH", "IJKLM"),
    ("ABC", "ABC", "DEFG"),
    ("A", "BC", "DE"),
    ("AB", "C", "DE"),
    
    # Single digit patterns
    ("A", "B", "C"),
    ("A", "B", "CD"),
    ("A", "B", "DE"),
    ("A", "BC", "DE"),
    ("A", "BCD", "EFG"),
    ("A", "BCDE", "FGHI"),
    
    # Two digit patterns
    ("AB", "C", "D"),
    ("AB", "C", "DE"),
    ("AB", "CD", "E"),
    ("AB", "CD", "EF"),
    ("AB", "CD", "EFG"),
    ("AB", "CD", "EFGH"),
    ("AB", "CDE", "FGH"),
    ("AB", "CDE", "FGHI"),
    ("AB", "CDEF", "GHIJ"),
    
    # Three digit patterns
    ("ABC", "D", "E"),
    ("ABC", "D", "EF"),
    ("ABC", "D", "EFG"),
    ("ABC", "DE", "F"),
    ("ABC", "DE", "FG"),
    ("ABC", "DE", "FGH"),
    ("ABC", "DEF", "G"),
    ("ABC", "DEF", "GH"),
    ("ABC", "DEF", "GHI"),
    ("ABC", "DEF", "GHIJ"),
    ("ABC", "DEFG", "HIJK"),
    ("ABC", "DEFG", "HIJKL"),
    
    # Four digit patterns
    ("ABCD", "E", "F"),
    ("ABCD", "E", "FG"),
    ("ABCD", "E", "FGH"),
    ("ABCD", "EF", "G"),
    ("ABCD", "EF", "GH"),
    ("ABCD", "EF", "GHI"),
    ("ABCD", "EFG", "HIJ"),
    ("ABCD", "EFG", "HIJK"),
    ("ABCD", "EFGH", "IJKL"),
    ("ABCD", "EFGH", "IJKLM"),
    ("ABCD", "EFGHI", "JKLMN"),
    
    # Five digit patterns
    ("ABCDE", "F", "G"),
    ("ABCDE", "FG", "H"),
    ("ABCDE", "FGH", "I"),
    ("ABCDE", "FGHI", "J"),
    ("ABCDE", "FGHIJ", "KLMNO"),
    
    # Repeated operand patterns (important for specific puzzle types)
    ("A", "A", "B"),
    ("A", "A", "BC"),
    ("AB", "AB", "C"),
    ("AB", "AB", "CD"),
    ("AB", "AB", "CDE"),
    ("ABC", "ABC", "D"),
    ("ABC", "ABC", "DE"),
    ("ABC", "ABC", "DEF"),
    ("ABC", "ABC", "DEFG"),
    ("ABC", "ABC", "DEFGH"),
    ("ABCD", "ABCD", "E"),
    ("ABCD", "ABCD", "EF"),
    ("ABCD", "ABCD", "EFG"),
    ("ABCD", "ABCD", "EFGH"),
    ("ABCD", "ABCD", "EFGHI"),
    ("ABCDE", "ABCDE", "F"),
    ("ABCDE", "ABCDE", "FG"),
    ("ABCDE", "ABCDE", "FGH"),
    
    # Overlapping character patterns (some shared letters)
    ("AB", "BC", "CD"),
    ("AB", "BC", "CDE"),
    ("ABC", "BCD", "CDE"),
    ("ABC", "BCD", "CDEF"),
    ("ABC", "ADE", "FGH"),
    ("ABC", "ACE", "DEF"),
    ("ABCD", "BCDE", "EFGH"),
    ("ABCD", "ACEF", "BEGH"),
    
    # Asymmetric patterns
    ("A", "BCDE", "FGHI"),
    ("AB", "CDEFG", "HIJKL"),
    ("ABC", "DEFGHI", "JKLMNO"),
    ("ABCDE", "F", "GHIJKL"),
    ("ABCDEF", "G", "HIJKLMN"),
    
    # Medium complexity patterns
    ("AB", "CDE", "FGHI"),
    ("ABC", "DEFG", "HIJKL"),
    ("ABCD", "EFGHI", "JKLMNO"),
    ("AB", "CDEF", "GHIJK"),
    ("ABC", "DEFGH", "IJKLMN"),
    
    # High reuse patterns (for testing character frequency)
    ("AA", "BB", "CC"),
    ("AAB", "BBC", "CCD"),
    ("AABB", "BBCC", "CCDD"),
    ("AAA", "BBB", "CCC"),
    ("AAAA", "BBBB", "CCCC"),
    
    # Mixed reuse patterns
    ("AA", "BC", "DE"),
    ("AB", "CC", "DE"),
    ("AB", "CD", "EE"),
    ("AAB", "CCD", "EEF"),
    ("ABB", "CDD", "EFF"),
    ("ABC", "AAD", "EEF"),
    
    # Edge case patterns (very short or very long)
    ("A", "B", "C"),           # Minimal
    ("ABCDEFG", "H", "I"),     # Very unbalanced
    ("A", "BCDEFGH", "IJKLMNO"), # Extremely unbalanced
    ("ABCDEFGH", "IJKLMNOP", "QRSTUVWXY"), # Very long
    
    # Specific challenging patterns
    ("ABCD", "DCBA", "EFGHI"),  # Reversed patterns
    ("ABC", "CBA", "DEFG"),     # Reversed patterns
    ("AB", "BA", "CDE"),        # Reversed patterns
    
    # Multiplication-friendly patterns (shorter results)
    ("AB", "C", "DE"),
    ("ABC", "D", "EFG"),
    ("AB", "CD", "EFG"),
    ("ABC", "DE", "FGHI"),
    
    # Division-friendly patterns (larger first operand)
    ("ABCD", "AB", "CD"),
    ("ABCDE", "ABC", "DE"),
    ("ABCDEF", "ABC", "DEF"),
    ("ABCDEFG", "ABCD", "EFG"),
    
    # Subtraction patterns (first >= second, result <= first)
    ("ABC", "AB", "C"),
    ("ABCD", "ABC", "D"),
    ("ABCDE", "ABCD", "E"),
    ("ABC", "DE", "FG"),
    ("ABCD", "EF", "GHI"),
    
    # Patterns with maximum character usage (close to 10 unique)
    ("ABCD", "EFGH", "IJ"),     # 10 chars
    ("ABC", "DEFG", "HIJ"),     # 10 chars  
    ("AB", "CDEFG", "HIJ"),     # 10 chars
    ("ABCDE", "FGH", "IJ"),     # 10 chars
    ("AB", "CDEF", "GHIJ"),     # 10 chars
    
    # Real-world inspired patterns (based on actual puzzles)
    ("SEND", "MORE", "MONEY"),  # Classic
    ("TWO", "TWO", "FOUR"),     # Classic
    ("ONE", "ONE", "TWO"),      # Simple
    ("SIX", "SIX", "NINE"),     # Medium
    ("NINE", "FIVE", "EIGHT"),  # Complex
    ("FORTY", "TEN", "SIXTY"),  # Complex
    ("CROSS", "ROADS", "DANGER"), # Very complex
]

def generate_training_data(num_samples=5000):
    """Generate synthetic training data for the ML model - UPDATED to use new make_features"""
    data = []
    labels = []
    
    # Known puzzles
    valid_puzzles = [
        ("SEND", "MORE", "MONEY", "+"),
        ("TWO", "TWO", "FOUR", "+"),
        ("ABC", "DEF", "GHIJ", "+"),
        ("CAB", "CAB", "DEED", "+"),
        ("EAT", "THAT", "APPLE", "+"),
        ("WE", "WE", "PLAY", "*"),          
        ("US", "US", "YEAH", "*"),          
        ("AB", "C", "DE", "*"),              
        ("A", "BC", "DE", "*"),              
        ("GREAT", "LARGE", "SMALL", "-"),    
        ("WRONG", "RIGHT", "LEFT", "-"),    
        ("MONEY", "SEND", "MORE", "-"),
        ("FORTY", "TEN", "THIRTY", "-")
    ]
    
    
    for str1, str2, result, op in valid_puzzles:
        features_df = make_features(str1, str2, result, op)
        data.append(features_df.iloc[0].to_dict()) 
        labels.append(1)
    
    operations = ['+', '-', '*', '/']

    # Generate possibly valid puzzles
    for _ in range(num_samples // 2):
        pattern = np.random.choice(len(word_patterns))
        str1_pattern, str2_pattern, result_pattern = word_patterns[pattern]
        operation = np.random.choice(operations)
        
        def pattern_to_word(pattern):
            return ''.join(chr(ord('A') + np.random.randint(0, 26)) for _ in pattern)
        
        str1 = pattern_to_word(str1_pattern)
        str2 = pattern_to_word(str2_pattern)
        result = pattern_to_word(result_pattern)
        
        features_df = make_features(str1, str2, result, operation)
        data.append(features_df.iloc[0].to_dict())
        
        # Label based on heuristics
        unique_chars = len(set(str1 + str2 + result))
        if unique_chars <= 10 and len(result) <= max(len(str1), len(str2)) + 2:
            labels.append(1) # Valid
        else:
            labels.append(0) # Invalid
 
    # Impossible Puzzles
    for _ in range(num_samples // 2):
        invalid_type = np.random.randint(0, 4)
        
        if invalid_type == 0:
            
            str1 = ''.join(chr(ord('A') + i) for i in range(6))
            str2 = ''.join(chr(ord('G') + i) for i in range(6))
            result = ''.join(chr(ord('M') + i) for i in range(6))
            operation = '+'
        elif invalid_type == 1:
            
            str1 = "ABC"
            str2 = "DEF"
            result = "GHIJKLM"
            operation = '+'
        elif invalid_type == 2:
            
            str1 = "ABCDE"
            str2 = "FGHIJ"
            result = "KL"
            operation = '+'
        else:
            
            str1 = ''.join(chr(ord('A') + np.random.randint(0, 26)) for _ in range(np.random.randint(1, 8)))
            str2 = ''.join(chr(ord('A') + np.random.randint(0, 26)) for _ in range(np.random.randint(1, 8)))
            result = ''.join(chr(ord('A') + np.random.randint(0, 26)) for _ in range(np.random.randint(1, 12)))
            operation = np.random.choice(['+', '-', '*', '/'])
        
        features_df = make_features(str1, str2, result, operation)
        data.append(features_df.iloc[0].to_dict())
        labels.append(0)
    
    return data, labels


def train_classifier():
    # Trianing RF Classifier
    data, labels = generate_training_data()

    df = pd.DataFrame(data)
    X = df.fillna(0)
    y = np.array(labels)
    print(Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    y_pred = clf.predict(X_test)
    print(f"\nClassification Report:")
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)
    print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None))
    
    print(classification_report(y_test, y_pred))
    
    return clf, X.columns