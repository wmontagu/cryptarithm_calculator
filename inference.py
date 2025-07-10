import numpy as np
import pandas as pd
from collections import Counter

def make_features(str1, str2, equal, operation):
    # Feature extraction function - same as in hardversion.py
    # This only needs pandas and numpy, not scikit-learn
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
