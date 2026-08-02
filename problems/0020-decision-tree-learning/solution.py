import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total_labels = [ex[target_attr] for ex in examples]
    total_entropy = calculate_entropy(total_labels)

    # Group examples by the value of `attr`
    subsets = {}
    for ex in examples:
        subsets.setdefault(ex[attr], []).append(ex)

    total = len(examples)
    weighted_entropy = 0.0
    for value, subset in subsets.items():
        weight = len(subset) / total
        subset_labels = [ex[target_attr] for ex in subset]
        weighted_entropy += weight * calculate_entropy(subset_labels)

    return total_entropy - weighted_entropy


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    max_count = max(counts.values())
    # Among classes with max_count, choose alphabetically first
    candidates = [cls for cls, cnt in counts.items() if cnt == max_count]
    return sorted(candidates)[0]


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]

    # Base case 1: all examples have the same label -> pure leaf
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: no attributes left to split on -> majority class leaf
    if not attributes:
        return majority_class(examples, target_attr)

    # Choose the attribute with the highest information gain.
    # Ties broken by order in `attributes` (first one wins) since we only
    # update best_gain on strictly greater gain.
    best_attr = None
    best_gain = -1.0
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]

    # Group examples by value of best_attr
    values_to_examples = {}
    for ex in examples:
        values_to_examples.setdefault(ex[best_attr], []).append(ex)

    # Process values in sorted order for consistent tree structure
    for value in sorted(values_to_examples.keys()):
        subset = values_to_examples[value]
        if not subset:
            tree[best_attr][value] = majority_class(examples, target_attr)
        else:
            tree[best_attr][value] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return tree