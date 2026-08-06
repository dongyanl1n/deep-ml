def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """

    merges = []
    for i_merge in range(num_merges):
        # if all words are single tokens, stop early
        if " "  not in "".join(corpus.keys()):
            break

        adjacent_tokens = {}
        # Count all adjacent token pairs across the corpus, weighted by word frequency
        for word, freq in corpus.items():
            word_chars = word.split()
            for i, char in enumerate(word_chars[:-1]):
                if char+" "+word_chars[i+1] not in adjacent_tokens:
                    adjacent_tokens[char+" "+word_chars[i+1]] = freq
                else:
                    adjacent_tokens[char+" "+word_chars[i+1]] += freq
        
        # Find the most frequent pair
        most_count = max(adjacent_tokens.values())
        most_freq_pair = next((k for k, v in adjacent_tokens.items() if v == most_count), None)
        
        # Merge that pair everywhere it appears in the corpus (replacing the two tokens with their concatenation)
        # Build a NEW corpus instead of mutating the one you're iterating over to avoid RuntimeError: dictionary keys changed during iteration 
        new_corpus = {}
        for word, freq in corpus.items():
            new_word = word.replace(most_freq_pair, "".join(most_freq_pair.split()))
            new_corpus[new_word] = new_corpus.get(new_word, 0) + freq
        corpus = new_corpus

        # Record the merge operation
        merges.append(tuple(most_freq_pair.split()))

    return merges







