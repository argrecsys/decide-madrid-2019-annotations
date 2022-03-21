from defs import *

def get_next_phrase(file):
    """
    Reads the file and yields every phrase as a list
    """
    phrase = []
    for line in file:
        if line == "\n":
            yield phrase
            phrase = []
            continue
        else:
            phrase.append(line.strip().replace(" ", "").split("|"))
    yield phrase

def separate_text_from_features_from_phrase(phrase):
    """
    Separates:
    - The text (returned as a string with every word separated by a space)
    - The features (all the features together)
    """
    text = " ".join([word[0] for word in phrase])
    features = [word[1:] for word in phrase]
    return text, features

def get_texts_and_features(file):
    """
    Returns the texts and features (with the same format as the previous function)
    """
    texts = []
    features = []
    for phrase in get_next_phrase(file):
        T, F = separate_text_from_features_from_phrase(phrase)
        texts.append(T)
        features.append(F)
    return texts, features

def translate_tokens_to_features(tok, features):
    """
    Rearranges the features from every word to every token.
    Note that a word could be separated in several tokens by the tokenizer.
    Also pads these features to match the length of the tokenizer
    """
    tokenized_features = []
    # For phrase in tokenizer
    for i in range(len(tok.input_ids)):
        features_tok_i = []
        # For token in phrase
        for j in range(len(tok.input_ids[i])):
            w = tok[i].token_to_word(j) # Get the word_index from the token j in the phrase i
            try:
                features_tok_i.append(features[i][w])
            except:
                features_tok_i.append(DEFAULT_TOKEN_FEATURE)
        tokenized_features.append(features_tok_i)
    return tokenized_features

def separate_each_feature(features):
    """
    Separates in the following way:
    [A1, B1, C1, D1], [A2, B2, C2, D2] -> [A1, A2], [B1, B2], [C1, C2], [D1, D2]
    """
    return [list(zip(*phrase)) for phrase in features] 

def map_all_features_to_numeric(features, mappings):
    """
    Maps all features with all dicts
    """
    feats = []
    for phrase in features:
        nmaps = len(mappings)
        aux = [list(map(m.get, f)) for f,m in zip(phrase[:nmaps], mappings)]
        for i in range(nmaps, len(phrase)):
            aux.append([int(n) for n in phrase[i]])
        feats.append(aux)
    return feats
    



