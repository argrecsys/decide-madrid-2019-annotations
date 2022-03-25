import sys, getopt, json
from defs import *

def extract_features_from_annotation(proposal):
    tokens = [token["text"] for token in proposal["tokens"]]
    sentence_length = len(tokens)

    # Returns initiated to defaults
    spans = [DEFAULTS["bio"]] * sentence_length
    span_types = [DEFAULTS["argument_type"]] * sentence_length
    relation_types = [DEFAULTS["relation_type"]] * sentence_length
    relation_intents = [DEFAULTS["relation_intent"]] * sentence_length
    relation_distances =  [DEFAULTS["relation_distance"]] * sentence_length

    # Used in the relation distance calculation
    span_end_tokens = {}
    span_id = 1

    # Set all spans
    for span in proposal["spans"]:
        token_start = span["token_start"]
        token_end = span["token_end"]
        span_end_tokens[token_end] = span_id
        span_id += 1
        span_type = span["label"]
        spans[token_start] = "B"
        span_types[token_start] = span_type
        for idx in range(token_start + 1, token_end + 1):
            spans[idx] = "I"
            span_types[idx] = span_type
    
    # Set all relations
    for relation in proposal["relations"]:
        head_token_start = relation["head_span"]["token_start"]
        head_token_end = relation["head_span"]["token_end"]
        distance = span_end_tokens[relation["child"]] - span_end_tokens[relation["head"]]
        relation_type = relation["label"]
        for idx in range(head_token_start, head_token_end + 1):
            if relation_type in RELATION_TYPES_NAMES:
                relation_types[idx] = relation_type
            if relation_type in RELATION_INTENTS_NAMES:
                relation_intents[idx] = relation_type
            relation_distances[idx] = distance

    return tokens, spans, span_types, relation_types, relation_intents, relation_distances

def write_features(fout, tokens, spans, span_types, relation_types, relation_intents, relation_distances):
    sentence_length = len(tokens)
    for idx in range(0, sentence_length):
        print(tokens[idx], spans[idx], span_types[idx], relation_types[idx], relation_intents[idx], relation_distances[idx], sep="| ", file=fout)
    # Empty line to indicate a change of text
    print(file=fout)

def from_jsonl_to_BIO(fin, fout=sys.stdout):
    with open(fin, "r") as json_file:
        json_list = list(json_file)

    if fout is not sys.stdout:
        fout = open(fout, "w")
    for json_str in json_list:
        proposal = json.loads(json_str)
        tokens, spans, span_types, relation_types, relation_intents, relation_distances = extract_features_from_annotation(proposal)
        write_features(fout, tokens, spans, span_types, relation_types, relation_intents, relation_distances)
    if fout is not sys.stdout:
        fout.close()


def main(argv):
    script_name = argv[0]
    fin = None
    fout = sys.stdout
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print(f"run with $ {script_name} -i <inputfile> [-o <outputfile>]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(f"run with $ {script_name} -i <inputfile> [-o <outputfile>]")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            fin = arg
        elif opt in ("-o", "--ofile"):
            fout = arg

    if fin is None:
        print(f"run with $ {script_name} -i <inputfile> [-o <outputfile>]")
        sys.exit(2)

    from_jsonl_to_BIO(fin, fout)


if __name__ == "__main__":
    main(sys.argv)