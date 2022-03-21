import sys, getopt
from transformers import AutoTokenizer
import tensorflow as tf
from defs import *
from define_model import define_model
from BIO_to_numeric import *

# TODO
def train_model(fin):
    with open(fin, "r") as f:
        texts, features = get_texts_and_features(f)
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    tok = tokenizer(texts, padding="max_length", truncation=True, return_tensors="tf")
    tokenized_features = separate_each_feature(translate_tokens_to_features(tok, features))
    numeric_features = map_all_features_to_numeric(tokenized_features, ALL_MAPPINGS)
    numeric_features = np.asarray(numeric_features).astype('float32')
    numeric_features = np.transpose(numeric_features, (0,2,1))
    num_feat_tensor = tf.convert_to_tensor(numeric_features)

    model.fit(tok.input_ids, num_feat_tensor)


def main(argv):
    script_name = argv[0]
    fin = None
    fout = sys.stdout
    try:
        opts, args = getopt.getopt(argv[1:],"hi:",["ifile="])
    except getopt.GetoptError:
        print(f"run with $ {script_name} -i <inputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(f"run with $ {script_name} -i <inputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            fin = arg

    if fin is None:
        print(f"run with $ {script_name} -i <inputfile> [-o <outputfile>]")
        sys.exit(2)

    train_model(fin)


if __name__ == "__main__":
    main(sys.argv)