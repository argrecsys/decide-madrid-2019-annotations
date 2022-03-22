import sys, getopt
from transformers import AutoTokenizer
import tensorflow as tf
from defs import *
from define_model import define_model
from BIO_to_numeric import *

# TODO
def train_model(fin, batch_size, epochs):
    X,Y = obtain_features(fin)
    model = define_model()

    history = model.fit(X, Y, batch_size=batch_size, epochs=epochs)

    return model, history

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