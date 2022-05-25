import sys, getopt
from transformers import AutoTokenizer
import tensorflow as tf
from defs import *
from define_model import define_model
from BIO_to_numeric import *

# TODO
def train_model(ftrain, ftest, fvalidation, batch_size, epochs, **kwargs):
    Xtrain, Ytrain = obtain_features(ftrain)
    Xtest, Ytest = obtain_features(ftest)
    Xvalidation, Yvalidation = obtain_features(fvalidation)
    model = define_model(**kwargs)

    history = model.fit(
        Xtrain,
        Ytrain, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(Xvalidation, Yvalidation),
        validation_freq=1
    )

    return model, history

def main(argv):
    script_name = argv[0]
    trainfile = None
    testfile = None
    validationfile = None
    batchsize = None
    epochs = None
    error_string = f"run with $ {script_name} -r <trainfile> -s <testfile> -v <validationfile>"
    try:
        opts, args = getopt.getopt(
            argv[1:],
            "hr:s:v:b:e",
            ["trainfile=", "testfile=", "validationfile=", "batchsize=", "epochs="]
        )
    except getopt.GetoptError:
        print(error_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(error_string)
            sys.exit()
        elif opt in ("-r", "--trainfile"):
            trainfile = arg
        elif opt in ("-s", "--testfile"):
            testfile = arg
        elif opt in ("-v", "--validationfile"):
            validationfile = arg
        elif opt in ("-b", "--batchsize"):
            validationfile = arg
        elif opt in ("-e", "--epochs"):
            validationfile = arg

    if trainfile is None or testfile is None or validationfile is None or batchsize is None or epochs is None:
        print(error_string)
        sys.exit(2)

    train_model(trainfile, testfile, validationfile, batchsize, epochs)


if __name__ == "__main__":
    main(sys.argv)
