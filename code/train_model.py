import sys, getopt
from transformers import AutoTokenizer
import tensorflow as tf
from defs import *
from define_model import define_model, set_bert_to_trainable
from BIO_to_numeric import *


def train_model(ftrain, ftest, batch_size, epochs, train_bert_epochs, **kwargs):
    Xtrain, Ytrain = obtain_features(ftrain)
    Xtest, Ytest = obtain_features(ftest)

    model = define_model(**kwargs)

    history = model.fit(
        Xtrain,
        Ytrain, 
        batch_size=batch_size, 
        epochs=epochs
    )

    if train_bert_epochs > 0:
        model = set_bert_to_trainable(model)

        bert_history = model.fit(
            Xtrain,
            Ytrain, 
            batch_size=batch_size, 
            epochs=train_bert_epochs + epochs,
            initial_epoch = epochs
        )
        history = (history, bert_history)

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
