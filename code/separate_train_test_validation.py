import sys, getopt, random
from os.path import join as path_join

def separate(fin, dirout, train, test, validation):
    assert train + test + validation == 100, "The sum of the elements of the proportion must be 100"

    fouttrain = path_join(dirout, "train.jsonl")
    fouttest = path_join(dirout, "test.jsonl")
    foutvalidation = path_join(dirout, "validation.jsonl")

    with open(fin, "r") as json_file:
        json_list = list(json_file)

    random.shuffle(json_list)

    total = len(json_list)

    trainp = round(total*train/100)
    testp = round(total*test/100)
    validationp = round(total*validation/100)

    i = 0
    if trainp > 0:
        with open(fouttrain, "w") as fout:
            for i in range(0, trainp):
                print(json_list[i].rstrip(), file=fout)

    if testp > 0:
        with open(fouttest, "w") as fout:
            for i in range(i, i+testp):
                print(json_list[i].rstrip(), file=fout)

    if validationp > 0:
        with open(foutvalidation, "w") as fout:
            for i in range(i, i+validationp):
                print(json_list[i].rstrip(), file=fout)


def main(argv):
    script_name = argv[0]
    fin = None
    dirout = None
    proportion = None
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:p:",["ifile=","odir=", "proportion="])
    except getopt.GetoptError:
        print(f"run with $ {script_name} -i <inputfile> -o <outputdir> -p <train:test:validation>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(f"run with $ {script_name} -i <inputfile> -o <outputdir> -p <train:test:validation>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            fin = arg
        elif opt in ("-o", "--odir"):
            dirout = arg
        elif opt in ("-p", "--proportion"):
            proportion = arg

    if fin is None or dirout is None or proportion is None:
        print(f"run with $ {script_name} -i <inputfile> -o <outputdir> -p <train:test:validation>")
        sys.exit(2)

    print(fin, dirout, proportion)
    separate(fin, dirout, *[int(value) for value in proportion.split(":")])


if __name__ == "__main__":
    main(sys.argv)