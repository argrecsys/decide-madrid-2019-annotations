import numpy as np

def get_one_hot_vector(length, element):
    vec = np.zeros(length)
    vec[element] = 1
    return vec

BIO = {
	"O": get_one_hot_vector(3,0),
	"B": get_one_hot_vector(3,1),
	"I": get_one_hot_vector(3,2)
}

BIO_NAMES = list(BIO.keys())

ARGUMENT_TYPES = {
	"NA": get_one_hot_vector(5,0),
	"MAJOR_CLAIM": get_one_hot_vector(5,1),
	"CLAIM": get_one_hot_vector(5,2),
	"PREMISE": get_one_hot_vector(5,3),
    "LINKER": get_one_hot_vector(5,4),
}

ARGUMENT_TYPES_NAMES = list(ARGUMENT_TYPES.keys())

RELATION_TYPES = {
	"NA": get_one_hot_vector(18,0),
	"CONDITION": get_one_hot_vector(18,1),
	"REASON": get_one_hot_vector(18,2),
	"CONCLUSION": get_one_hot_vector(18,3),
	"EXEMPLIFICATION": get_one_hot_vector(18,4),
	"RESTATEMENT": get_one_hot_vector(18,5),
	"SUMMARY": get_one_hot_vector(18,6),
	"EXPLANATION": get_one_hot_vector(18,7),
	"GOAL": get_one_hot_vector(18,8),
	"RESULT": get_one_hot_vector(18,9),
	"ALTERNATIVE": get_one_hot_vector(18,10),
	"COMPARISON": get_one_hot_vector(18, 11),
	"CONCESSION": get_one_hot_vector(18,12),
	"OPPOSITION": get_one_hot_vector(18,13),
	"ADDITION": get_one_hot_vector(18,14),
	"PRECISION": get_one_hot_vector(18,15),
	"SIMILARITY": get_one_hot_vector(18,16),
    "LINKS": get_one_hot_vector(18,17)
}

RELATION_TYPES_NAMES = list(RELATION_TYPES.keys())

RELATION_INTENTS = {
	"NA": get_one_hot_vector(3,0),
	"SUPPORT": get_one_hot_vector(3,1),
	"ATTACK": get_one_hot_vector(3,2)
}

RELATION_INTENTS_NAMES = list(RELATION_INTENTS.keys())

DEFAULTS = {
	"bio": "O",
	"argument_type": "NA",
	"relation_type": "NA",
	"relation_intent": "NA",
	"relation_distance": 0
}

DEFAULT_TOKEN_FEATURE = [
	DEFAULTS["bio"],
	DEFAULTS["argument_type"],
	DEFAULTS["relation_type"],
	DEFAULTS["relation_intent"],
	DEFAULTS["relation_distance"]
]

ALL_MAPPINGS = [BIO, ARGUMENT_TYPES, RELATION_TYPES, RELATION_INTENTS]

## NN constants
TRANSFORMER_HIDDEN_STATES_SIZE = 4
COMMON_LAYER_SIZE = 1024
BIO_LAYER_SIZE = 128
AM_TYPE_LAYER_SIZE = 128
REL_TYPE_LAYER_SIZE = 128
REL_INTENT_LAYER_SIZE = 128
REL_DISTANCE_LAYER_SIZE = 128


