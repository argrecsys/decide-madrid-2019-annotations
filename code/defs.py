BIO = {
	"O": 0,
	"B": 1,
	"I": 2
}

ARGUMENT_TYPES = {
	"NA": 0,
	"MAJOR_CLAIM": 1,
	"CLAIM": 2,
	"PREMISE": 3
}

ARGUMENT_TYPES_NAMES = list(ARGUMENT_TYPES.keys())

RELATION_TYPES = {
	"NA": 0,
	"CONDITION": 1,
	"REASON": 2,
	"CONCLUSION": 3,
	"EXEMPLIFICATION": 4,
	"RESTATEMENT": 5,
	"SUMMARY": 6,
	"EXPLANATION": 7,
	"GOAL": 8,
	"RESULT": 9,
	"ALTERNATIVE": 10,
	"COMPARISON": 11,
	"CONCESSION": 12,
	"OPPOSITION": 13,
	"ADDITION": 14,
	"PRECISION": 15,
	"SIMILARITY": 16
}

RELATION_TYPES_NAMES = list(RELATION_TYPES.keys())

RELATION_INTENTS = {
	"NA": 0,
	"SUPPORT": 1,
	"ATTACK": 2
}

RELATION_INTENTS_NAMES = list(RELATION_INTENTS.keys())

DEFAULTS = {
	"bio": "0",
	"argument_type": "NA",
	"relation_type": "NA",
	"relation_intent": "NA",
	"relation_distance": 0
}


