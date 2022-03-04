ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR := $(shell dirname $(ROOT_DIR))
PRODIGY_CONFIG := $(ROOT_DIR)/prodigy-scripts/.prodigy-intra-argument-config.json


intra:
	prodigy rel.manual intra_$(USER) blank:es $(ROOT_DIR)/data/raw_inputs.jsonl --label CONDITION,REASON,CONCLUSION,EXEMPLIFICATION,RESTATEMENT,SUMMARY,EXPLANATION,GOAL,RESULT,ALTERNATIVE,COMPARISON,CONCESSION,OPPOSITIONADDITION,PRECISION,SIMILARITY,SUPPORT,ATTACK --span-label PREMISE,CLAIM,MAJOR_CLAIM --wrap
