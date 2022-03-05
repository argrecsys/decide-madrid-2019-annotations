ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR := $(shell dirname $(ROOT_DIR))
INTRA_CONFIG := $(ROOT_DIR)/prodigy-scripts/.prodigy-intra-argument-config.json
CONFIG_OVERRIDE := '{"db_settings":{"sqlite":{"path":"'$(ROOT_DIR)/db'", "name":"'$(USER)'.db"}}}'


intra:
	PRODIGY_CONFIG_OVERRIDES=$(CONFIG_OVERRIDE) PRODIGY_CONFIG=$(INTRA_CONFIG) prodigy rel.manual intra_$(USER) blank:es $(ROOT_DIR)/data/raw_inputs.jsonl --label CONDITION,REASON,CONCLUSION,EXEMPLIFICATION,RESTATEMENT,SUMMARY,EXPLANATION,GOAL,RESULT,ALTERNATIVE,COMPARISON,CONCESSION,OPPOSITION,ADDITION,PRECISION,SIMILARITY,SUPPORT,ATTACK --span-label PREMISE,CLAIM,MAJOR_CLAIM --wrap

