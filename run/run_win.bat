@echo on
cd "C:\Dev Projects\decide-madrid-2019-labels\prodigy-scripts"
conda activate prodigy_env
python -m prodigy rel.manual intra_ivan blank:es "C:/Dev Projects/decide-madrid-2019-labels/data/879.jsonl" --label LINKS,CONDITION,REASON,CONCLUSION,EXEMPLIFICATION,RESTATEMENT,SUMMARY,EXPLANATION,GOAL,RESULT,ALTERNATIVE,COMPARISON,CONCESSION,OPPOSITION,ADDITION,PRECISION,SIMILARITY,SUPPORT,ATTACK --span-label PREMISE,CLAIM,MAJOR_CLAIM,LINKER --wrap
python -m prodigy db-out intra_ivan "C:/Dev Projects/decide-madrid-2019-labels/results/ivan-06032022-233112"
start chrome http://localhost:8080/
