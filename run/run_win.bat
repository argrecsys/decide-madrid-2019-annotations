@echo on
cd "C:\Dev Projects\decide-madrid-2019-labels\prodigy-scripts"
call conda activate base
call conda activate prodigy_env
call python -m prodigy rel.manual intra_andres blank:es "C:/Dev Projects/decide-madrid-2019-labels/data/50.jsonl" --label LINKS,CONDITION,REASON,CONCLUSION,EXEMPLIFICATION,RESTATEMENT,SUMMARY,EXPLANATION,GOAL,RESULT,ALTERNATIVE,COMPARISON,CONCESSION,OPPOSITION,ADDITION,PRECISION,SIMILARITY,SUPPORT,ATTACK --span-label PREMISE,CLAIM,MAJOR_CLAIM,LINKER --wrap
