import pandas as pd
import sys

def get_comment_tree(id, sub_comments):
    comment_comments = sub_comments.loc[sub_comments.parentId == id, :]
    return {id_comment: get_comment_tree(id_comment, sub_comments) for id_comment in comment_comments.id}

def get_proposal_tree(id, comments):
    proposal_comments = comments.loc[comments.proposalId == id, :]
    return {id: get_comment_tree(-1, proposal_comments)}

def print_comment_as_separate_texts(comments, comment_tree, file=sys.stdout):
    if len(comment_tree) == 0:
        return
    current_comments = comment_tree.keys()
    for comm in current_comments:
        print("{\"text\": \"" + str(comments.loc[comments.id == comm, "text"].values[0]) + "\", \"comment_id\": \"" + str(comm) + "\"}", file=file, sep='')
        print_comment_as_separate_texts(comments, comment_tree[comm], file=file)

def print_proposal_as_separate_texts(comments, proposals, proposal_id, file=sys.stdout):
    comment_tree = get_proposal_tree(proposal_id, comments)
    prop = proposals.loc[proposals.id == proposal_id]
    comments = comments.loc[comments.proposalId == proposal_id]
    title = str(prop.title.values[0])
    summary = str(prop.summary.values[0])
    text = str(prop.text.values[0])
    if len(title) > 0:
        print("{\"text\": \"" + title + "\", \"proposal_id\": \"" + str(proposal_id) + "\", \"info\": \"title\"}", file=file, sep='')
    if len(summary) > 0:
        print("{\"text\": \"" + summary + "\", \"proposal_id\": \"" + str(proposal_id) + "\", \"info\": \"summary\"}", file=file, sep='')
    if len(text) > 0:
        print("{\"text\": \"" + text + "\", \"proposal_id\": \"" + str(proposal_id) + "\", \"info\": \"text\"}", file=file, sep='')
    print_comment_as_separate_texts(comments, comment_tree[proposal_id], file=file)

def generate_json_separate_texts(proposals, comments, file_name):
    with open(file_name, "w") as f:
        for proposal in proposals.id.unique():
            print_proposal_as_separate_texts(comments, proposals, proposal, file=f)

proposals = pd.read_csv("../data/proposals.csv")
comments = pd.read_csv("../data/comments.csv")
generate_json_separate_texts(proposals, comments, "../data/raw_inputs.jsonl")