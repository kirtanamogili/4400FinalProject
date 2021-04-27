# CS4400 Final Project

## Entity Matching

You are given two tables (left table and right table) of electronic products. Each table is from a different shopping website. Each row in a table represents a product instance. For every pair of tuples (Li,Rj), where Li is a tuple in the left table and Rj is a tuple in the right table, it is either a match or a non-match. A pair of tuples is a match is they refer to the same real-world entity.
Three files are provided in data.zip: ltable.csv (the left table), rtable.csv (the right table), and train.csv (the training set). The training set contains a subset of tuple pairs, where some of them are matches and some of them are non-matches. The training set has three columns "ltable_id", "rtable_id", and "label". "label" being 1/0 denotes match/non-match.
The task is to find all remaining matching pairs like (Li,Rj) in the two tables, excluding those matches already found in the training set.

To run the solution:

1. install dependencies for the solution `pip install python-Levenshtein scikit-learn pandas numpy keras tensorflow`
2. run the solution `python solution.py`

Kirtana Mogili
