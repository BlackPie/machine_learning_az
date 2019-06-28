# Part 5: Association Rule Learning

People who bought also bought ... That is what Association Rule Learning will help us figure out.

## Section 24: Apriori algorithm

That’s quite straightforward thing. We are going to define some values and compare them for all possible combinations.
Apriory algorithm has three parts: support, confidence and the lift. Let’s use movie recomendation system to explain these terms:

**Support** (M) = number of user watchlists containing M / total number of user watchlists.

**Confidence** (M1 -> M2) = number of user watchlists containing both M1 and M2 / number of user watchlists containing M1.

**Lift** (M1 -> M2) = Confidence(M1 -> M2) / Support(M2)

* Step 1. Set a minimum support and confidence
* Step 2. Take all the subsets in transactions having higher support than minimum support
* Step 3. Take all the rules of of these subsets having higher confidence than minimum confidence
* Step 4. Sort the rules by decreasing lift

Unfortunately the lectors didn’t use any library and implemented it by themselves so this code can be treated as a study example only.
In my opinion, source code of apyory.py is much more interesting that using of the function itself.

## Section 25: Eclat model
It’s just a basic version of apriory, so no one uses it.

This part is described in the course very poorly.

[Next Part >>>](6_reinforcement_learning.md)
