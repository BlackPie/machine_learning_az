Part 7: Natural language processing

Section 27
————————————
Natural Language Processing

Natural Language Processing is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages. NLP is used to apply ML models to text and language. Teach machines to understand what is said in spoken and written word is the focus of NLP. Whenever you dictate something to your phone that is then converted to a text, that’s an NLP algorithm in action.

NLP uses:
Sentiment analysis. Identifying the mood or subjective opinions within large amounts of text, including average sentiment and opinion minig.
Use it to predict the genre of the book.
Question answering
Use NLP to build a machine translator or a speech recognition system
Document summarization

We have an input tsv(tab separated values) file with two colums - Review which contains atext of a review and Liked which tells us if it is a positive or negative review.
The model we are going to implement is called Bag of Words. It doesn’t work with all words, so texts have to be cleaned. For example we have to delete all articles from there, change all tenses of verbs to the present tense and get rid of capitals.
Аfter that we start tokenization process, which will calculate number of times each word appears in a review.
When it is done our data is ready and the problem can be treated as a clssification one.
Usually people use Decision Tree or Naive Bayes models for NLP tasks.

```
TODO: put implementation here
```

TODO: homework?
```
Hello students,

congratulations for having completed Part 7 - Natural Language Processing.

If you are up for some practical activities, here is a little challenge:

1. Run the other classification models we made in Part 3 - Classification, other than the one we used in the last tutorial.

2. Evaluate the performance of each of these models. Try to beat the Accuracy obtained in the tutorial. But remember, Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall). Please find below these metrics formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

3. Try even other classification models that we haven't covered in Part 3 - Classification. Good ones for NLP include:

CART
C5.0
Maximum Entropy
Submit your results in the Q&A for this Lecture or by pm and justify in few words why you think it's the most appropriate model.

Enjoy Machine Learning!

Best to all,

Hadelin

```



