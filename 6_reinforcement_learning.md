Part 6: Reinforcement Learning

Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to solve interacting problems where the data observed up to time t is considered to decide which action to take at time t + 1. It is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.

# TODO: what is distribution

Section 27
————————————
Upper Confidence Bound

The Multi Armed Bandit Problem s a problem in which a limited set of resources must be allocated between competing choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. For example it solves a problem when you enter a casino and want to decide what slot machine to use. The longer you observe the more data you have to find the optimal slot machine, but if you start using it too late, then you can regret it.
One another example is ad campaign. Imagine you have 5 options for an ad and you have to pick only one. You can run multiple AB tests, but if you do them too much, then you spend a lot of time and money. So maybe in some options it is more profitable to pick not the best solution but one which is close to it and start using it earlier, so you will save money on AB tests and start earning extra money because of the new ad.

Here is explanation of the problem using ad campaign example:
We have d arms. For example, arms are ads that we display to users each time they connect to a web page.
Each time a user connects to this web page, that maks a round
At each round n, we choose one ad to display to the user
At each round n, ad i gives reward r(n) = 0 or 1, 1 if the user clicked on the ad and 0 if the user didn’t
Our goal is to maximize the total reward we get over many rounds

Upper Confidence Bound Algorithm
Step 1. At each round n we consider two numbers for each ad i :
            N(n)  —  the number of times the ad i was selected up to round n
            R(n)  —  the sum of rewards of the ad i up to round n
Step 2. From these numbers we compute:
            The average reward of ad i up to round n:   r(n) = R(n) / N(n)
            The confidence interval at round n with ...
Step 3. We select the ad i that has the maximum UCB (sum of average reward and confidence interval)
So we start at the same confidence level for every option, choose an option to use and then increase or decrease confidence level depending on if the action we wanted to happen occured or it didnt. And in any case we decrease confidence interval of the option we used because we get more and more observations and therefore we more sure in the confidence level we got.
http://prntscr.com/o4uj52
# TODO: can be useful for formatting: http://prntscr.com/o4u0yj

```
# TODO: put the code here
```

Section 27
————————————
Thompson Sampling

It is another solution for the Multi Armed Bandit Problem.

Thompson Sampling Algorithm:
Step 1. At each round n, we consider two numbers for each ad i:
            N1  -  the number of times the ad i got reward 1 up to round n
            N0  -  the number of times the ad i got reward 0 up to round n
Step 2. For each ad i, we take a random draw from the distribution below: take formula from there: http://prntscr.com/o4wfsq
Step 3. We select the ad that has the highest value

There is a destribution behind every slot machine. We pull an arm and construct a distribution several times for every machine. We don’t try to to predict distribution of a slot machine, but we know that the peak of the real distribution wil be somewhere in just created distribution. We take a random point from new distributions for every machine and check their returns. We pick a michine with the highest return, pull its arm, update distributions and the iteration is done.
Over the time new distribution will become higher and more narrow.

# TODO: learn more about it

UCB vs Thompson Sampling
http://prntscr.com/o4xjco
Thompson can be updated by batches and therefore it is much more computational friendly: http://prntscr.com/o4xewg

random result: 1126
UCB result: 2178
Thompson: ~2600

implementation
```
# TODO: put the code here
```
