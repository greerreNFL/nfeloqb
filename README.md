# nfeloqb

## About
nfeloqb is a model for valuing and predicting NFL QB performance based on [538's QB Elo model](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/).

Since 538's model may be discontinued heading in to the 2023 season, the primary goal of this project is to recreate and maintain a CSV of weekly data in the same format as [538's original model](https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv). To start, this file will only cover the QB portion of the model. For those looking for Elo based power-rankings, the main [nfelo model](https://www.nfeloapp.com/nfl-power-ratings/) will likely be your best replacement.

As a secondary goal, nfeloqb will also seek to make improvements to the original 538 QB model where possible. Improvement here, means lower error in predicting 538's VALUE metric. The VALUE metric itself (which is a measure of QB performance somewhere between Passer Rating and QBR) will remain untouched.

The current improvements are as follows:
1) Seasonal Regression -- instead of regressing QBs to a league average, nfeloqb gradually phases out league average for a rolling career average, allowing for stronger season-over-season mean reversion
2) Initial Value -- like the original model, nfeloqb bases a QB's starting value on their draft position. However, in nfeloqb, this value decays exponentially rather than linearly and is applied as a value relative to the team's previous performance in the passing game rather than as an absolute value. This better accounts for passing performance inflation and introduces the notion that a QB's performance, especially early in their career, can be heavily influenced by surrounding talent and circumstance
3) Defensive Adjustment -- the original 538 methodology did not specify how opposing defensive adjustments were created. In nfeloqb, a team's rolling offensive and defensive passing performance are calculated by two seperate models with their own parameters. Offensive team values are optimized to match the original model's weekly QB adjustments as closely as possible, while the defensive model is optimized to minimize error in predicting how well a QB will perform against a given defense.

In the spirit of this package's primary goal, the accompanying CSV uses 538's original model for all seasons up until 2023. From 2023 onward, it uses the improved nfeloqb model

## Package Structure
For those looking to simply ingest the model output, the "qb_elos.csv" file will be updated every Tuesday and Thursday morning during the NFL season. For those looking to run the model themselves, this can be done by importing the pacakge and executing "nfeloqb.run()", which will pull all necessary data from nflfastR, calculate VALUES, determine QBs, etc. To run the model by itself for all games that have been played, use the argument "model_only=True" which will return a QBModel class instance that contains a an array of records (QBModel.data) with a record for each weekly QB performance.
