# Tesco-Journey
Analysis of Tesco household data of over two years from a group of 2,500 households that are frequent shoppers at a retailer.

The data source is https://www.dunnhumby.com/sourcefiles,
which gives insight into:
a. Household level transactions over two years from a group of 2,500 households who are frequent shoppers at a retailer.
b. All of a household’s purchases within the store, not just those from a limited number of categories.
c. Demographics and direct marketing contact history for select households.

I have used this data to create a classifier to predicts if a user will buy a product the next time he / she shops with the retailer.

1. Simple gradient boosting is used to understand the variability and data.
2. Variable importance is understood and substituted in subsequent models.
3. Seperated validation datasets and cleaned the data.
4. Due to skew in the dependent variable, ROSE package is used for a combination of over and under sampling to build a model.
5. Model is optimized for best results and validated using validation sets.


