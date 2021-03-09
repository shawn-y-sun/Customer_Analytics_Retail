# Customer Analytics for Retail/FMCG Company

## Project Overview
This project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate suitable marketing strategies that could further maximize revenues on Candy Bars. To reach the fullest potential of bringing up revenues, a company should find the 'sweet spot' on the relationship between price and quantity. 

Conducting analysis on price elasicity would support us to find the optimal point. We will look at price elasticities of three aspects: purchase proability, brand choice probability, and purchase quantity. By doing this, we can construct strategies to increase the likelihood of a customer purchasing our products on all shopping stages. To better position our products, we will firstly perform segmentation on our customers to support our analysis on price elasiticity, allowing us to customize marketing strategies for customers with different backrgounds.


## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Dataset Source__: https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases

## Datasets Information
[_**segmentation data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=segmentation+data.csv) contains data of our customers that we use to build model for segmentation.<br>
[_**'purchase data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=purchase+data.csv) contains data of each purchase transaction of customers, including price, quantity, brand, incidence.



## [1. Segmentation](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/1.Customer%20Analytics%20-%20Customer%20Segmentation.ipynb)

### 1.1 Exploratory Analysis

#### Dataset overview

|                  |     Sex    |     Marital status    |     Age    |     Education    |     Income    |     Occupation    |     Settlement size    |
|------------------|------------|-----------------------|------------|------------------|---------------|-------------------|------------------------|
|     ID           |            |                       |            |                  |               |                   |                        |
|     100000001    |     0      |     0                 |     67     |     2            |     124670    |     1             |     2                  |
|     100000002    |     1      |     1                 |     22     |     1            |     150773    |     1             |     2                  |
|     100000003    |     0      |     0                 |     49     |     1            |     89210     |     0             |     0                  |
|     100000004    |     0      |     0                 |     45     |     1            |     171565    |     1             |     1                  |
|     100000005    |     0      |     0                 |     53     |     1            |     149031    |     1             |     1                  |

Notes:
- Sex: 0 - male, 1 - female
- Marital status: 0 - single, 1-non-single
- Education: 0 - other/unknown, 1 - high school, 2 - university, 3 - graduate school
- Occupation: 0 - unemployed, 1 - skilled, 2 - highly qualified
- Settlement size: 0 - small, 1 - mid sized, 2 - big


#### Correlation estimate
![image](https://user-images.githubusercontent.com/77659538/110475812-487ab200-811c-11eb-9e8a-4bd503838d54.png)

🔶 Insights: we can spot some level of correlations between certain pairs of variables, such as Income vs Occupation, Education vs Age, Settlement Size vs Occupation. It indicates that we can reduce the dimensions of portraying our customers without losing too much information, allowing us to segmenting our customers more accurately.


### 1.2 Clustering
#### Standardization
Before everything, we standardize our data, so that all features have equal weight
```
# Standardizing data, so that all features have equal weight
scaler = StandardScaler() #Create an instance
segmentation_std = scaler.fit_transform(df_segmentation) #Apply the fit transformation
```

#### K-Means Clustering
First, we perform K-means clustering, considering 1 to 10 clusters, and visisulize the Within Cluster Sum of Square (WCSS)

![image](https://user-images.githubusercontent.com/77659538/110477934-b922ce00-811e-11eb-9149-9b7ece615965.png)

Using 'Elbow method', we choose 4 clusters to segment our customers and get the following characteristics for each group<br>
|                            |     Sex         |     Marital status    |     Age          |     Education    |     Income           |     Occupation    |     Settlement size    |     N Obs    |     Prop Obs    |
|----------------------------|-----------------|-----------------------|------------------|------------------|----------------------|-------------------|------------------------|--------------|-----------------|
|     Segment K-means        |                 |                       |                  |                  |                      |                   |                        |              |                 |
|     well-off               |     0.501901    |     0.692015          |     55.703422    |     2.129278     |     158338.422053    |     1.129278      |     1.110266           |     263      |     0.1315      |
|     fewer-opportunities    |     0.352814    |     0.019481          |     35.577922    |     0.746753     |     97859.852814     |     0.329004      |     0.043290           |     462      |     0.2310      |
|     standard               |     0.029825    |     0.173684          |     35.635088    |     0.733333     |     141218.249123    |     1.271930      |     1.522807           |     570      |     0.2850      |
|     career focused         |     0.853901    |     0.997163          |     28.963121    |     1.068085     |     105759.119149    |     0.634043      |     0.422695           |     705      |     0.3525      |

🔶 Insights: we have 4 segments of customers
- Well-off: senior-aged, highly-educated, high income,
- Fewer-opportunities: single, middle-aged, low income, low-level occupation, small living size
- Career-focused: non-single, young, educated
- Standard: others

However, if we have choose 2 dimensions to visualize the segmentation, it's hard to identify the groups.
![image](https://user-images.githubusercontent.com/77659538/110480458-79a9b100-8121-11eb-9829-16a71211f9d4.png)

Therefore, we need to perform the clustering with PCA

#### PCA
After fitting the PCA with our standardized data, we visualize the explained variance
![image](https://user-images.githubusercontent.com/77659538/110480893-f50b6280-8121-11eb-9acd-2c062ee886f3.png)

We choose 3 components to represent our data, with over 80% variance explained.<br>
After fitting our data with the selected number of compoenents, we get the loadings (i.e. correlations) of each component on each of the seven original features

|                    |     Sex          |     Marital status    |     Age         |     Education    |     Income       |     Occupation    |     Settlement size    |
|--------------------|------------------|-----------------------|-----------------|------------------|------------------|-------------------|------------------------|
|     Component 1    |     -0.314695    |     -0.191704         |     0.326100    |     0.156841     |     0.524525     |     0.492059      |     0.464789           |
|     Component 2    |     0.458006     |     0.512635          |     0.312208    |     0.639807     |     0.124683     |     0.014658      |     -0.069632          |
|     Component 3    |     -0.293013    |     -0.441977         |     0.609544    |     0.275605     |     -0.165662    |     -0.395505     |     -0.295685          |

Visualize the loadings by heatmap<br>
![image](https://user-images.githubusercontent.com/77659538/110481794-e83b3e80-8122-11eb-9438-02b1742b8e84.png)

🔶 Insights: each component shows a dimension of individual features
- Component 1: represents the career focusness by relating to income, occupation, and settlement size
- Component 2: represents the individual education and life style by relating to sex, marital status, and education
- Component 3: represents the level of experience (work&life) by relating to marital status, age, and occupation

#### K-Means Clustering with PCA
We fit K means using the transformed data from the PCA, and get the WCSS below<br>
![image](https://user-images.githubusercontent.com/77659538/110483187-706e1380-8124-11eb-86a2-febcb80f8096.png)

Again, we choose 4 clusters to fit our data, and get the below results<br>
|                            |     Sex         |     Marital status    |     Age          |     Education    |     Income           |     Occupation    |     Settlement size    |     Component 1    |     Component 2    |     Component 3    |
|----------------------------|-----------------|-----------------------|------------------|------------------|----------------------|-------------------|------------------------|--------------------|--------------------|--------------------|
|     Segment K-means PCA    |                 |                       |                  |                  |                      |                   |                        |                    |                    |                    |
|     standard                      |     0.307190    |     0.098039          |     35.383442    |     0.766885     |     93566.102397     |     0.248366      |     0.039216           |     -1.048838      |     -0.892116      |     1.010446       |
|     career focused                      |     0.027350    |     0.167521          |     35.700855    |     0.731624     |     141489.721368    |     1.266667      |     1.475214           |     1.367167       |     -1.050209      |     -0.247981      |
|     fewer opportunities                      |     0.900433    |     0.965368          |     28.913420    |     1.062049     |     107551.946609    |     0.676768      |     0.440115           |     -1.106918      |     0.706367       |     -0.778269      |
|     well-off                      |     0.505703    |     0.688213          |     55.722433    |     2.129278     |     158391.676806    |     1.129278      |     1.110266           |     1.706153       |     2.031716       |     0.838839       |


We plot data by 2 PCA components: Y axis - component 1, X axis - component 2<br>
![image](https://user-images.githubusercontent.com/77659538/110485951-220e4400-8127-11eb-95df-af5173713103.png)

We can clearly identify 4 clusters!

## [2. Purchase Descriptive Analytics](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/2.%20Customer%20Analytics%20-%20Purchase%20Descriptive%20Analysis.ipynb)

### 2.1 Data Segmentation
We implement the standardization, PCA, and K-means clustering models from previous part, to segment our customers in purchase dataset. We have the following

|          |     ID           |     Day    |     Incidence    |     Brand    |     Quantity    |     Last_Inc_Brand    |     Last_Inc_Quantity    |     Price_1    |     Price_2    |     Price_3    |     ...    |     Promotion_4    |     Promotion_5    |     Sex    |     Marital status    |     Age    |     Education    |     Income    |     Occupation    |     Settlement size    |     Segment    |
|----------|------------------|------------|------------------|--------------|-----------------|-----------------------|--------------------------|----------------|----------------|----------------|------------|--------------------|--------------------|------------|-----------------------|------------|------------------|---------------|-------------------|------------------------|----------------|
|     0    |     200000001    |     1      |     0            |     0        |     0           |     0                 |     0                    |     1.59       |     1.87       |     2.01       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     1    |     200000001    |     11     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     2    |     200000001    |     12     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     3    |     200000001    |     16     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.98       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     4    |     200000001    |     18     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |

We visualize the segments<br>
![image](https://user-images.githubusercontent.com/77659538/110487824-e8d6d380-8128-11eb-906b-12674c5f0c3b.png)

### 2.2 Purchase Occasion and Purchase Incidence
Plot the average number of store visits for each of the four segments using a bar chart, and display the standard deviation as a straight line

![image](https://user-images.githubusercontent.com/77659538/110488070-1face980-8129-11eb-8e68-d3b46de7cc3a.png)

🔶 Insights:
- The standard deviation amongst customers from the second segment is quite high. This implies that the customers in this segment are at least homogenous that is least alike when it comes to how often they visit the grocery store
- The standard fewer opportunities and well-off clusters are very similar in terms of their average store purchases. This is welcome information because it would make them more comparable with respect to our future analysis!

Display the average number of purchases by segments, help us understand how often each group buys chocholate candy bars<br>
![image](https://user-images.githubusercontent.com/77659538/110488334-5a168680-8129-11eb-865c-b6c1184ca63a.png)

🔶 Insights:
- For Career-focused, standard deviation is the highest it might be that a part of the segment buys products very frequently.And another part less so. Although consumers in this segment have a somewhat similar income, the way that they might want to spend their money might differ.
- The most homogenous segment appears to be that of the fewer opportunities. This is signified by the segment having the lowest standard deviation or shortest vertical line The first segment seems consistent as well with about 25 average purchases and a standard deviation of 30.

### Brand Choice
First, we select only rows where incidence is one. Then we make dummies for each of the 5 brands.<br>
|              |     Brand_1    |     Brand_2    |     Brand_3    |     Brand_4    |     Brand_5    |     Segment    |     ID           |
|--------------|----------------|----------------|----------------|----------------|----------------|----------------|------------------|
|     6        |     0          |     1          |     0          |     0          |     0          |     0          |     200000001    |
|     11       |     0          |     0          |     0          |     0          |     1          |     0          |     200000001    |
|     19       |     1          |     0          |     0          |     0          |     0          |     0          |     200000001    |
|     24       |     0          |     0          |     0          |     1          |     0          |     0          |     200000001    |
|     29       |     0          |     1          |     0          |     0          |     0          |     0          |     200000001    |
|     ...      |     ...        |     ...        |     ...        |     ...        |     ...        |     ...        |     ...          |
|     58621    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58648    |     1          |     0          |     0          |     0          |     0          |     0          |     200000500    |
|     58674    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58687    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |
|     58691    |     0          |     1          |     0          |     0          |     0          |     0          |     200000500    |


Visualize the brand choice by segments<br>
![image](https://user-images.githubusercontent.com/77659538/110488884-df01a000-8129-11eb-82f9-ffdad58c91ea.png)

🔶 Insights:
- Well-off and Career-focused prefer pricy brands
- Fewer-opportunities and standard prefer low price products

### 2.3 Revenue
Compute the total revenue for each of the segments. <br>
|                            |     Revenue Brand 1    |     Revenue Brand 2    |     Revenue Brand 3    |     Revenue Brand 4    |     Revenue Brand 5    |     Total Revenue    |     Segment Proportions    |
|----------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|----------------------|----------------------------|
|     Segment                |                        |                        |                        |                        |                        |                      |                            |
|     Standard               |     2258.90            |     13909.78           |     722.06             |     1805.59            |     2214.82            |     20911.15         |     0.378                  |
|     Career-Focused         |     736.09             |     1791.78            |     664.75             |     2363.84            |     19456.74           |     25013.20         |     0.222                  |
|     Fewer-Opportunities    |     2611.19            |     4768.52            |     3909.17            |     861.38             |     2439.75            |     14590.01         |     0.206                  |
|     Well-Off               |     699.47             |     1298.23            |     725.54             |     14009.29           |     5509.69            |     22242.22         |     0.194                  |

🔶 Insights:
- Career-focused is the most prominent segment
- Standard contributes the least
- If brand 3 reduces its price, the Standard segment could pivot towards it
- For well-off segment, Brand 4 could increase its price


## [3. Purchase Predictive Analytics](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/3.%20Customer%20Analytics%20-%20Purchase%20Predictive%20Analytics.ipynb)

### 3.1 Purchase Probability

#### Model Building
We implement the standardization, PCA, and K-means clustering models from part 1, to segment our customers in purchase dataset.<br>
```
# Y is Incidence, as we want to predict the purchase probability for our customers
Y = df_pa['Incidence']
```
```
# Dependant variable is based on the average price of chocolate candy bars. 
# X is a data frame, containing the mean across the five prices.
X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] +
                   df_pa['Price_2'] +
                   df_pa['Price_3'] +
                   df_pa['Price_4'] +
                   df_pa['Price_5'] ) / 5
X.head()
```
Since dependent variable has 2 outcomes, we choose logistic regression
```
# Create a Logistic Regression model
# Fit the model with our X or price and our Y or incidence
model_purchase = LogisticRegression(solver = 'sag')
model_purchase.fit(X, Y)
```

#### Price Elasticity of Purchase Probability
Since the prices of all 5 brands ranges with from 1.1 to 2.8. We will perform analysis on a slightly wider price range: 0.5 - 3.5<br>
Then we fit our 'test price range' in our model to get the corresponding Purchase Probability.<br>
Next, we apply below formula to derive the price elasticity at each price point<br>
```
# Elasticity = beta*price*(1-P(purchase))
pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_pr)
```
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110493304-921fc880-812d-11eb-834d-4094585e315e.png)

🔶 Insights:
- With prices lower than 1.25, we can increase our product price without losing too much in terms of purchase probability.
- For prices higher than 1.25, We have more to gain by reducing our prices.


#### Purchase Probability by Segments
![image](https://user-images.githubusercontent.com/77659538/110494029-2853ee80-812e-11eb-9504-a992392b0349.png)

🔶 Insights:
- The career-focused segment are the least elastic when compared to the rest. So, their purchase probability elasticity is not as affected by price.
- The price elasticities for the Standard segment seem to differ across price range. This may be due to the fact that the standard segment is least homogenous, which we discovered during our descriptive analysis.
- It may be that the customers in this segment have different shopping habbits, which is why their customers start with being more elastic than average but then shift to being more inelastic than the average customer and indeed the Career-focused segment.

#### Purchase Probability with and without Promotion Feature
![image](https://user-images.githubusercontent.com/77659538/110494242-5e916e00-812e-11eb-8c99-f7568c7bd4ab.png)

🔶 Insights:
- The purchase probability elasticity of the customer is less elastic when there is promotion
- This is an important insight for marketers, as according to our model people are more likely to buy a product if there is some promotional activity rather than purchase a product with the same price, when it isn't on promotion.

### 3.2 Brand Choice Probability
#### Model Building
```
# Set the dependent variable
Y = brand_choice['Brand']
```
```
# Predict based on the prices for the five brands.
features = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price_5']
X = brand_choice[features]
```
We again choose logistic regression
```
# Brand Choice Model fit.
model_brand_choice = LogisticRegression(solver = 'sag', 
                                        multi_class = 'multinomial')
model_brand_choice.fit(X, Y)
```
We get the following coefficients:
|                |     Coef_Brand_1    |     Coef_Brand_2    |     Coef_Brand_3    |     Coef_Brand_4    |     Coef_Brand_5    |
|----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
|     Price_1    |     -3.92           |     1.27            |     1.62            |     0.57            |     0.44            |
|     Price_2    |     0.66            |     -1.88           |     0.56            |     0.40            |     0.26            |
|     Price_3    |     2.42            |     -0.21           |     0.50            |     -1.40           |     -1.31           |
|     Price_4    |     0.70            |     -0.21           |     1.04            |     -1.25           |     -0.29           |
|     Price_5    |     -0.20           |     0.59            |     0.45            |     0.25            |     -1.09           |


#### Own Price Elasticity Brand 5
We fit our model and calcuate the price elasticity for brand 5 at the same 'test price range'<br>
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110494977-0c048180-812f-11eb-8b0e-699f82603802.png)


#### Cross Price Elasticity Brand 5, Cross Brand 4
To calculate the cross brand price elasticity, we use new formula
```
brand5_cross_brand4_price_elasticity = -beta5 * price_range * pr_brand_4
```

We visualize the the cross price elasticity of purchase probability for brand 5 vs brand 4<br>
![image](https://user-images.githubusercontent.com/77659538/110495271-5423a400-812f-11eb-9c59-4023c515089d.png)

🔶 Insights:
- We observe they are positive. As the price of the competitor brand increases, so does the probability for purchasing our own brand.
- Even though the elasticity starts to decrease from the 1.45 mark, it is still positive, signalling that the increase in purchase probability for the own brand happens more slowly.

#### Own and Cross-Price Elasticity by Segment
![image](https://user-images.githubusercontent.com/77659538/110495416-74536300-812f-11eb-9be2-188a45164262.png)

🔶 Insights: The two segments, which seem to be of most interested for the marketing team of brand 5, seem to be the 'Career-focused' and the 'Well-off'. They are also the segments which purchase this brand most often.
- The Career-focused segment is the most inelastic and they are the most loyal segment.
 - Based on our model, they do not seem to be that affected by price, therefore brand 5 could increase its price, without fear of significant loss of customers from this segment.
- The Well-off segment on the other hand, seems to be more elastic. They also purchase the competitor brand 4 most often.
 - In order to target this segment, our analysis signals, that price needs to be decreased. However, keep in mind that other factors aside from price might be influencing the purchase behaivour of this segment.

### 3.3 Purchase Quantity

#### Model Estimation

To determine price elasticity of purchase quantity, also known as price elasticity of demand, we're interested in purchase ocassion, where the purchased quantity is different from 0.
```
# Filter our data
df_purchase_quantity = df_pa[df_pa['Incidence'] == 1]
```

Independent variable: price, promotion
```
X = df_purchase_quantity[['Price_Incidence', 'Promotion_Incidence']]
X
```
|              |     Price_Incidence    |     Promotion_Incidence    |
|--------------|------------------------|----------------------------|
|     6        |     1.90               |     0                      |
|     11       |     2.62               |     1                      |
|     19       |     1.47               |     0                      |
|     24       |     2.16               |     0                      |
|     29       |     1.88               |     0                      |
|     ...      |     ...                |     ...                    |
|     58621    |     1.89               |     0                      |
|     58648    |     1.35               |     1                      |
|     58674    |     1.85               |     1                      |
|     58687    |     1.51               |     0                      |
|     58691    |     1.82               |     0                      |

Dependent variable: quantity
```
Y = df_purchase_quantity['Quantity']
```
We choose linear regression to fit the model
```
model_quantity = LinearRegression()
model_quantity.fit(X, Y)
```

```
In [110]:
model_quantity.coef_

Out[110]:
array([-0.8173651 , -0.10504673])
```
It appears that promotion reflects negatively on the purchase quantity of the average client, which is unexpected.

#### Price Elasticity of Purchase Quantity with and without Promotion
Calculate the price elasticity with new formula
```
price_elasticity_quantity_promotion_yes = beta_quantity * price_range / predict_quantity
```
Plot the two elasticities (with and without promotion) side by side<br>
![image](https://user-images.githubusercontent.com/77659538/110496501-7d90ff80-8130-11eb-80ff-976295954d42.png)

🔶 Insights:
- We observe that the two elasticities are very close together for almost the entire price range.
- It appears that promotion does not appear to be a significant factor in the customers' decission what quantity of chocolate candy bars to purchase.
