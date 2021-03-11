# Customer Analytics for Retail/FMCG Company

## Project Overview
This project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate marketing and pricing strategies that could maximize revenues on each brand of candy bars. To reach the fullest potential of bringing up revenues, a company should find the 'sweet spot' for price to maximize three customer behaviours: purchase probability, brand choice probability, and purchase quantity. 

Data from customer purchase history were used for training the regression models to predict those three customer behaviours in a preconceived price range. The results were then converted into price elasticities so that we can examine the effects of changing price on each of the behaviours. Hence, we will be able to find the suitable marketing and pricing strategies.

To better position our products, we will firstly perform segmentation on our customers to support our analysis on customer behaviours, allowing us to customize marketing strategies for customers with different backgrounds.


## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Algorithms__: clustering(K-means, PCA), regression(logistic, linear)
* __Dataset Source__: https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases

## Datasets Information
[_**'segmentation data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=segmentation+data.csv) contains data of our customers that we use to build model for segmentation.<br>
[_**'purchase data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=purchase+data.csv) contains data of each purchase transaction of customers, including price, quantity, brand, incidence.



## [1. Segmentation](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/1.Customer%20Analytics%20-%20Customer%20Segmentation.ipynb)

In this part, we will segment our customers by grouping them in different clusters based on 7 different features. It will allow us to analyze purchase data by groups and customize marketing strategy for each of them.

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
First, we perform K-means clustering, considering 1 to 10 clusters, and visualize the Within Cluster Sum of Square (WCSS)

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
- Well-off: senior-aged, highly-educated, high income
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
After fitting our data with the selected number of components, we get the loadings (i.e. correlations) of each component on each of the seven original features

|                    |     Sex          |     Marital status    |     Age         |     Education    |     Income       |     Occupation    |     Settlement size    |
|--------------------|------------------|-----------------------|-----------------|------------------|------------------|-------------------|------------------------|
|     Component 1    |     -0.314695    |     -0.191704         |     0.326100    |     0.156841     |     0.524525     |     0.492059      |     0.464789           |
|     Component 2    |     0.458006     |     0.512635          |     0.312208    |     0.639807     |     0.124683     |     0.014658      |     -0.069632          |
|     Component 3    |     -0.293013    |     -0.441977         |     0.609544    |     0.275605     |     -0.165662    |     -0.395505     |     -0.295685          |

Visualize the loadings by heatmap<br>
![image](https://user-images.githubusercontent.com/77659538/110481794-e83b3e80-8122-11eb-9438-02b1742b8e84.png)

🔶 Insights: each component shows a dimension of individual features
- Component 1: represents the career focuses by relating to income, occupation, and settlement size
- Component 2: represents the individual education and lifestyle by relating to gender, marital status, and education
- Component 3: represents the level of experience (work&life) by relating to marital status, age, and occupation

#### K-Means Clustering with PCA
We fit K means using the transformed data from the PCA, and get the WCSS below<br>
![image](https://user-images.githubusercontent.com/77659538/110483187-706e1380-8124-11eb-86a2-febcb80f8096.png)

Again, we choose 4 clusters to fit our data, and get the below results<br>
|                            |     Sex         |     Marital status    |     Age          |     Education    |     Income           |     Occupation    |     Settlement size    |     Component 1    |     Component 2    |     Component 3    |
|----------------------------|-----------------|-----------------------|------------------|------------------|----------------------|-------------------|------------------------|--------------------|--------------------|--------------------|
|     Segment K-means PCA    |                 |                       |                  |                  |                      |                   |                        |                    |                    |                    |
|     fewer opportunities                      |     0.307190    |     0.098039          |     35.383442    |     0.766885     |     93566.102397     |     0.248366      |     0.039216           |     -1.048838      |     -0.892116      |     1.010446       |
|     career focused                      |     0.027350    |     0.167521          |     35.700855    |     0.731624     |     141489.721368    |     1.266667      |     1.475214           |     1.367167       |     -1.050209      |     -0.247981      |
|     standard                      |     0.900433    |     0.965368          |     28.913420    |     1.062049     |     107551.946609    |     0.676768      |     0.440115           |     -1.106918      |     0.706367       |     -0.778269      |
|     well-off                      |     0.505703    |     0.688213          |     55.722433    |     2.129278     |     158391.676806    |     1.129278      |     1.110266           |     1.706153       |     2.031716       |     0.838839       |


We plot data by 2 PCA components: Y axis - component 1, X axis - component 2<br>
![image](https://user-images.githubusercontent.com/77659538/110772298-6c620300-8296-11eb-95af-2244b9f87254.png)

We can clearly identify 4 clusters!

## [2. Purchase Descriptive Analytics](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/2.%20Customer%20Analytics%20-%20Purchase%20Descriptive%20Analysis.ipynb)

In this part, we want to get some ideas about the past bebaviors of our customer: how often they shopped and bought candy bars, which brand they chose more often, and how much they spent. The results can be used to cross-check our predictive results in part 3.

### 2.1 Data Segmentation
We implement the standardization, PCA, and K-means clustering models from previous part, to segment our customers in purchase dataset. We have the following

|          |     ID           |     Day    |     Incidence    |     Brand    |     Quantity    |     Last_Inc_Brand    |     Last_Inc_Quantity    |     Price_1    |     Price_2    |     Price_3    |     ...    |     Promotion_4    |     Promotion_5    |     Sex    |     Marital status    |     Age    |     Education    |     Income    |     Occupation    |     Settlement size    |     Segment    |
|----------|------------------|------------|------------------|--------------|-----------------|-----------------------|--------------------------|----------------|----------------|----------------|------------|--------------------|--------------------|------------|-----------------------|------------|------------------|---------------|-------------------|------------------------|----------------|
|     0    |     200000001    |     1      |     0            |     0        |     0           |     0                 |     0                    |     1.59       |     1.87       |     2.01       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     1    |     200000001    |     11     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     2    |     200000001    |     12     |     0            |     0        |     0           |     0                 |     0                    |     1.51       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     3    |     200000001    |     16     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.98       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |
|     4    |     200000001    |     18     |     0            |     0        |     0           |     0                 |     0                    |     1.52       |     1.89       |     1.99       |     ...    |     0              |     0              |     0      |     0                 |     47     |     1            |     110866    |     1             |     0                  |     0          |

We visualize the proportions of total number of purchases by segments<br>
![image](https://user-images.githubusercontent.com/77659538/110772351-7b48b580-8296-11eb-9695-a712479dd9e1.png)

🔶 Insights: we will most often see fewer-opportunities group shopping candy bars in our store. There are a few possible reasons:
- they are the biggest customer segments (have more observations)
- they visit the store more often than the others (more visits -> more purchases)
- they are more likely to buy candy bars each time they shopping
We will investigate further below

### 2.2 Purchase Occasion and Purchase Incidence
Plot the average number of store visits for each of the four segments using a bar chart, and display the standard deviation as a straight line

![image](https://user-images.githubusercontent.com/77659538/110772378-84d21d80-8296-11eb-830a-5ff15858fec8.png)

🔶 Insights:
- The standard deviation amongst 'Career-Focused' is quite high. This implies that the customers in this segment are at least homogenous that is least alike when it comes to how often they visit the grocery store
- The standard, fewer opportunities, and well-off clusters are very similar in terms of their average store purchases. This is welcome information because it would make them more comparable with respect to our future analysis!

Display the average number of purchases by segments, help us understand how often each group buys candy bars<br>
![image](https://user-images.githubusercontent.com/77659538/110772420-8dc2ef00-8296-11eb-9a9e-a5b856d34386.png)


🔶 Insights:
- For Career-focused, standard deviation is the highest it might be that a part of the segment buys products very frequently, and another part less so. Although consumers in this segment have a somewhat similar income, the way that they might want to spend their money might differ.
- The most homogenous segment appears to be that of the fewer opportunities. This is signified by the segment having the lowest standard deviation or shortest vertical line. The standard segment seems consistent as well with about 25 average purchases and a standard deviation of 30.

### 2.3 Brand Choice
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


Visualize the brand choice by segments (on average, how often each customer buy each brand in each segment)<br>
![image](https://user-images.githubusercontent.com/77659538/110772463-974c5700-8296-11eb-9bf6-d7bafb55949f.png)

🔶 Insights: Each segment has preference on 1 or 2 brands
- Well-off and Career-focused prefer pricy brands
- Fewer-opportunities and standard prefer low price products

### 2.4 Revenue
Compute the total revenue for each of the segments. <br>
|                            |     Revenue Brand 1    |     Revenue Brand 2    |     Revenue Brand 3    |     Revenue Brand 4    |     Revenue Brand 5    |     Total Revenue    |     Segment Proportions    |
|----------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|----------------------|----------------------------|
|     Segment                |                        |                        |                        |                        |                        |                      |                            |
|     Fewer-Opportunities               |     2258.90            |     13909.78           |     722.06             |     1805.59            |     2214.82            |     20911.15         |     0.378                  |
|     Career-Focused         |     736.09             |     1791.78            |     664.75             |     2363.84            |     19456.74           |     25013.20         |     0.222                  |
|     Standard    |     2611.19            |     4768.52            |     3909.17            |     861.38             |     2439.75            |     14590.01         |     0.206                  |
|     Well-Off               |     699.47             |     1298.23            |     725.54             |     14009.29           |     5509.69            |     22242.22         |     0.194                  |


![image](https://user-images.githubusercontent.com/77659538/110772022-23aa4a00-8296-11eb-891d-61725c01aa3e.png)

🔶 Insights:
- Career-focused brings the highest revenue although they are far from the biggest standard segment by total number of purchases
- Well-off brings the second highest revenue even though they are the smallest segment 
- Standard contributes the least though they are not the smallest segment because they tend to buy low-priced products

![image](https://user-images.githubusercontent.com/77659538/110772002-1ee59600-8296-11eb-9475-c3da1926372b.png)

🔶 Insights:
- Brand 3 does not have any segment as its loyal customers. If brand 3 reduces its price, the standard segment could pivot towards it since they seem to be struggling between brand 3 and brand 2.
- Well-off segments mostly prefer brand 4, followed by brand 5. They seem to be not affected by price. Therefore, brand 4 could cautiously try to increase its price. (hypothesis here: will retain most of the customers and increase the revenue per sale)
- Likewise, for career-focused, Brand 5 could increase its price. 


## [3. Purchase Predictive Analytics](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/3.%20Customer%20Analytics%20-%20Purchase%20Predictive%20Analytics.ipynb)

### 3.1 Purchase Probability

#### Model Building
We implement the standardization, PCA, and K-means clustering models from part 1, to segment our customers in purchase dataset.<br>
```
# Y is Incidence (if the customer bought candy bars or not), as we want to predict the purchase probability for our customers
Y = df_pa['Incidence']
```
```
# Dependent variable is based on the average price of all five brands. 
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
We first look at the price information for all brands<br>
|       | Price_1  | Price_2  | Price_3  | Price_4  | Price_5  |
|-------|----------|----------|----------|----------|----------|
| count |    58693 |    58693 |    58693 |    58693 |    58693 |
|  mean | 1.392074 | 1.780999 | 2.006789 | 2.159945 | 2.654798 |
|   std | 0.091139 | 0.170868 | 0.046867 | 0.089825 | 0.098272 |
|   min |      1.1 |     1.26 |     1.87 |     1.76 |     2.11 |
|   25% |     1.34 |     1.58 |     1.97 |     2.12 |     2.63 |
|   50% |     1.39 |     1.88 |     2.01 |     2.17 |     2.67 |
|   75% |     1.47 |     1.89 |     2.06 |     2.24 |      2.7 |
|   max |     1.59 |      1.9 |     2.14 |     2.26 |      2.8 |

Since the prices of all 5 brands ranges with from 1.1 to 2.8. We will perform analysis on a slightly wider price range: 0.5 - 3.5<br>
Then we fit our 'test price range' in our model to get the corresponding Purchase Probability for each price point.<br>
Next, we apply below formula to derive the price elasticity at each price point<br>
```
# Elasticity = beta*price*(1-P(purchase))
pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_pr)
```
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110493304-921fc880-812d-11eb-834d-4094585e315e.png)

🔶 Insights: we should decrease the overall price so we can gain more on overall purchase probability
- With prices lower than 1.25, we can increase our product price without losing too much in terms of purchase probability. For prices higher than 1.25, We have more to gain by reducing our prices.
- Since all brands have average price over 1.25, it's not good news for us.
We have to investigate further by segments!



#### Purchase Probability by Segments
![image](https://user-images.githubusercontent.com/77659538/110771869-f6f63280-8295-11eb-8590-513b246f31df.png)

🔶 Insights:
- The well-off segment are the least elastic when compared to the rest. So, their purchase probability elasticity is not as affected by price. Fewer-opportunities are a lot more price-sensitive than other groups
- The price elasticities for the fewer-opportunities segment seems to differ across price range (low in low prices, high in high prices). Reasons might be:
  - We have more observations, so it is more accurate
  - This segments enjoys candy bars so much that a price increase in the low price range doesn't affect them; once it becomes expensive, it doesn't make any financial sense to them to invest in it

#### Purchase Probability with and without Promotion Feature
we prepare the data and decide Y and X variable
```
Y = df_pa['Incidence']

X = pd.DataFrame()
X['Mean_Price'] = (df_pa['Price_1'] + 
                   df_pa['Price_2'] + 
                   df_pa['Price_3'] + 
                   df_pa['Price_4'] + 
                   df_pa['Price_5']) / 5

# Include a second promotion feature. 
#  To examine the effects of promotions on purchase probability.
# Calculate the average promotion rate across the five brands. 
#  Add the mean price for the brands.
X['Mean_Promotion'] = (df_pa['Promotion_1'] +
                       df_pa['Promotion_2'] +
                       df_pa['Promotion_3'] +
                       df_pa['Promotion_4'] +
                       df_pa['Promotion_5'] ) / 5
```
We visualize the results with without promo side-by-side
![image](https://user-images.githubusercontent.com/77659538/110770861-ceba0400-8294-11eb-9f5b-282f288169db.png)

🔶 Insights: when we apply the promotion, we can at the same time increase the price a little bit without the fear that they will be less likely to buy our products
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

🔶 Interpretation: each coefficient shows how the price change would affect the probability of choosing the relative brand. In general, brand choice probability goes up if its own price is lower and other brands' prices are higher.

#### Own Price Elasticity Brand 5
We fit our model and calculate the price elasticity for brand 5 at the same 'test price range'<br>
By visualizing the result, we get<br>
![image](https://user-images.githubusercontent.com/77659538/110494977-0c048180-812f-11eb-8b0e-699f82603802.png)

🔶 Interpretation: It shows us how it would affect brand 5 if they change their own price. 

#### Cross Price Elasticity Brand 5, Cross Brand 4
To calculate the cross brand price elasticity, we use new formula
```
brand5_cross_brand4_price_elasticity = -beta5 * price_range * pr_brand_4
```

We visualize the the cross-price elasticity of purchase probability for brand 5 vs brand 4<br>
![image](https://user-images.githubusercontent.com/77659538/110495271-5423a400-812f-11eb-9c59-4023c515089d.png)

🔶 Interpretation: It shows us how it would affect brand 5 if brand 4 change their price. 

![image](https://user-images.githubusercontent.com/77659538/110764924-957e9580-828e-11eb-80cc-151841f98585.png)


🔶 Insights:
- Brand 4 is a strong substitute for brand 5 for all prices up to \$1.65
  - Note: the observed price range of brand 4 lies between \$1.76 and \$2.6 in this region
  - These prices are out of the natural domain of brand 4, therefore if brand 4 had a substantially lower price it would be a very strong competitor a brand 5
- Even though the elasticity starts to decrease from the 1.45 mark, it is still positive, signaling that the increase in purchase probability for brand 5 happens more slowly.
  - When it comes to average customer, brand 4 is a weak substitute for brand 5 
  - Brand 5 can create a marketing strategy targeting customers who choose brand 4, and attract them to buy own brand 5

#### Own and Cross-Price Elasticity by Segment
![image](https://user-images.githubusercontent.com/77659538/110771779-da59fa80-8295-11eb-9d7e-f3b9ad51c093.png)


🔶 Insights: Brand 5 should decrease its own price offering while gaining solid market share from the well-off and retaining the career-focused segment, the most frequent buyers of brand 5
- For Career-focused segment, Brand 5 could increase its price, without fear of significant loss of customers from this segment
  - The Career-focused segment is the most inelastic and they do not seem to be that affected by price
  - The cross price elasticity also has extremely low values, meaning they are unlikely to switch to brand 4
- For the Well-off segment, we'd better decrease brand 5 price to gain market share from this segment
  - For this segment, own elasticity is much higher than 'career-focused'
  - They also purchase the competitor brand 4 most often by having highest cross brand elasticity, meaning a tiny increase in price will lose customers

### 3.3 Purchase Quantity

#### Model Estimation

To determine price elasticity of purchase quantity, also known as price elasticity of demand, we're interested in purchase occasion, where the purchased quantity is different from 0.
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
🔶 Interpretation: It appears that promotion reflects negatively on the purchase quantity of the average client, which is unexpected.

#### Price Elasticity of Purchase Quantity with and without Promotion
Calculate the price elasticity with new formula
```
price_elasticity_quantity_promotion_yes = beta_quantity * price_range / predict_quantity
```
Plot the two elasticities (with and without promotion) side by side<br>
![image](https://user-images.githubusercontent.com/77659538/110770960-f1e4b380-8294-11eb-9843-451a59cbd383.png)

🔶 Insights:
- We observe that the two elasticities are very close together for almost the entire price range.
- It appears that promotion does not appear to be a significant factor in the customers' decision what quantity of chocolate candy bars to purchase.
