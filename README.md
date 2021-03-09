# Customer Analytics for Retail Store

## Project Overview
This project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate suitable marketing strategies to maximize its revenue on Candy Bars. 

To measure the 
The key metric to measure the variation of 
Analysis are mainly conducted on segmentation and positioning, two major components of the tradition STP framework, to extract insights on customer characteristics and purchase behaviors. 


* Goal: this project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate suitable marketing strategies for different brands of candy bars according to the insights gained through customer segementation analytics, purchase descriptive analysis, and purchase predictive analytics
* Reasons: it helps to create a single, accurate view of a customer to make decisions about how best to acquire and retain customers, identify high-value customers and proactively interact with them
* Approach: this project conducts analysis on Segmentation and Positioning, two major components of the traditional STP Framework, by calculating the following parameters within each customer segments
  * Purchase probability
  * Brand choice probability
  * Purchase quantity

## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Dataset Source__: https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases

## Datasets Information
[_**segmentation data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=segmentation+data.csv)<br> contains the data of our customers that we use to build model for segmentation
[_**'purchase data.csv'**_](https://www.kaggle.com/shawnysun/fmcg-data-customers-and-purchases?select=purchase+data.csv)



## [1. Segmentation](https://github.com/shawn-y-sun/Customer_Analytics_Retail/blob/main/1.Customer%20Analytics%20-%20Customer%20Segmentation.ipynb)

### Exploratory Analysis

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

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)Insights: we can spot some level of correlations between certain pairs of variables, such as Income vs Occupation, Education vs Age, Settlement Size vs Occupation. It indicates that we can reduce the dimensions of portraying our customers without losing too much information, allowing us to segmenting our customers more accurately.


### Clustering
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

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)Insights: we have 4 segments of customers
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

Visualize the loadings by heatmap
![image](https://user-images.githubusercontent.com/77659538/110481794-e83b3e80-8122-11eb-9438-02b1742b8e84.png)

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)Insights: each component shows a dimension of individual features
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







____________________






__K-Means Clustering with PCA__<br>
![image](https://user-images.githubusercontent.com/77659538/109501936-45a40f80-7ad3-11eb-946d-935919e8ceb9.png)

### Model and Files Output

```
pickle.dump(scaler, open('scaler.pickle', 'wb'))
pickle.dump(pca, open('pca.pickle', 'wb'))
pickle.dump(kmeans_pca, open('kmeans_pca.pickle', 'wb'))
```

## Purchase Descriptive Analysis
Dataset: *'purchase data.csv'* <br>

__Segmentation Proportions__<br>
![image](https://user-images.githubusercontent.com/77659538/109502230-b0554b00-7ad3-11eb-93cd-9987164707c6.png)

__Purchase Occasion and Purchase Incidence__<br>
![image](https://user-images.githubusercontent.com/77659538/109502296-c531de80-7ad3-11eb-842c-fc41f4a3e7c6.png)

Insights:
- The standard deviation amongst customers from the second segment is quite high. This implies that the customers in this segment are at least homogenous that is least alike when it comes to how often they visit the grocery store
- The standard fewer opportunities and well-off clusters are very similar in terms of their average store purchases.
This is welcome information because it would make them more comparable with respect to future analysis

![image](https://user-images.githubusercontent.com/77659538/109502755-5d2fc800-7ad4-11eb-8aab-dec5310f4339.png)

Insights:
- For Career-focused, standard deviation is the highest it might be that a part of the segment buys products very frequently.And another part less so. Although consumers in this segment have a somewhat similar income, the way that they might want to spend their money might differ.
- The most homogenous segment appears to be that of the fewer opportunities. This is signified by the segment having the lowest standard deviation or shortest vertical line 
- The first segment seems consistent as well with about 25 average purchases and a standard deviation of 30

__Brand Choice__<br>
![image](https://user-images.githubusercontent.com/77659538/109502877-86505880-7ad4-11eb-9f12-6026af9e58ee.png)

__Revenue__

| Segment             | Revenue Brand 1 | Revenue Brand 2 | Revenue Brand 3 | Revenue Brand 4 | Revenue Brand 5 | Total Revenue | Segment Proportions |
|---------------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------|---------------------|
| Standard            | 2258.9          | 13909.78        | 722.06          | 1805.59         | 2214.82         | 20911.15      | 0.378               |
| Career-Focused      | 736.09          | 1791.78         | 664.75          | 2363.84         | 19456.74        | 25013.2       | 0.222               |
| Fewer-Opportunities | 2611.19         | 4768.52         | 3909.17         | 861.38          | 2439.75         | 14590.01      | 0.206               |
| Well-Off            | 699.47          | 1298.23         | 725.54          | 14009.29        | 5509.69         | 22242.22      | 0.194               |

Insights:
- Career-focused is the most prominent segment<br>
- Standard contributes the least<br>
- If brand 3 reduces its price, the Standard segment could pivot towards it<br>
- For well-off segment, Brand 4 could increase its price

## Purchase Predictive Analytics

### Purchase Probability

__Price Elasticity of Purchase Probability__<br>
![image](https://user-images.githubusercontent.com/77659538/109503434-3625c600-7ad5-11eb-8eee-c26db33c2141.png)

Insights:
- With prices lower than 1.25, we can increase our product price without losing too much in terms of purchase probability<br>
- For prices higher than 1.25, We have more to gain by reducing our prices

__Purchase Probability by Segments__<br>
![image](https://user-images.githubusercontent.com/77659538/109503627-784f0780-7ad5-11eb-9f49-2f85c86bb6e7.png)

Insights:
- The Career-focused segment are the least elastic when compared to the rest; so, their purchase probability elasticity is not as affected by price <br>
- The price elasticities for the Standard segment seem to differ across price range; this may be due to the fact that the standard segment is least homogenous, which we discovered during our descriptive analysis <br>
- It may be that the customers in this segment have different shopping habbits, which is why their customers start with being more elastic than average but then shift to being more inelastic than the average customer and indeed the Career-focused segment

__Purchase Probability with Promotion Feature__<br>
![image](https://user-images.githubusercontent.com/77659538/109503858-c06e2a00-7ad5-11eb-8ec3-1d3cca70ae9b.png)

Insights:
- The purchase probability elasticity of the customer is less elastic when there is promotion
- This is an important insight for marketers, as according to our model people are more likely to buy a product if there is some promotional activity rather than purchase a product with the same price, when it isn't on promotion

### Brand Choice Probability

__Own and Cross-Price Elasticity by Segment of Brand 5__<br>
![image](https://user-images.githubusercontent.com/77659538/109504054-0925e300-7ad6-11eb-94b4-ba8aa675647a.png)

Insights:
We can observe differences and similiraties between the segments and examine their preference, when it comes to brand choice.<br>
The two segments, which seem to be of most interested for the marketing team of brand 5, seem to be the 'Career-focused' and the 'Well-off'. They are also the segments which purchase this brand most often. 
-  The Career-focused segment is the most inelastic and they are the most loyal segment. 
    -  Based on our model, they do not seem to be that affected by price, therefore brand 5 could increase its price, without fear of significant loss of customers from this segment. 
-  The Well-off segment on the other hand, seems to be more elastic. They also purchase the competitor brand 4 most often.
    -  In order to target this segment, our analysis signals, that price needs to be decreased. However, keep in mind that other factors aside from price might be influencing the purchase behaivour of this segment.

### Purchase Quantity

__## Price Elasticity with Promotion__<br>
![image](https://user-images.githubusercontent.com/77659538/109504250-4db17e80-7ad6-11eb-97ed-6acac5851839.png)

Insights:
- We observe that the two elasticities are very close together for almost the entire price range
- It appears that promotion does not appear to be a significant factor in the customers' decission what quantity of chocolate candy bars to purchase

