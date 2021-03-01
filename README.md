# Customer Analytics for Retail Store

## Project Overview
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
*'segmentation data.csv'* <br>
_'purchase data.csv'_ <br>

## Customer Segmentation
Dataset: *'segmentation data.csv'* <br>

### K-Means Clustering
__K-Means WCSS__<br>
![image](https://user-images.githubusercontent.com/77659538/109501544-b0088000-7ad2-11eb-9fcd-b34a7a21429b.png)

__K-Means Segmentation__<br>
![image](https://user-images.githubusercontent.com/77659538/109501602-c6164080-7ad2-11eb-9268-bbe8d4024a54.png)

### K-Means Clustering with PCA
__PCA__<br>
![image](https://user-images.githubusercontent.com/77659538/109501727-f827a280-7ad2-11eb-83d7-67c04d98b694.png)

![image](https://user-images.githubusercontent.com/77659538/109501741-fe1d8380-7ad2-11eb-82f9-0336953a6380.png)

![image](https://user-images.githubusercontent.com/77659538/109501751-01187400-7ad3-11eb-8cd5-fa20bbea2133.png)

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

