---
layout: default
title: NBA Game Prediction
---

# NBA Game Prediction System Using Composite Stacked Machine Learning Model
#### <a href="https://inoliao.github.io/portfolio/" target="_blank">I-No Liao</a>

## Abstract
<div align="justify">
Predicting NBA games is not an easy task. Conventional data analysis and statistic approaches are usually complicated and the accuracy is not high. In this work, a machine learning based method is proposed and the complete design flow is thoroughly introduced and explained. 8 single-stage machine learning models are trained and compared. More complex composite models such as voting mechanism and stacking method are also designed and elaborated. The proposed model reaches 76.8% accuracy on predicting all 2018 NBA playoffs. Furthermore, for the Eastern Conference Final, Western Conference Finals, and Conference Finals, our model achieves an extraordinary prediction accuracy of 85.7%, 71.4%, and 100%, respectively. The source code and dataset are available <a href="https://github.com/INoLiao/nbaGamePrediction" target="_blank">here</a>.
</div>

## Problem Definition
<div align="justify">
The mission of this work is to precisely predict NBA games' winning and losing result. Machine learning models are trained to predict game results based on the information of two teams' recent status. The idea is expressed as in Figure 1. Beside game-winning prediction, information of the prediction's confidence level is also valuable for us to understand how intense the matchup might be.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/problem_definition_1.png" alt="Problem Definition"/>
<center><h4> Figure 1: Problem definition. </h4></center>

## Dataset
<div align="justify">
This work predicts NBA games based on the dataset collected from the <a href="https://stats.nba.com" target="_blank">official NBA stats website</a>. A crawler program is designed to scrape the game boxes and save the data automatically. Details about the crawler design are available <a href="https://github.com/INoLiao/nbaGamePrediction" target="_blank">here</a>. NBA games, including seasons and playoffs, from 1985 to 2018 are collected. The dataset contains 68,458 season matches and 4,816 playoff matches. Figure 2 shows the number of games played by all 30 NBA teams. Due to various history from each team, the number of games for 30 teams are not homogeneous. Moreover, since there are at most 16 teams entitled to enter playoff each year, the number of playoff games played by each team is different as well.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/dataset_1.png" alt="Number of games played by each team"/>
<center><h4> Figure 2: Statistics of the number of games played by each NBA team. </h4></center>

<div align="justify">
Before processing our data, a classification regarding data types is conducted. Table 1 shows data types of the dataset. As we can see, most of the data are numeric. There is one categorical data, Team, and there are two binary data, Win/Lose and Home/Away. Our target is to precisely predict which team wins a game when two teams meet. Therefore, Win/Lose is the label and our machine learning model predicts the Win/Lose outcome and provides the confidence level of its prediction.
</div>


<center><h4> Table 1: Type of Attirbutes </h4></center>

<table>
    <thead>
        <tr>
            <th > Binary </th>
            <td> Win/Lose, Home/Away </td>
        </tr>
        <tr>
            <th > Categorical </th>
            <td> Team </td>
        </tr>
        <tr>
            <th > Numeric </th>
            <td> Date, PTS, FG%, FGM, FGA, 3P%, 3PM, 3PA, FT%, FTM, FTA, REB, OREB, DREB, AST, STL, BLK, TOV, PF </td>
        </tr>        
    </thead>
</table>

## Data Preprocessing and Feature Extraction
<div align="justify">
Typical data preprocessing is conducted as shown in Figure 3. Preprocessing includes data cleaning, one-hot encoding, numeric data normalization, game pairing, validity checking, etc. The final legitimate data volume is 61,368, including seasons and playoffs.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/data_preprocessing_1.png" alt="Data preprocessing block diagram"/>
<center><h4> Figure 3: Data preprocessing flow chart. </h4></center>

<div align="justify">
To train machine learning models, feature extraction is carried out as shown in Figure 4 and 5. Firstly, select the attributes that are more representative to the winning or losing of games. Then, put all selected attributes in a vector. The attribute X is the average performance considered from previous games played by two teams prior to the date we target to predict. In other words, attribute X represents the teams' recent status. Label Y is Win/Lose since we would like to predict which team wins the game.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/feature_extraction_1.png" alt="Feature extraction"/>
<center><h4> Figure 4: Feature extraction. </h4></center>

<br>
<center><img src="{{ site.baseurl }}/assets/img/feature_extraction_2.png" alt="Feature extraction" width="550"/></center>
<center><h4> Figure 5: How attribute X and label Y look like. </h4></center>

## Model Training and Testing
<div align="justify">
After data preprocessing and feature extraction are completed, model training and testing can proceed. In this section, grid search with cross validation is firstly applied to find the optimal model parameters. Afterward, data size evaluation is conducted to help us understand how data volume influences the model performance. Then, voting and stacking models are introduced. At last, a comprehensive performance comparison of different machine learning models is presented.
</div>

### Grid Search with Cross Validation
<div align="justify">
8 different frequently used single-stage machine learning models are analyzed in this work. Model parameters are optimized by grid search. To prevent possible overfitting issue that happens frequently in model training, cross validation is applied. Table 2 presents which parameters are considered and in what ranges are they examined. Note that since the Naïve Bayes model has no parameters to choose, it does not require grid search.
</div>

<center><h4> Table 2: Grid Search Parameters </h4></center>

<table>
    <thead>
        <tr>
            <th > Model </th>
            <th> Parameters Sweeping Table </th>
            <th > Model </th>
            <th> Parameters Sweeping Table </th>
        </tr>     
    </thead>
    <tbody>
        <tr>
            <td> Logistic Regression </td><td> 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],<br>'max_iter': [100, 200, 300, 400, 500] </td>
            <td> GBDT </td><td> 'loss': ['deviance', 'exponential'],<br>'n_estimators': [600, 800, 1000],<br>'learning_rate': [0.1, 0.2, 0.3],<br>'max_depth': [3, 5, 10],<br>'subsample': [0.5],<br>'max_features': ['auto', 'log2', 'sqrt'] </td>
        </tr>
        <tr>
            <td> SVM </td><td> 'C': [0.01, 0.1, 1, 10, 100],<br>'kernel': ['rbf', 'linear'],<br>'gammas': ['auto', 0.001, 0.01],<br>'shrinking': [True, False] </td>
            <td> LightGBM </td><td> 'learning_rate': [0.1, 0.2, 0.3],<br>'n_estimators': [600, 800, 1000],<br>'max_depth': [-1, 5, 10],<br>'subsample' : [0.5] </td>
        </tr>
        <tr>
            <td> XGBoost </td><td> 'max_depth': [3, 5, 7],<br>'learning_rate': [0.1, 0.3],<br>'n_estimators': [100, 200, 300],<br>'min_child_weight': [1, 3],<br>'gamma': [x/10 for x in range(0, 5)] </td>
            <td> AdaBoost </td><td> 'learning_rate': [1, 0.1, 0.2, 0.3],<br>'n_estimators': [50, 100, 600, 800, 1000] </td>
        </tr>
        <tr>
            <td> Random Forest </td><td> 'n_estimators': [600, 800, 1000],<br>'criterion': ['gini', 'entropy'],<br>'bootstrap': [True, False],<br>'max_depth': [None, 5, 10],<br>'max_features': ['auto', 'log2', 'sqrt'] </td>
            <td> Naïve Bayes </td><td> N/A </td>
        </tr>        
        
    </tbody>
</table>

### Data Size Evaluation
<div align="justify">
The data size evaluation is an important step when training models. Since the play style of NBA games changes rapidly as time goes, training models using more data does not mean a better prediction accuracy. As a result, the relation between training data size and performance is evaluated and the outcome is presented in Table 3. As shown in the table, training data covering three-year previous games presents the best performance and it is chosen as the optimal dataset for all our models.
</div>

<center><h4> Table 3: Data Size Evaluation </h4></center>

<table>
    <thead>
        <tr>
            <th rowspan = "2"> Training Data (yr) </th>
            <th rowspan = "2"> Training Data (#) </th>
            <th colspan = "8"> Accuracy (%) </th>
        </tr>
        <tr>
            <th> LogiRegr </th>
            <th> SVM </th>
            <th> XGBoost </th>
            <th> Naïve Bayes </th>
            <th> Random Forest </th>
            <th> GBDT </th>
            <th> LightGBM </th>
            <th> AdaBoost </th>
        </tr>
        
    </thead>
    <tbody>
        <tr>
            <td> 1 </td><td> 2460 </td><td> 69.6 </td><td> 70.9 </td><td> 74.7 </td><td> 60.8 </td><td> 68.4 </td><td> 72.2 </td><td> 68.4 </td><td> 73.4 </td>
        </tr>
        <tr>
            <td> 2 </td><td> 5078 </td><td> 70.9 </td><td> 72.2 </td><td> 72.2 </td><td> 59.5 </td><td> 69.6 </td><td> 69.6 </td><td> 74.7 </td><td> 68.4 </td>
        </tr><tr>
            <td> 3 </td><td> 7234 </td><td> 70.9 </td><td> 74.7 </td><td> 74.7 </td><td> 60.8 </td><td> 70.9 </td><td> 73.4 </td><td> 68.4 </td><td> 76.0 </td>
        </tr><tr>
            <td> 4 </td><td> 9370 </td><td> 69.6 </td><td> 72.2 </td><td> 72.2 </td><td> 59.5 </td><td> 72.2 </td><td> 70.9 </td><td> 73.4 </td><td> 73.4 </td>
        </tr><tr>
            <td> 5 </td><td> 11702 </td><td> 70.9 </td><td> 70.9 </td><td> 76.0 </td><td> 59.5 </td><td> 74.7 </td><td> 74.7 </td><td> 69.6 </td><td> 74.7 </td>
        </tr>
        
    </tbody>
</table>


### Voting
<div align="justify">
To prevent bias from a single machine learning model, a voting mechanism, as shown in Figure 6, is applied to make the prediction decision more convincing. 5 machine learning models, including Logistic Regression, SVM, XGBoost, GBDT, and AdaBoost, are considered in the voting model owing to their better performance. The voting mechanism is simple. The decision agreed by most of the models is the final decision. Furthermore, the ratio of agreed votes to total votes is an indicator implying the confidence level of the final decision.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/voting_1.png" alt="Voting Model"/>
<center><h4> Figure 6: Voting model. </h4></center>

### Stacking
<div align="justify">
Stacking is a more sophisticated approach that consolidates the predictions from multiple well-trained models and uses them as a new set of training attributes to train another model. It can be considered a multi-stage model or a stacked model that is helpful for preventing bias from certain models. At some level, is can be seen as a mode complicated voting mechanism. Figure 7 shows the block diagram of the stacking model and the details of how stacking works are presented in Figure 8. In this work, several combinations of different machine learning models constructing the stacked model are evaluated. In addition, both 2-stage and 3-stage stacked models are analyzed.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/stacking_1.png" alt="Stacking Model"/>
<center><h4> Figure 7: Stacking model. </h4></center>

<br>
<img src="{{ site.baseurl }}/assets/img/stacking_2.png" alt="Stacking Model"/>
<center><h4> Figure 8: Details in stacking block. </h4></center>

<div align="justify">
As shown in Table 4, 3-stage stacking model is slightly better than 2-stage stacking model. To thoroughly consider all models, 2-stage stacking of SVM/GBDT/XGBoost + AdaBoost and 3-stage stacking of SVM/XGBoost + RF/GBDT + AdaBoost are selected for the consideration of the final performance comparison.
</div>

<center><h4> Table 4: Stacking Model Performance Evaluation </h4></center>

<table>
    <thead>
        <tr>
            <th> Stacking </th>
            <th> Stage 1 </th>
            <th> Stage 2 </th>
            <th> Final Stage </th>
            <th> Total Estimators (#) </th>
            <th> Prediction Accuracy (%) </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan = "7"> 2-Stage </td>
        </tr>
        <tr><td> SVM/GBDT/XGBoost </td><td> None </td><td> AdaBoost </td><td> 4 </td><td> 76.8 (1st) </td></tr>
        <tr><td> SVM/GBDT/AdaBoost </td><td> None </td><td> XGBoost </td><td> 4 </td><td> 74.4 </td></tr>
        <tr><td> SVM/XGBoost/AdaBoost </td><td> None </td><td> GBDT </td><td> 4 </td><td> 72.0 </td></tr>
        <tr><td> XGBoost/GBDT/AdaBoost </td><td> None </td><td> SVM </td><td> 4 </td><td> 73.2 </td></tr>
        <tr><td> SVM/RF/GBDT/XGBoost </td><td> None </td><td> AdaBoost </td><td> 5 </td><td> 74.4 </td></tr>
        <tr><td> SVM/RF/GBDT/AdaBoost </td><td> None </td><td> XGBoost </td><td> 5 </td><td> 72.0 </td></tr>
        <tr>
            <td rowspan = "6"> 3-Stage </td>
        </tr>
        <tr><td> SVM/XGBoost </td><td> RF/GBDT </td><td> AdaBoost </td><td> 5 </td><td> 76.8 (1st) </td></tr>
        <tr><td> RF/GBDT </td><td> SVM/XGBoost </td><td> AdaBoost </td><td> 5 </td><td> 75.6 </td></tr>
        <tr><td> SVM/AdaBoost </td><td> RF/GBDT </td><td> XGBoost </td><td> 5 </td><td> 75.6 </td></tr>
        <tr><td> RF/GBDT </td><td> SVM/AdaBoost </td><td> XGBoost </td><td> 5 </td><td> 75.6 </td></tr>
        <tr><td> SVM/RF </td><td> XGBoost/GBDT </td><td> AdaBoost </td><td> 5 </td><td> 75.6 </td></tr>
    </tbody>
</table>

## Experimental Results
<div align="justify">
This work evaluates eight single-stage models, one voting model, one 2-stage stacked model, and one 3-stage stacked model. The performance comparison is summarized in Table 5. We can observe that for the single-stage estimators, all models have decent prediction accuracy except for Naïve Bayes and LightGBM. Moreover, composite models such as voting and stacking are even more accurate than single-stage estimators. AdaBoost, 2-stage stacked, and 3-stage stacked models possess the peak performance of 76.8 % prediction accuracy. In conclusion, stacked machine learning model is an appropriate approach for our task.
</div>

<center><h4> Table 5: 2018 NBA Playoff Game Winning Prediction </h4></center>

<table>
    <thead>
        <tr>
            <th> Model </th>
			  <th> Algorithms/Architectures </th>
			  <th> Prediction Accuracy (%) </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan = "9"> Single-Stage Estimator </td>
        </tr>
        <tr><td> Logistic Regression </td><td> 72.0 </td></tr>
        <tr><td> SVM </td><td> 75.6 </td></tr>
        <tr><td> XGBoost </td><td> 75.6 </td></tr>
        <tr><td> Naïve Bayes </td><td> 62.2 </td></tr>
        <tr><td> Random Forest </td><td> 72.0 </td></tr>
        <tr><td> GBDT </td><td> 74.4 </td></tr>
        <tr><td> LightGBM </td><td> 69.5 </td></tr>
        <tr><td> AdaBoost </td><td> 76.8 </td></tr>
        <tr>
            <td> Voting </td>
            <td> Logistic Regreesion/SVM/XGBoost/GBDT/AdaBoost </td><td> 73.2 </td>
        </tr>
        <tr>
            <td> 2-Stage Stacking </td>
            <td> SVM/GBDT/XGBoost + AdaBoost </td><td> 76.8 </td>
        </tr>
        <tr>
            <td> 3-Stage Stacking </td>
            <td> SVM/XGBoost + RF/GBDT + AdaBoost </td><td> 76.8 </td>
        </tr>
    </tbody>
</table>

<div align="justify">
The most important games in the NBA are Eastern/Western Finals and the Conference Finals. GBDT is applied as an example to show our predictions on each game as shown in Table 6. The accuracy of the model prediction manifests the tension of the games to some extent. For example, in the 2018 NBA Conference Finals, Golden State Warriors swept Cleveland Cavaliers and our model precisely predicted the fact without incorrect predictions. As shown in the table, only one game had a confidence level lower than 60% and that game was indeed more intense than the other three games. As for Eastern and Western Conference Finals, since both matchups were more competitive, the resulting confidence level of our model was lower compared to the Conference Finals. In summary, this work designs a machine learning model that can reach prediction accuracy of 85.7%, 71.4%, and 100%  for Eastern Conference Final, Western Conference Finals, and Conference Finals, respectively.
</div>

<center><h4> Table 6: 2018 NBA Finals/Semi-Finals Game Winning Prediction by GBDT Model </h4></center>

<table>
    <thead>
        <tr>
            <th> Game (#) </th>
            <th> Home </th>
            <th> Away </th>
            <th> Actual Winner </th>
            <th> Predicted Winner </th>
            <th> Confidence (%) </th>
            <th> Accuracy (%) </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th colspan = "7"> NBA Conference Finals </th>
        </tr>
        <tr>
            <td> 1 </td>
            <td> GSW </td>
            <td> CLE </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 70.9 </td>
            <td rowspan = "4"> 100.0 </td>
        </tr>
        <tr>
            <td> 2 </td>
            <td> GSW </td>
            <td> CLE </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 68.9 </td>
        </tr>
        <tr>
            <td> 3 </td>
            <td> CLE </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 58.3 </td>
        </tr>
        <tr>
            <td> 4 </td>
            <td> CLE </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 62.3 </td>
        </tr>
        
        <tr>
            <th colspan = "7"> NBA Western Conference Finals </th>
        </tr>
        <tr>
            <td> 1 </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 56.3 </td>
            <td rowspan = "7"> 71.4 </td>
        </tr>
        <tr>
            <td> 2 </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> HOU </td>
            <td> 53.9 </td>
        </tr>
        <tr>
            <td> 3 </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 50.2 </td>
        </tr>
        <tr>
            <td> 4 </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> 64.2 </td>
        </tr>
        <tr>
            <td> 5 </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> HOU </td>
            <td> 60.0 </td>
        </tr>
        <tr>
            <td> 6 </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> 54.0 </td>
        </tr>
        <tr>
            <td> 7 </td>
            <td> HOU </td>
            <td> GSW </td>
            <td> GSW </td>
            <td> HOU </td>
            <td> 63.7 </td>
        </tr>
        
        <tr>
            <th colspan = "7"> NBA Eastern Conference Finals </th>
        </tr>
        <tr>
            <td> 1 </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> BOS </td>
            <td> 53.6 </td>
            <td rowspan = "7"> 85.7 </td>
        </tr>
        <tr>
            <td> 2 </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> BOS </td>
            <td> 58.9 </td>
        </tr>
        <tr>
            <td> 3 </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> CLE </td>
            <td> 61.2 </td>
        </tr>
        <tr>
            <td> 4 </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> CLE </td>
            <td> 61.3 </td>
        </tr>
        <tr>
            <td> 5 </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> BOS </td>
            <td> 57.8 </td>
        </tr>
        <tr>
            <td> 6 </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> CLE </td>
            <td> 55.9 </td>
        </tr>
        <tr>
            <td> 7 </td>
            <td> BOS </td>
            <td> CLE </td>
            <td> CLE </td>
            <td> BOS </td>
            <td> 51.8 </td>
        </tr>
    </tbody>
</table>