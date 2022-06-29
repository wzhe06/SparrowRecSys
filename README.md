# Sparrow Recommender System (English Version)

## Introduction of SparrowRecSys

Sparrow Recommender System (SparrowRecSys) is a movie recommender system, the name SparrowRecSys, is taken from the meaning of "one sparrow is small, but has all the internal organs". The project is a maven-based mixed language project, which also includes different modules of recommendation systems such as TensorFlow, Spark, and Jetty Server. I hope readers can use SparrowRecSys to learn about recommender systems and have the opportunity to improve it together.

## Practical Lessons based on SparrowRecSys (Original Version)

Invited by Geek Time to offer the [Deep Learning Recommendation System Combat](http://gk.link/a/10lyE) course, which explains all the technical details of SparrowRecSys in detail, covering deep learning model structure, model training, feature engineering, Modules such as model evaluation, model online service and recommendation server internal logic.

## Environment Setup and Dependencies

- Java 8
- Scala 2.11
- Python 3.6+
- TensorFlow 2.0+

## Quick Start

After opening the project with `IntelliJ IDEA`, find `RecSysServer`, right-click `Run`, and then enter `http://localhost:6010/` in the browser to see the front-end effect of the recommendation system.

## Project Data Source

The project data comes from the open-source movie dataset [MovieLens](https://grouplens.org/datasets/movielens/). The MovieLens dataset is simplified by the project's own dataset, and only 1000 movies and related comments and user data are retained. . Please download the full dataset from the official MovieLens website. MovieLens 20M Dataset is recommended.

## SparrowRecSys Technical Architecture

The technical architecture of SparrowRecSys follows the classic industrial-level deep learning recommendation system architecture, including multiple modules such as offline data processing, model training, near-line stream processing, online model service, and front-end recommendation result display. The following is the architecture diagram of SparrowRecSys:
![English Version of SparowRecSys](https://github.com/cuicaihao/Sparrow-Recommender-System/blob/master/docs/sparrowrecsysarch.png?raw=true)

## SparrowRecSys Deep Learning Model(s)

- Word2vec (Item2vec)
- DeepWalk (Random Walk based Graph Embedding)
- Embedding MLP
- Wide&Deep
- Nerual CF
- Two Towers
- DeepFM
- DIN(Deep Interest Network)

## References

- [[FFM] Field-aware Factorization Machines for CTR Prediction (Criteo 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFFM%5D%20Field-aware%20Factorization%20Machines%20for%20CTR%20Prediction%20%28Criteo%202016%29.pdf) <br />
- [[GBDT+LR] Practical Lessons from Predicting Clicks on Ads at Facebook (Facebook 2014)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BGBDT%2BLR%5D%20Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook%20%28Facebook%202014%29.pdf) <br />
- [[PS-PLM] Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction (Alibaba 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BPS-PLM%5D%20Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction%20%28Alibaba%202017%29.pdf) <br />
- [[FM] Fast Context-aware Recommendations with Factorization Machines (UKON 2011)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFM%5D%20Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines%20%28UKON%202011%29.pdf) <br />
- [[DCN] Deep & Cross Network for Ad Click Predictions (Stanford 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDCN%5D%20Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions%20%28Stanford%202017%29.pdf) <br />
- [[Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeep%20Crossing%5D%20Deep%20Crossing%20-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features%20%28Microsoft%202016%29.pdf) <br />
- [[PNN] Product-based Neural Networks for User Response Prediction (SJTU 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BPNN%5D%20Product-based%20Neural%20Networks%20for%20User%20Response%20Prediction%20%28SJTU%202016%29.pdf) <br />
- [[DIN] Deep Interest Network for Click-Through Rate Prediction (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIN%5D%20Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202018%29.pdf) <br />
- [[ESMM] Entire Space Multi-Task Model - An Effective Approach for Estimating Post-Click Conversion Rate (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BESMM%5D%20Entire%20Space%20Multi-Task%20Model%20-%20An%20Effective%20Approach%20for%20Estimating%20Post-Click%20Conversion%20Rate%20%28Alibaba%202018%29.pdf) <br />
- [[Wide & Deep] Wide & Deep Learning for Recommender Systems (Google 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BWide%20%26%20Deep%5D%20Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems%20%28Google%202016%29.pdf) <br />
- [[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BxDeepFM%5D%20xDeepFM%20-%20Combining%20Explicit%20and%20Implicit%20Feature%20Interactions%20for%20Recommender%20Systems%20%28USTC%202018%29.pdf) <br />
- [[Image CTR] Image Matters - Visually modeling user behaviors using Advanced Model Server (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BImage%20CTR%5D%20Image%20Matters%20-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server%20%28Alibaba%202018%29.pdf) <br />
- [[AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BAFM%5D%20Attentional%20Factorization%20Machines%20-%20Learning%20the%20Weight%20of%20Feature%20Interactions%20via%20Attention%20Networks%20%28ZJU%202017%29.pdf) <br />
- [[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIEN%5D%20Deep%20Interest%20Evolution%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202019%29.pdf) <br />
- [[DSSM] Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (UIUC 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDSSM%5D%20Learning%20Deep%20Structured%20Semantic%20Models%20for%20Web%20Search%20using%20Clickthrough%20Data%20%28UIUC%202013%29.pdf) <br />
- [[FNN] Deep Learning over Multi-field Categorical Data (UCL 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BFNN%5D%20Deep%20Learning%20over%20Multi-field%20Categorical%20Data%20%28UCL%202016%29.pdf) <br />
- [[DeepFM] A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeepFM%5D%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction%20%28HIT-Huawei%202017%29.pdf) <br />
- [[NFM] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BNFM%5D%20Neural%20Factorization%20Machines%20for%20Sparse%20Predictive%20Analytics%20%28NUS%202017%29.pdf) <br />

## Extra Resources

- [Papers on Computational Advertising](https://github.com/wzhe06/Ad-papers) <br />
- [Papers on Recommender System](https://github.com/wzhe06/Ad-papers) <br />
- [CTR Model Based on Spark](https://github.com/wzhe06/SparkCTR) <br />

END
