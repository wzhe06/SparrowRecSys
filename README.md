# SparrowRecSys
SparrowRecSys是一个电影推荐系统，名字SparrowRecSys（麻雀推荐系统），取自“麻雀虽小，五脏俱全”之意。项目是一个基于maven的混合语言项目，同时包含了Spark，Jetty Server，TensorFlow等推荐系统的不同模块。希望你能够利用SparrowRecSys进行推荐系统的学习，并有机会一起完善它。

# 项目数据
项目数据来源于开源电影数据集[MovieLens](https://grouplens.org/datasets/movielens/)，项目自带数据集对MovieLens数据集进行了精简，仅保留1000部电影和相关评论、用户数据。全量数据集请到MovieLens官方网站进行下载，推荐使用MovieLens 20M Dataset。

# SparrowRecSys技术架构
SparrowRecSys技术架构遵循经典的工业级深度学习推荐系统架构，包括了离线数据处理、模型训练、近线的流处理、线上模型服务、前端推荐结果显示等多个模块。以下是SparrowRecSys的架构图：


