# SparrowRecSys
SparrowRecSys是一个电影推荐系统，名字SparrowRecSys（麻雀推荐系统），取自“麻雀虽小，五脏俱全”之意。项目是一个基于maven的混合语言项目，同时包含了TensorFlow，Spark，Jetty Server等推荐系统的不同模块。希望你能够利用SparrowRecSys进行推荐系统的学习，并有机会一起完善它。

## 基于SparrowRecSys的实践课程
受极客时间邀请开设 [深度学习推荐系统实战](http://gk.link/a/10lyE) 课程，详细讲解了SparrowRecSys的所有技术细节，覆盖了深度学习模型结构，模型训练，特征工程，模型评估，模型线上服务及推荐服务器内部逻辑等模块。

## 环境要求
* Java 8
* Scala 2.11
* Python 3.6+
* TensorFlow 2.0+

## 快速开始
将项目用IntelliJ打开后，找到`RecSysServer`，右键点选`Run`，然后在浏览器中输入`http://localhost:6010/`即可看到推荐系统的前端效果。

## 项目数据
项目数据来源于开源电影数据集[MovieLens](https://grouplens.org/datasets/movielens/)，项目自带数据集对MovieLens数据集进行了精简，仅保留1000部电影和相关评论、用户数据。全量数据集请到MovieLens官方网站进行下载，推荐使用MovieLens 20M Dataset。

## SparrowRecSys技术架构
SparrowRecSys技术架构遵循经典的工业级深度学习推荐系统架构，包括了离线数据处理、模型训练、近线的流处理、线上模型服务、前端推荐结果显示等多个模块。以下是SparrowRecSys的架构图：
![alt text](https://github.com/wzhe06/SparrowRecSys/raw/master/docs/sparrowrecsysarch.png)
