# DeepAlpha
##简介
DeepAIpha是个简易的模型训练测试框架。
共三组实验:
phase1: train on [2015, 2016, 2017, 2018]， valid on 2019, test on 2020
phase2: train on [2016, 2017, 2018, 2019], valid on 2020， test on 2021
phase3: train on [2017, 2018, 2019, 2020], valid on 2021, test on 2022
. /config
保存各个实验配置. yaml文件，文件名遵循(MODEL)/(PHASE)/(ID)
/model
保存各种模型的结构，定义
/results
analysis:分析类的图表
ckpts:模型的checkpoints
logs:训练和测试时的1og文件
preds:模型输出预测. feather文件
tsboards:保存训练中的loss, ic等信息，可直接load查看tensorboard
/scale
保存数据预处理中的scale . pk1文件
/data
保存原始. feather文件
/ temp
保存中间数据分别在. /data和./seq data
/util:
一些util和help
functions

##模型训练测试细节
step1:对原始dataframe进行RobustScale, 每一个phase仅对train set fit值并transform到valid, test. 上
对应文件:
./utils/scale . py
step2:生成Torch Dataset等中间文件，训练时直接load到内存
对应文件: . /utils/seq data_ dump. py
dump后的数据保存在. /temp/seq_ data/ (phase)中’
step3:配置超参数和其他实验设置
对应文件夹: .. /Localformer/phase1, 2, 3中
三组实验不同phase模型结构均保持致
step4:运行train. py和test . py即可，文件保存在. /results中
可直接按照model, phase, id查询子文件夹即可
python train.py -C config/Local former/phase1/ Localformer 01 yaml
python test.py -C config/Localformer/phase1/Localformer 01 . yaml

##结果文件
目前最优的模型是. /model/localformer. ts .py,参数配置详见. /config/Localformer/phase*
合并三组实验结果在[ / cpfs/ shared/ zzhao2/new_ localformer preds . feather中
合并后的rcor图表结果在/cpfs/ shared/ zzhao2/DeepAlpha/ report/中
