watch --color -n1 gpustat -cpu
PS1="\[\e]0; \u@\h: \w\a\]$ {debian chroot :+($debian_ chroot)}\[ \033[01 ;32m\]\u@\h[ \033[ 00m\]:\[ \033[01;34m\]\w\[ \033[00m\]\$"
export PS1
git clone https://gi thub . com/zijzhao1996/ codebase . git
# Linear
为base.
ine表现不错， 整体正向cos sim都保持在0.05水平，只有2021年的结果稍微偏了一些
# MLP
t XGboost 
# ALSTM
01:双层，普通1oss 
02:双云，符殊1oSS
4:层，普通loss, 最好positive Cosine similarity: 0.0648
# GATS
最后使用了ALSTM的一层结构，其他均保持不变
# Localformer
01是使用了最新的数据,但03最好，是phase1
84是03的相同架构，为phase2
06是03的相同架构，为phase3
# TabNet
03是使用了最新的数据,phase1 ic大概是0.06
04是phase2
05是pahse3
放用卡奶
# Linear
01使用的是BN+-层， 效果为0. 0576，02使用的时候是把batch norm 去掉了0. 0569
效果非常好#TODO
03, 04仅是为了检查框架的可复现性，之后直接删除
# MLP
01为4层，但是感觉过拟合了，效果为0.039 #TODO
# localfomer
phase 1的03应该是没用的
正在train的是localformer的phase2 ,和Ultraformwer的phase1和phase2
TODO
summarize all .pynb to .Py and re-org
add train to each model
修改所有. py里写死的绝对路径
Informerde parse attention 还不能用
local former embedding name change
MTLformer训练还是有问题
# total params = sum(item.nume1() for item in 1ist(mode1. parameters()))

