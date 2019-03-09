对论文《An Experimental Study on Speech Enhancement Based on Deep Neural Networks》的复现
1.用matlab运行get_data_mat.m,处理htk格式的对数功率谱文件，得到相应的.mat（scipy.io可读取）文件
2.将得到的.mat文件的10%用作测试集，90%用作训练集
3.运行PreTrain.py,用noisy_data对4层的神经网络（包括一个Gaussian-Bernoulli RBM 和两个Bernoulli-Bernoulli RBM）进行预训练
4.运行trainDnn.py,对整个神经网络（5层）进行微调
