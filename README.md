# 学习记录
  本人是零基础接触深度学习，没有找到《动手学习深度学习》，使用了清华大学出版社出版的《深度学习理论与应用》，通过几天的学习对神经网络有了一个大概的了解，在做题时同时借鉴书中的例子并在网上进行学习。

## <font size="9">1.全连接神经网络</font>
  这个部分花费了近4天的时间，主要是对神经网络先有一个大致的了解。在着手解题时首先遇到的问题时怎么导入数据，翻看资料后，采用下载MNIST数据集的方法:  
 `#加载MNIST数据集  `  
`train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)`  
`test_dataset = datasets.MNIST('./data', train=False, transform=transform)`  
 并将批量设置成64.
