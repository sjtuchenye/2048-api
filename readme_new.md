已训练好的模型 [百度云链接:] (https://pan.baidu.com/s/1RPO1_vAOHtxuPKGcykPCgw) 提取码: diyi  
运行环境：CUDA 8.0.61  
cudnn 6.0  
tensorflow 1.2  
keras 2.0.8  
python3.6(python2下也有测试，可以兼容)  
  
    
      



如何训练：  
offline: 运行my_train_offline.py,其中，离线训练过程分为四个模型，分别为0-256，256-512，512-1024，1024-2048四个分层，将第52行的txt文件改为对应的数据文件即可。  
  

online：在my_train_online.py第34-37行load进对应的四个分层模型，运行该文件即可。    
  
  
如何测试：直接运行evaluate.py即可，在这里的agent调用的是game2048/agents.py中的myAgent类。