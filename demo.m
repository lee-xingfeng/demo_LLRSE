warning('off');
clc
clear memory;
%addpath('./lib');


[TrainData,TrainLab,TestData,TestLab]=gen_data_random('YaleB_32x32',10);

lambda1 = 0.1;%Q
lambda2 = 0.01;%E
lambda3 = 0.0001;%Z

[P,Q,Z,E,H,obj] = LLRSE(TrainData,TrainLab,lambda1,lambda2,lambda3);

Test_Maa  = Q'*TestData;
Test_Maa = NormalizeFea(Test_Maa,0);
Train_Maa = Q'*TrainData;
Train_Maa = NormalizeFea(Train_Maa,0);
[pred] = knnclassify(real(Test_Maa'),real(Train_Maa'),TrainLab,1,'euclidean','nearest');
acc_test     = sum(TestLab == pred)/length(TestLab)*100

