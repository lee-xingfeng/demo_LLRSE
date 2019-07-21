warning('off');
clc
clear memory;
%addpath('./lib');


[Train_Ma,Train_Lab,Test_Ma,Test_Lab]=gen_data_random('YaleB_32x32',10);

lambda1 = 0.1;%Q
lambda2 = 0.001;%E
lambda3 = 0.0001;%Z

[P,Q,Z,E,H,obj] = LLRSE(Train_Ma,Train_Lab,lambda1,lambda2,lambda3);

Test_Maa  = Q'*Test_Ma;
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);
Train_Maa = Q'*Train_Ma;
Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
[pred] = knnclassify(real(Test_Maa'),real(Train_Maa'),Train_Lab,1,'euclidean','nearest');
acc_test     = sum(Test_Lab == pred)/length(Test_Lab)*100

