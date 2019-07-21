function [TrainData,TrainLab,TestData,TestLab] = gen_data_random(name, sele_num)
load (name);

nnClass = length(unique(gnd));                        
num_Class=[];
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];    
end
TrainData  = [];
TrainLab = [];
TestData   = [];
TestLab  = [];
for j = 1:nnClass
    idx=find(gnd==j);
    randIdx  = randperm(num_Class(j));  
    TrainData = [TrainData; fea(idx(randIdx(1:sele_num)),:)]; 
    TrainLab= [TrainLab;gnd(idx(randIdx(1:sele_num)))];
    TestData  = [TestData;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  
    TestLab = [TestLab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
TrainData = TrainData';
TestData  = TestData';
Save_Train_Ma =TrainData;
TrainData = NormalizeFea(TrainData);
TestData = NormalizeFea(TestData);

