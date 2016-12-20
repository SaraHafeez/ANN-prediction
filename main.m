clc
format long
[data]=xlsread('data2');

T=data(9,:);
X=data(1:8,:);
 inputDelays = 1:2;
 feedbackDelays = 1:2;
 hiddenLayerSize = 21;
 inputs=num2cell(X,1);
 T=num2cell(T,1);
 net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
[Xs,Xi,Ai,Ts] = preparets(net,inputs,{},T); 
net = train(net,Xs,Ts,Xi,Ai); 
view(net) 
Y = net(Xs,Xi,Ai);
plotresponse(Ts,Y)


