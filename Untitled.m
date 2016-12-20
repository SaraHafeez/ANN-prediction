A = 1:1000;  B = 10000*sin(A); C = A.^2  +B;
 Set = [A' B' C'];
 input = Set(:,1:end-1);
 target = Set(:,end);
 inputSeries = tonndata(input(1:700,:),false,false);
 targetSeries = tonndata(target(1:700,:),false,false);

 inputSeriesVal = tonndata(input(701:end,:),false,false);
 targetSeriesVal = tonndata(target(701:end,:),false,false);

 inputDelays = 1:2;
 feedbackDelays = 1:2;
 hiddenLayerSize = 5;
 net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);

[inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);
net.divideFcn = 'divideblock';  % Divide data in blocks
net.divideMode = 'time';  % Divide up every value

 % Train the Network
[net,tr] = train(net,inputs,targets,inputStates,layerStates);
Y = net(inputs,inputStates,layerStates); 

 % Prediction Attempt
delay=length(inputDelays); N=300;
inputSeriesPred  = [inputSeries(end-delay+1:end),inputSeriesVal];
targetSeriesPred = [targetSeries(end-delay+1:end), con2seq(nan(1,N))];
netc = closeloop(net);
[Xs,Xi,Ai,Ts] = preparets(netc,inputSeriesPred,{},targetSeriesPred);
yPred = netc(Xs,Xi,Ai);
perf = perform(net,yPred,targetSeriesVal);

 figure;
plot([cell2mat(targetSeries),nan(1,N);
      nan(1,length(targetSeries)),cell2mat(yPred);
      nan(1,length(targetSeries)),cell2mat(targetSeriesVal)]')
legend('Original Targets','Network Predictions','Expected Outputs')
