% Importing Dataset after download it by Kaggle Website
data = readtable('HAM10000_metadata.csv');
head(data,5)

% Statistics about counting for each sensible column
G1 = groupcounts(data,'age');
disp(G1);

G2 = groupcounts(data,'sex');
disp(G2);


G3 = groupcounts(data,'localization');
disp(G3);

% Look about missing values after scan rows
vars = data.Properties.VariableNames;
figure
imagesc(ismissing(data))
ax = gca;
ax.XTick = 1:12;
ax.XTickLabel = vars;
ax.XTickLabelRotation = 90;
title('Missing Rows Values')

avgAge = mean(data.age);
disp(avgAge)

figure
histogram(data.age);

data.sex = categorical(data.sex, {'male' 'female'});

figure
histogram(data.sex)
title('Gender Percentage')

data.localization = categorical(data.localization, {'abdomen' 'acral' 'back' 'chest' 'ear' 'face' 'foot' 'genital' 'hand' 'lower extremity' 'neck' 'scalp' 'trunk' 'unknown' 'upper extremity'});
                                                    
                                                    
                                                    
figure
histogram(data.localization)
title('Localization Percentage')   

[m,n] = size(data);
P = 0.80 ;
idx = randperm(m)  ;
Training = data(idx(1:round(P*m)),:) ; 
Testing = data(idx(round(P*m)+1:end),:) ;



layers = [
    imageInputLayer([75 100 3])
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Testing, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(Training,layers,options);


