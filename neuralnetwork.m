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


vars = data.Properties.VariableNames;
figure
imagesc(ismissing(data))
ax = gca;
ax.XTick = 1:12;
ax.XTickLabel = vars;
ax.XTickLabelRotation = 90;
title('Missing Rows Values')

avgAge = mean(data.age);

figure
histogram(data.age);

data.sex = categorical(data.sex, {'male' 'female'});

figure
histogram(data.sex)
title('Gender Percentage')




