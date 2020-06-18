rgb = imread ('/Users/nomeutente/Desktop/monkeys_project/data_train/n0/n0021.jpg');
imshow(rgb)
histogram2(g,b,'DisplayStyle','tile','ShowEmptyBins','on',...
    'XBinLimits',[0 255],'YBinLimits',[0 255]);
axis equal
colorbar
xlabel('Green Values')
ylabel('Blue Values')
title('Green vs. Blue Pixel Components')
ax = gca;
ax.CLim = [0 500];
