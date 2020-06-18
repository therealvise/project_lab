rgb = imread ('/Users/nomeutente/Desktop/monkeys_project/data_train/n0/n0021.jpg');
imshow(rgb)
histogram2(r,b,'DisplayStyle','tile','ShowEmptyBins','on',...
    'XBinLimits',[0 255],'YBinLimits',[0 255]);
axis equal
colorbar
xlabel('Red Values')
ylabel('Blue Values')
title('Blue vs. Red Pixel Components')
ax = gca;
ax.CLim = [0 500];
