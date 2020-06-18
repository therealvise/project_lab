rgb = imread ('/Users/nomeutente/Desktop/monkeys_project/data_train/n0/n0021.jpg');
imshow(rgb)

r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);
histogram2(r,g,'DisplayStyle','tile','ShowEmptyBins','on', ...
    'XBinLimits',[0 255],'YBinLimits',[0 255]);
axis equal
colorbar
xlabel('Red Values')
ylabel('Green Values')
title('Green vs. Red Pixel Components')
ax = gca;
ax.CLim = [0 500];
