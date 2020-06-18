% In questo script andiamo ad effettuare un'analisi sui colori.
% Nello specifico elaboriamo l'immagine a colori "n0021.png" raffigurante un esempio di scimmia appartenente
% alla specie 'Alouatta Palliata' tracciando un istogramma bivariato dei valori RGB Rosso e Verde per ogni pixel,
% per visualizzare la distribuzione del colore.

rgb = imread ('/Users/nomeUtente/Desktop/monkeys_project/training/n0/n0021.jpg');
imshow(rgb)

r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);
histogram2(r,g,'DisplayStyle','tile','ShowEmptyBins','on', ...
    'XBinLimits',[0 255],'YBinLimits',[0 255]);
axis equal
colorbar
xlabel('Valori Rosso')
ylabel('Valori Blu')
title('Componenti a pixel Verde vs. Rosso')
ax = gca;
ax.CLim = [0 500];
