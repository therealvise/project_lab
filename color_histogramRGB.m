
% In questo script andiamo ad effettuare un'analisi sui colori.
% Nello specifico elaboriamo l'immagine a colori "n0021.png" raffigurante un esempio di scimmia appartenente
% alla specie 'Alouatta Palliata'.
% Confermiamo i risultati creando un istogramma dei colori nello spazio di colore RGB.

rgb = imread ('/Users/nomeutente/Desktop/monkeys_project/data_train/n0/n0021.jpg');
imshow(rgb)
histogram(r,'BinMethod','integers','FaceColor','r','EdgeAlpha',0,'FaceAlpha',1)
hold on
histogram(g,'BinMethod','integers','FaceColor','g','EdgeAlpha',0,'FaceAlpha',0.7)
histogram(b,'BinMethod','integers','FaceColor','b','EdgeAlpha',0,'FaceAlpha',0.7)
xlabel('Valore RGB')
ylabel('Frequenza')
title('Istogramma dei colori nello spazio di colore RGB')
xlim([0 257])
