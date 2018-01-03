%Q3.1.3
layers=[1024,400,26];
[W,~]=InitializeNetwork(layers);


weights = cell(400, 1);
W = W{1};
% 
for i = 1:400
     r = reshape(W(i, :), 32, 32);
     fileName = sprintf('../result/img%d.png', i);
     imwrite(mat2gray(r), fileName);
     weights{i} = fileName;
 end
 
montage(weights, 'Size', [20 20]);
saveas(gcf, '../result/Initial_weights.png');

load('nist26_model.mat')
W = W{1};
for i = 1:400
     r = reshape(W(i, :), 32, 32);
     fileName = sprintf('../result/img%d.png', i);
     imwrite(mat2gray(r), fileName);
     weights{i} = fileName;
 end
 
montage(weights, 'Size', [20 20]);
saveas(gcf, '../result/Learned_weights.png');

clear all
clc

%Q3_1_4 confusion matrix
load('nist26_model.mat')
load('../data/nist26_test.mat')

[outputs] = Classify(W, b, test_data);
[D,C]=size(outputs);

confusion=zeros(C,C);
%x axis is labels
%y axis is learned labels
for i=1:D
    [~,label]=max(test_labels(i,:));
    [~,learned_label]=max(outputs(i,:));
   confusion(learned_label,label)= confusion(learned_label,label)+1;
end
%scale
for i=1:C
   for j=1:C
      if i~=j
         confusion(i,j)=confusion(i,j)*10;
      end
   end
end
confusion=imresize((mat2gray(confusion)),20);
figure
imshow(confusion)
title('Confusion matrix')
saveas(gcf, '../result/Confusion_matrix.png');
