%Q3.2.2
clear all
clc
layers = [32*32, 400, 36];
load('../data/nist26_model_60iters')
W{end}=[W{end};rand(10,size(W{end}, 2))];



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
saveas(gcf, '../result/Initial_weights_finetuned.png');

load('nist36_model.mat')
W = W{1};
for i = 1:400
     r = reshape(W(i, :), 32, 32);
     fileName = sprintf('../result/img%d.png', i);
     imwrite(mat2gray(r), fileName);
     weights{i} = fileName;
 end
 
montage(weights, 'Size', [20 20]);
saveas(gcf, '../result/Learned_weights_finetuned.png');

clear all
clc

%Q3_2_3 confusion matrix
load('nist36_model.mat')
load('../data/nist36_test.mat')

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
saveas(gcf, '../result/Confusion_matrix_finetuned.png');
