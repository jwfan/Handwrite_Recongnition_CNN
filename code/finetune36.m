num_epoch = 5;
classes = 36;
layers = [32*32, 400, classes];
learning_rate = 0.01;

load('../data/nist26_model_60iters')
load('../data/nist36_train.mat', 'train_data', 'train_labels')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
load('../data/nist36_valid.mat', 'valid_data', 'valid_labels')

%modify class to 36
W{end}=[W{end};0.01*rand(10,size(W{end}, 2))];
b{end}=[b{end};zeros(10,1)];

train_acc = zeros(num_epoch, 1);
train_loss = zeros(num_epoch, 1);
valid_acc = zeros(num_epoch, 1);
valid_loss = zeros(num_epoch, 1);
for j = 1:num_epoch
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);

    [train_acc(j), train_loss(j)] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    [valid_acc(j), valid_loss(j)] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);


    fprintf('\nEpoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j,...
        train_acc(j), valid_acc(j), train_loss(j), valid_loss(j))
end

save('nist36_model.mat', 'W', 'b')

% plot accuracy
figure; hold on
title('Accuracy with each epochs')
xlabel('Epoch')
ylabel('Accuracy')
p = plot(1:num_epoch, train_acc'); 
q = plot(1:num_epoch, valid_acc'); 
legend('Training', 'Validation');
saveas(gcf,'../result/Q3_2_1_Accuracy.png')

% plot loss
figure; hold on
title('Loss with each epochs')
xlabel('Epoch')
ylabel('Loss')
p = plot(1:num_epoch, train_loss'); 
q = plot(1:num_epoch, valid_loss'); 
legend('Training', 'Validation');
saveas(gcf,'../result/Q3_2_1_Loss.png')
