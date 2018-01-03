% Your code here.
%load('../data/nist26_train.mat');
%load('../data/nist26_test.mat');
%load('../data/nist26_valid.mat');
train_data = rands(1, 1024);      
train_labels = [0, 0, 1, 0, 0];   
N=1024;
layers=[N,128,size(train_labels,2)];
learning_rate=0.01;
epoch=1;
H=length(layers)-2;

[W, b] = InitializeNetwork(layers);

epsilon=0.0001;
L=size(W,1);



for e=1:epoch
   X=train_data(1,:)';
   Y=train_labels(1,:)';
   error1=0;
   error2=0;

   [output,act_h,act_a] = Forward(W,b,X);
   [grad_W,grad_b] = Backward(W,b,X,Y,act_h,act_a);
   % Check W
   for l=1:L
       wsize=size(W{l},1)*size(W{l},2);
       w=randsample(wsize,round(wsize/4));
       [row, col] = ind2sub(size(W{l}),w);
        for i=1:length(row)
                W1=W;
                W2=W;
                W1{l}(row(i),col(i))=W{l}(row(i),col(i))-epsilon;
                W2{l}(row(i),col(i))=W{l}(row(i),col(i))+epsilon;
                [~,loss1]=ComputeAccuracyAndLoss(W1,b,train_data,train_labels);
                [~,loss2]=ComputeAccuracyAndLoss(W2,b,train_data,train_labels);
                error=abs(grad_W{l}(row(i),col(i))-(loss2-loss1)/(2*epsilon));
                error1=error1+error;   
        end
        %Check b
        bsize=length(b{l});
        b_=randsample(bsize,round(bsize/4));
        for q=1:length(b_)
           b1=b;
           b2=b;
           b1{l}(q)=b1{l}(b_(q))-epsilon;
           b2{l}(q)=b2{l}(b_(q))+epsilon;
           [~, loss1] = ComputeAccuracyAndLoss(W, b1, train_data, train_labels);
           [~, loss2] = ComputeAccuracyAndLoss(W, b2, train_data, train_labels);
           
           dJ=abs(grad_b{l}(b_(q))-(loss2-loss1)/(2*epsilon));
           error2=error2+1;
                %fprintf('Layer%d, W:(%d,%d), Error = %f\n',l, i,j,  dJ)
        end
   end
    average_error1=error1/(L*length(row));
    average_error2=error2/(L*length(b_));
    fprintf('Epoch%d,Average error: (W:%d,b:%d)\n', e, average_error1,average_error2)
end