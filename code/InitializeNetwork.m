function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

% Your code here
% Input dim
N=layers(1);
% Hidden layers size
H=length(layers)-2;
% Output dim
C=layers(end);

W=cell(length(layers)-1,1);
b=cell(length(layers)-1,1);


for i=1:length(layers)-1
    h=layers(i);
    c=layers(i+1);
    b{i}=zeros(c,1);
    W{i}=0.01*randn(h,c)';
end

C = size(b{end},1);

assert(size(W{1},2) == 1024, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'b{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');

end
