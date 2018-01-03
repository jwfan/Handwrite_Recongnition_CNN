function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the text contained in the image as a string.

im=imread(fname);
[lines, bw] = findLetters(im);

N=0;
for i=1:length(lines)
   N=N+size(lines{i},1); 
end

data=zeros(N,1024);
dataindex=1;
line_enter=[0];
for i=1:length(lines)
   line=lines{i};
   n=size(line,1);
   for j=1:n
       o=line(j,:);
       letter=~bw(o(2):o(4), o(1):o(3));
       letter=~imresize(letter,[32,32]);
       %imshow(letter);
       data(dataindex,:)=letter(:)';
       dataindex=dataindex+1;
   end
   t=n+line_enter(end);
   line_enter=[line_enter;t];
end
load('nist36_model.mat');
[outputs] = Classify(W, b, data);
label = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
text = '';
enter_index=2;
for i=1:N
    labels=outputs(i,:);
    [~, index] = max(labels);
    text=[text label(index)];
    if i==line_enter(enter_index)
       text=[text '\n'];
       enter_index=enter_index+1;
    end
end
end
