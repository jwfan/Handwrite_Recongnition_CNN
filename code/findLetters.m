function [lines, bw] = findLetters(im)
% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.


%Your code here
im= rgb2gray(im);
bw=imcomplement(imbinarize(im));
bw = imdilate(bw, strel('disk', 5));
%imshow(bw);
[M,N]=size(bw);
connected=bwconncomp(bw);
n=connected.NumObjects;
c=connected.PixelIdxList;
o=[];
padding=30;
for i=1:n
    x1=N;
    y1=M;
    x2=0;
    y2=0;
    for j=1:length(c{i})
        [row,col] =ind2sub([M,N], c{i}(j));
        x2 = max(x2, col);
        x1 = min(x1, col);
        y2 = max(y2, row);
        y1 = min(y1, row);
    end
    
    if abs((x2-x1)*(y2-y1))>500
       padding=(x2-x1)*0.1;
       [x1, y1] = deal(max(1, x1 - padding), max(1, y1 - padding));
       [x2, y2] = deal(min(N, x2 + padding),min(M, y2 + padding));
       if abs((x2-x1)-(y2-y1))>25
            if (x2-x1)<(y2-y1)
                x2=min(N,x2+((y2-y1)-(x2-x1))/2);
                x1=max(1,x1-((y2-y1)-(x2-x1))/2);
            else
                y2=min(M,y2+((x2-x1)-(y2-y1))/2);
                y1=max(1,y1-((x2-x1)-(y2-y1))/2);
            end
       end
       o=[o;x1,y1,x2,y2];
       
       %figure
       %imshow(bw(y1:y2,x1:x2));
    end
    
    
end

lines = cell(size(o, 1), 1);
i=1;
while length(o)~=0
    [ymin,index]=min(o(:,2));
    ymax=o(index,4);
    oneline= (o(:,2) >= ymin) & (o(:,2) <= ymax);
    if length(oneline)~=0
        onelines=sortrows(o(oneline,:));
        lines{i}=onelines;
    end
    i=i+1;
    o(oneline,:)=[];
end
lines = lines(1:i - 1);
bw=~bw;

assert(size(lines{1},2) == 4,'each matrix entry should have size Lx4');
assert(size(lines{end},2) == 4,'each matrix entry should have size Lx4');
lineSortcheck = lines{1};
assert(issorted(lineSortcheck(:,1)) | issorted(lineSortcheck(end:-1:1,1)),'Matrix should be sorted in x1');

end
