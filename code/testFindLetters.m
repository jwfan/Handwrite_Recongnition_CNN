% Your code here.
%im=imread('../images/01_list.jpg');
%im=imread('../images/02_letters.jpg');
%im=imread('../images/03_haiku.jpg');
im=imread('../images/04_deep.jpg');

[lines, bw] = findLetters(im);
for i = 1:length(lines)
   line = lines{i}; 
   [num, ~] = size(line);
   for j = 1:num
       letter = line(j, :);
       width=letter(3) - letter(1);
       height=letter(4) - letter(2);
       im = insertShape(im,'Rectangle',[letter(1), letter(2), width, height],'LineWidth', 6);
   end
end
figure
imshow(im);
saveas(gcf, '../result/Q4_2_04.png');
figure
imshow(bw);