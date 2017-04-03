%*****************************************************************************************************
% Write out plot to png file
%*****************************************************************************************************

function [fnpng]=wrplotpng(fn)

% Print to png file
fnpng=[fn,'.png'];
print('-dpng',fnpng);


% Print to pdf file
fnpdf=[fn,'.pdf'];
print('-dpdf',fnpdf);

% Print to png file
fneps=[fn,'.eps'];
print('-depsc',fneps);


