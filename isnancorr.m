function [r] = isnancorr(x,y)

iwant=find(~isnan(x) & ~isnan(y));
disp(['Dropped ' num2str(length(x)-length(iwant)) ' NaN values'])
r=corr(x(iwant),y(iwant));