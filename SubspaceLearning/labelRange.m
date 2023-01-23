function [labels] = labelRange(score,D,cur_trainIndex)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
labels=[];
fi =1;
    for iii=1:size(cur_trainIndex,2)
        ind = find(D.XtB==cur_trainIndex(iii));
        sc = (score(ind));
        
%         sc = (sc./size(sc,2))*10;
        
        for jj=1:size(sc,2)
            if sc>0 & sc<0.05
                labels(fi)=1;
            elseif sc>0.05 & sc<0.1
                labels(fi)=2;
            elseif sc>0.1 & sc<0.15
                labels(fi)=3;
            elseif sc>0.15 & sc<0.2
                labels(fi)=4;
            elseif sc>0.20 & sc<0.25
                labels(fi)=5;
            elseif sc>0.25 & sc<0.30
                labels(fi)=6;
            elseif sc>0.30 & sc<0.35
                labels(fi)=7;
            elseif sc>0.35 & sc<0.40
                labels(fi)=8;
            elseif sc>0.40 & sc<0.45
                labels(fi)=9;
            elseif sc>0.45 & sc<0.5
                labels(fi)=10;
            elseif sc>0.5 & sc<0.55
                labels(fi)=11;
            elseif sc>0.55 & sc<0.60
                labels(fi)=12;
            elseif sc>0.6 & sc<0.65
                labels(fi)=13;
            elseif sc>0.65 & sc<0.7
                labels(fi)=14;
            elseif sc>0.7 & sc<0.75
                labels(fi)=15;
             elseif sc>0.75 & sc<0.8
                labels(fi)=16;
             elseif sc>0.8 & sc<0.85
                labels(fi)=17;
             elseif sc>0.85 & sc<0.9
                labels(fi)=18;
            elseif  sc>0.9 & sc<0.95
                labels(fi)=19;
            elseif sc>0.95 & sc<1
                labels(fi)=20;
            end
            fi=fi+1;
        end
        
    end

end

