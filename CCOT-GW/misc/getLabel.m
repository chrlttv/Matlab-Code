function [partition]=getLabel(idx_edges,idxb)
d=length(idxb);
% Get a label for variables 
      k=1;
         label_tr=[];
         if (length(idx_edges)>1)
             for i1 = 1:(length(idx_edges)-1)
                 label_tr(idx_edges(k)+1:idx_edges(i1+1)) = i1;
                 k = k + 1;
             end;
             label_tr = [label_tr, (i1+1)*ones(1,d-idx_edges(end))];
             label_tr=label_tr+1;
         else 
             label_tr(1:idx_edges(k)) = 1;
             label_tr(idx_edges(k)+1:d) = 2;
         end;
         [~,idxbb]=sort(idxb);
         partition=label_tr(idxbb);
end

 