function [Indn] = labeling(ligne)

%ligne   : the original signal
ligne=ligne';
L=length(ligne);
M= floor(log2(L));

%computations of number of averaging steps
if (M>=7)
    niter=4;
elseif(M>=4)   
    niter=2;
else 
    niter=0;
end

if(niter==0)
ligne_cour = ligne;
res = abs(diff(ligne_cour));
H = zeros(1,L-2); 
H = res(1:end-1)+res(2:end);
 Ind = [];
    for i = 4:L-3, 
     if (res(i) > max(res(i+1:i+2))) %&& (res(i-1) > max(res(i-3:i-2)))
      Ind = [Ind i];      
     end 
    end
    Indn=Ind;
else    
%nbpoint : number of points for the initial neigbourhood 
nbpoint=4;
% profondeur minimale d'estimation des singularites
% calculé à partir de L0  et niter

ligne_cour = ligne;
label      = zeros(niter,length(ligne));
 
 for k = 1:niter,
    
    L = length(ligne_cour);
    H = zeros(1,L-2); 
    
    %computation of the first order differences
  
    res = abs(diff(ligne_cour));
   
    H = res(1:end-1)+res(2:end);
    Ind = [];
    for i = 2+3*(nbpoint-k+1):L-2-3*(nbpoint-k+1), 
     if (H(i) > max(H(i+1:i+3*(nbpoint-k+1))) && (H(i-1) > max(H(i-1-3*(nbpoint-k+1):i-2)))) 
      Ind = [Ind i];      
     end 
    end
 
    %There might be several neighbouring intervals of interest   
    %the location of the singularities of interest is written in Indn 
    
    q = 1;
    Indn=[];
  
    while (q <= length(Ind)-1) 
     if Ind(q)+1 == Ind(q+1),
       if (H(Ind(q+1)) >= H(Ind(q)))        
         Indn = [Indn Ind(q+1)];
       else
         Indn = [Indn Ind(q)];
       end
       q = q+2;
     else
      Indn = [Indn Ind(q)];   
      q = q+1;
     end
    end
   
    if ( q == length(Ind))
     Indn = [Indn Ind(q)];
    end
    
    label(k,Indn)    = ones(1,length(Indn));
     
    
    ligne_cour = (ligne_cour(1:2:end-1)+ligne_cour(2:2:end))/2;
    
 
 end 
 
 
 %we build the chain lines

 Xint = (1:length(ligne)).*label(niter,:);
 Xint = 2^(niter-1).*Xint(Xint > 0);
 Xsup = (1:length(ligne)).*label(1,:);
 Xsup = Xsup(Xsup > 0);
  
 N    = length(Xint);
 %number of steps at the coarse scale
 Ns    = length(Xsup);
 %number of steps at the fine scale
 
 %redefiniton of step indexes 
 
 Indn=[];
 
 for i=1:N
     for j=1:Ns
     if (abs(Xint(i)-Xsup(j))<2*niter)
        Indn = [Indn Xsup(j)];
     end
     end
 end
 

end  
 
 
end 


