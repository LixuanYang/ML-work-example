load dataset1.mat
D=16;%polynomial order plus 1

for l=0:1:10000%lambda range
    
    for i=1:1:10%cross-validation with 10 groups
        xT=(x(50*(i-1)+1:50*(i-1)+1,:))';%xTesting Group, transposed
        xN=[(x(1:50*(i-1),:))',(x(((50*i)+1):500))'];%xTraining Group
        yT=y(50*(i-1)+1:50*(i-1)+1,:);%yTesting Group, transposed
        yN=[y(1:50*(i-1),:);y(((50*i)+1):500)];%yTraining Group

        %form the xTraining matrix
        xx = zeros(D,length(xN));
        for j=1:D
          xx(j,:) = xN.^(D-j);
        end
        model = inv((xx*xx'+l*eye(16))')*xx*yN;%calculated coeff, w
        %note the difference in the equation from my attached notes
        %because I transpose x earlier, to get the right size of w, I
        %switch xx' and xx position in the parenthesis
        err(i) = (1/2)*sum((xx'*model-yN).^2)+(l/2)*(model'*model);%calculated training error using w
        
        %form the xTest matrix
        xxT = zeros(D,length(xT));
        for j=1:D
          xxT(j,:)=xT.^(D-j);
        end
        errT(i)=(1/2)*sum((xxT'*model-yT).^2)+(l/2)*(model'*model);%calculated testing error using w
    end

    erravg(l+1) = mean(err);%average the training errors for this specific lambda
    errTavg(l+1)= mean(errT);%average the testing errors for this specific lambda
end
%%
l=0:10000;
%Training Error Plot
figure(1)
plot(l,erravg)
xlabel('Lambda');
ylabel('Error')
title('Training Error');
%Testing Error Plot
figure(2)
plot(l,errTavg)
xlabel('Lambda');
ylabel('Error');
title('Testing Error');

%To choose the lambda for the smallest testing error, I choose lambda=7300.
%since as lambda increases, the testing error decreases to its lowest and 
%training error plateaus.

l(7301)%lambda
erravg(7301)%the training error given the selected lambda
errTavg(7301)%the testing error given the selected lambda
        