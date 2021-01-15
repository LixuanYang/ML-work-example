%% Lixuan Yang
%%
load dataset2.mat
x=X;
y=Y;
dim=size(y);
N=dim(1);
w=rand(3,1)/10;%randomly initialized w with values close to zero

counter=0;% count the num of iterations it take to convergence

g=zeros(1,N);% binary classify the y data set : 0-->-1, 1-->1
for i=1:200
    if y(i,:)==0
        g(i)=-1;
    else
        g(i)=1;
    end
end

flag=false;% initialize the flag to false to trigger the while loop

while ~flag% check convergence condition
    loss=0;%perceptron loss initialized to zero
    h=x*w;%predicted binary y
    flag=true;%change flag to true to exit the while loop
    for j = 1:N
        if h(j) > 0
            h(j) = 1;
 
        else
            h(j) = -1;
        end
        error(j)=abs((h(j)-g(j)));%sum up binary error for each point
    end
    bin_error(counter+1)=sum(error);%total binary error for each iteration
    
    rand200=1:N;%vector of 1-200
    rand_ind = rand200(randperm(length(rand200)));%shuffle the vector so that each index is different
    for k = 1:N
        check=g(rand_ind(k))*x(rand_ind(k),:)*w;%misclassified condition check
        if check <= 0%if misclassified
            w = w+g(rand_ind(k))*x(rand_ind(k),:)';  %updating w once one misclassfication is found
            flag=false;%change flag to false to continue the while loop
            loss=loss+(-g(k)*x(k,:)*w)%perceptron loss is calculated for each misclassified point
            break %Breaking out loop because we found the first missclassfied point
        end
    end
    ploss(counter+1)=loss;%the perceptron error is the one misclassified point error
    counter=counter+1;%increment counter to note the number of iterations

end

%%
counter

figure(1)
n=1:counter;
subplot(2,1,1)%plot perceptron loss vs. iteration index
plot(n,ploss)
title('Perceptron Error')
xlabel('iteration index')

subplot(2,1,2)%plot binary error vs. iteration index
plot(n,bin_error)
title('Binary Classification Error')
xlabel('iteration index')

figure(2)%linear decision boundary
plot(x(:,1),x(:,2),'x')% generate x data plane
hold on
plot(x(:,1),((-w(1)*x(:,1))-w(3)*x(:,3))/w(2),'r')% linear decision boundary
hold off
