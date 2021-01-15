%% Lixuan Yang
%%
load dataset2.mat
x=X;
y=Y;
dim=size(y);
N=dim(1);
step=0.1;%step size
w=rand(3,1)/10;%randomly initialized w with values close to zero

counter=0;% count the num of iterations it take to convergence
flag=1;%initialization on the number of misclassified points=binary error

g=zeros(1,N);% binary classify the y data set : 0-->-1, 1-->1
for i=1:N
    if y(i,:)==0
        g(i)=-1;
    else
        g(i)=1;
    end
end


while flag>0% convergence condition
    loss=0;% perceptron loss is reset for each iteration
    flag=0;% binary error is reset for each iteration
    wtemp=0;% weight update accumulation
    for j=1:N% in each iteration, check the misclassified points
        check=g(j)*x(j,:)*w;
        if check<=0% misclassified condition
            wtemp=wtemp+g(j)*(x(j,:)');
            flag=flag+1;%binary error +1 if misclassified
            loss=loss+(-g(j)*x(j,:)*w)%perceptron loss is accumulated for each misclassified point
        end
    end
    if flag>0
        w=w+(step*wtemp)/flag;% gradient descent update for the weight if misclassified points found
    end
    bin_error(counter+1)=flag;% put the binary error for each iteration into an array
    ploss(counter+1)=(1/flag)*loss;% put the perceptron loss for each iteration into an array
    counter=counter+1;% note the number of iterations taken to converge
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
