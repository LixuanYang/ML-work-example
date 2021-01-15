%% Lixuan Yang


load('datasetPCA.mat');%load image data

x=teapotImages;%rename image data
dim=size(x);%get dimension of image data

mean=zeros(1,dim(2));%initiate mean with 0
cov=zeros(dim(2));%initiate covariance with 0

c=3;%number of steps interested

%calculate mean
for i=1:dim(2)
     mean(i)=sum(x(:,i))/dim(1);
end

%calculate covariance
for j=1:dim(1)
    cov=cov+(x(j,:)-mean)'*(x(j,:)-mean); 
end
cov=cov/dim(1);

%eigenvector decomposition
[eigvec,eigval]=eig(cov);

%find the top 3 eigenvalues and their corresponding indices
[maxval,ind]=maxk(diag(eigval),c);

%retieve the top3 corresponding eigenvectors
topc=eigvec(:,ind)';

%PCA
x_new=zeros(dim(1),dim(2)); %initialize new image data

for m=1:dim(1)
    coeff=zeros(1,3);%update the step for each image
    coeff_sum=zeros(1,dim(2));%offset of the compressed data,update for each compressed image
    for n=1:c% calculate the 3 steps in 3 different directions
        coeff(n)=(x(m,:)-mean)*eigvec(:,ind(n));%calculate one particular step
        coeff_sum=coeff_sum+coeff(n)*(eigvec(:,ind(n))');%sum up steps
    end
    x_new(m,:)=mean+coeff_sum;%add calculated offset to the mean to get compressed image
end

%%

%show mean as image
figure(1)
imagesc(reshape(mean(1,:),38,50));
title('mean')
colormap gray;

%show top 3 eigvect as images
figure(2)
imagesc(reshape(topc(1,:),38,50));
title('1st eigenvector')
colormap gray;
figure(3)
imagesc(reshape(topc(2,:),38,50));
title('2nd eigenvector')
colormap gray;
figure(4)
imagesc(reshape(topc(3,:),38,50));
title('3rd eigenvector')
colormap gray;

%%
%show difference on 10 random images

figure(5)

for i=1:10
    
    image=randi(dim(1));
    subplot(2,10,i)
    imagesc(reshape(teapotImages(image,:),38,50));
    title([num2str(image) 'Before']);
    colormap gray;
    
    subplot(2,10,i+10)
    imagesc(reshape(x_new(image,:),38,50));
    title([num2str(image) 'Compressed']);
    colormap gray;
end
