%% Lixuan Yang 
%%
raw_im = Tiff('trees.tif','r');
im = raw_im.readRGBAImage();
im = im2double(im(1:200,1:200,:));

%show original image
figure(1)
imshow(im);
title('Original')
%make a copy of im
im_2=im;
%number of clusters
k=[5 6 7 8 9 10];

%implement in different # of clusters
for h=1:1:length(k)
    
    %get image RGB dimension
    dim=size(im_2);

    %data center
    c=zeros(k(h),3);
    

    %initialization of data center
    for i=1:k(h)
        a=randi(dim(1));
        b=randi(dim(2));
        c(i,:)=im_2(a,b,:);
    end
    
    %mark the cluster each data point belongs to
    cluster=zeros(200,200);
    %initialize cost
    prev_cost=0;

    %initialize checking criteria
    diff_cost=0.1;
    while diff_cost>0.0001%threshold to stop algorithm
        %within each loop, the cost is re-initialized
        cost=0;
        % count of data points in each cluster
        count=zeros(1,k(h));
        %each cluster, used for calculate cluster sum
        z=zeros(k(h),3);
        for m=1:dim(1)
            for n=1:dim(2)
                %record each datapoint's distances to the k centers
                dis=zeros(1,k(h));
                for j=1:k(h)
                    %calculate each datapoint's distances to the k centers
                    dis(j)=norm(reshape(im_2(m,n,:),[3 1])-c(j,:)').^2;
                end
                %find the nearest data center
                [val,ind]=min(dis);
                %mark the data point's cluster
                cluster(m,n)=ind;
                %add the data point to its class sum
                z(ind,:)=z(ind,:)+reshape(im_2(m,n,:),[1 3]);
                %increment the corresponding class's data points count
                count(ind)=count(ind)+1;
                %total cost is updated with the current data point's min
                %distance
                cost=cost+val;
            end
        end

        %update the mean to be the new data center
        for l=1:k(h)
            c(l,:)=z(l,:)/count(l);
        end

        %check to see the changing position of new data centers
        diff_cost=abs(cost-prev_cost);
        %update total cost to compare later
        prev_cost=cost;
        
        %substitue each data point with its centroid
        for i=1:dim(1)
            for j=1:dim(2)
                im_2(i,j,:)=c(cluster(i,j),:);
            end
        end
    end

    %show k-means processed image
    figure(2)
    subplot(1,length(k),h)
    imshow(im_2);
    title(['k=' num2str(k(h))])
    
    im_2=im;
end






