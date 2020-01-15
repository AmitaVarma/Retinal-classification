# Load the required packages
using Images
using CuArrays
using Flux: gpu
using Flux
using Metalhead
using Random
import Random:randperm

# Create the model and load it onto the GPU
model=VGG19()
model_last=Chain(Dense(1000,200),Dense(200,50),Dense(50,2),softmax)
model_start=model.layers[1:end-1]
new_model=Chain(model_start, model_last)
new_model=gpu(new_model)
Flux.testmode!(new_model,false)

# Load the preprocessed training images into X, copy into all three channels, and load X onto the GPU
cd("/mnt/juliabox/Retinal-classification/ProcessedTrain")
n=length(readdir("/mnt/juliabox/Retinalclassification/
ProcessedTrain"))
X=zeros(224,224,3,n);
for i=1:n
    I=load("$i_process.jpg")
    X[:,:,1,i]=I
end
X[:,:,2,:]=X[:,:,1,:]
X[:,:,3,:]=X[:,:,1,:]
X=gpu(X)
# Create labels y, one-hot encode, and load it onto the GPU
n_diseased=165
y=zeros(n)
y[1:n_diseased].=1.0
y=[Flux.onehot(i,0:1) for i in y]
y=reduce(hcat,y)
y=gpu(y)

# Define loss, optimizer, and parameters
loss(x,y)=Flux.crossentropy(new_model(x),y)
opt=ADAM()
model_last=gpu(model_last)
ps = params(model_last)
callback() = @show(loss(Xbatch,y[:,1:10]))

# Shuffle and train in batches, 10 times. Save times in sTime and tTime
# For shuffling:
sTime=zeros(10)
tTime=zeros(10)
for c=1:10
    r=randperm(n)
    sTime[c]=@timev begin
        Xshuf=zeros(224,224,3,n)
        yshuf=zeros(2,n)
        for i=1:n
            Xshuf[:,:,:,:,i]=X[:,:,:,r[i]]
            yshuf[:,i]=y[:,r[i]]
        end
    end
    # For training in batches
    tTime[c]=@timev begin
        Xbatch=Xshuf[:,:,:,1:9]
        ybatch=yshuf[:,1:9]
        Flux.train!(loss,ps,Iterators.repeated((Xbatch,ybatch)
        ,10),opt)
        for i=1:8
            r_s=10*i
            r_e=(10*i)+9
            Xbatch=Xshuf[:,:,:,r_s:r_e]
            ybatch=yshuf[:,r_s:r_e]
            Flux.train!(loss,ps,Iterators.repeated((Xbatch,ybatch),10),opt)
        end
    end
end

# Perform prediction on training set and find training accuracy, and other results
ypred=Flux.onecold(new_model(X),0:1)
yactual=Flux.onecold(y,0:1)
n_healthy=n-n_diseased
pred_diseased=sum(ypred.==1)
pred_healthy=n-pred_diseased
TP=sum((ypred.&&yactual).==1)
FP=sum((ypred.-yactual).==1)
TN=sum((ypred.||yactual).==0)
FN=sum((ypred.-yactual).==-1)
sens=TP/(TP+FN)
spec=TN/(TN+FP)
FPR=1-spec
accuracy=(TN+TP)/n

