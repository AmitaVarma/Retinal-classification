using Flux
using images

X=zeros(89,1_728_000)
R=readdir("/mnt/juliabox/Retinal-classification/Processed Images")
for i =2:90
    I=Float64.(load(R[i]));
    X[i-1,:]=(vec(I))';
end
Y=zeros(89,1);
for i=1:52
    Y[i]=1.0;
end


n = length(vec(images[1]))
model = Chain(Dense(n, 2, relu),softmax)
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM(params(model))

dataset = Base.Iterators.repeated((X, Y), 100)
for i in 1:100
    Flux.train!(loss, dataset, opt; cb = Flux.throttle(callback, 1))
end    
