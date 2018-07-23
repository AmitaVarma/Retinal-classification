using Flux
using images

# code to load data from txt files

n = length(vec(images[1]))
model = Chain(Dense(n, 2, relu),softmax)
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM(params(model))

dataset = Base.Iterators.repeated((X, Y), 100)
for i in 1:100
    Flux.train!(loss, dataset, opt; cb = Flux.throttle(callback, 1))
end    
