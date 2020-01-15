# Load required packages
using Plots
using DataFrames
using ROCAnalysis
using StatsBase

# Load the preprocessed test images into Xtest, copy into all three channels, and load Xtest onto the GPU
cd("/mnt/juliabox/Retinal-classification/ProcessedTest")
n=length(readdir("/mnt/juliabox/Retinalclassification/ProcessedTest"))
Xtest=zeros(224,224,3,n);
for i=1:n
  I=load("$i_process.jpg")
  Xtest[:,:,1,i]=I
end
Xtest[:,:,2,:]=Xtest[:,:,1,:]
Xtest[:,:,3,:]=Xtest[:,:,1,:]
Xtest=gpu(Xtest)

# Create labels ytest, one-hot encode, and load it onto the GPU
n_diseased=27
ytest=zeros(n)
ytest[1:n_diseased].=1.0
ytest=[Flux.onehot(i,0:1) for i in ytest]
ytest=reduce(hcat,ytest)
ytest=gpu(ytest)

# Perform prediction on test set and find test accuracy, and other results for threshold 0.5
ytestpred=Flux.onecold(new_model(Xtest),0:1)
yactualtest=Flux.onecold(ytest,0:1)
n_healthy=n-n_diseased
pred_diseasedtest=sum(ytestpred.==1)
pred_healthytest=n-pred_diseasedtest
TP=sum((ytestpred.&&yactualtest).==1)
FP=sum((ytestpred.-yactualtest).==1)
TN=sum((ytestpred.||yactualtest).==0)
FN=sum((ytestpred.-yactualtest).==-1)
sens=TP/(TP+FN)
spec=TN/(TN+FP)
FPR=1-spec
accuracy=(TN+TP)/n

# Plot test outputs
preds=new_model(Xtest)
scatter(preds[2,n_diseased+1:end],zeros(n_healthy),markercolor=:blue,label="Healthy",yaxis=false,legend=:bottomright)
scatter!(preds[2,1:n_diseased],zeros(n_diseased),markercolor=:red,label="Diseased",xlabel="NN Outputs")

# Find results for all possible thresholds, put everything into a dataframe, and sort according to threshold
fps=Float64[]
tps=Float64[]
fns=Float64[]
tns=Float64[]
healthy=preds[2,n_diseased+1:end]
diseased=preds[2,1:n_diseased]
for i in preds[2,:]
  fp=sum(healthy.>=i)
  fps=vcat(fps,fp)
  tp=sum(diseased.>=i)
  tps=vcat(tps,tp)
  fn=sum(diseased.<i)
  fns=vcat(fns,fn)
  tn=sum(healthy.<i)
  tns=vcat(tns,tn)
end
accus=(tns .+ tps)./n
senss=tps./(tps .+ fns)
fprs= 1 .- (tns./(tns .+ fps))
df=DataFrame(:Threshhold=>preds,:Accuracy=>accus,:Sensitivity=>senss,:FPRate=>fprs)
sort!(df,[order(:Threshhold)])

# Plot ROC plot and calculate AUC
plot(fprs,senss,xlabel="FP Rate", ylabel="Sensitivity",legend=false)
AUC=auc(roc(healthy,diseased))

# Plot threshold vs. accuracy, sensitivity, and FP rate
plot(df[:Threshhold],df[:Accuracy],xlabel="Threshold",label="Accuracy",legend=:bottom)
plot!(df[:Threshhold],df[:Sensitivity],markercolor=:red,label="Sensitivity")
plot!(df[:Threshhold],df[:FPRate],markercolor=:green,label="FPRate")
plot!([0.545,0.545],[1.00,0.00],label="Threshold=0.545")

# Find accuracy and other results for threshold 0.545
FP=sum(healthy.>=0.545)
TP=sum(diseased.>=0.545)
FN=sum(diseased.<0.545)
TN=sum(healthy.<0.545)
sens=TP/(TP+FN)
spec=TN/(TN+FP)
FPR=1-spec
accuracy=(TN+TP)/n

# Plot shuffling and training times, and find averages
iterations=range(1:10)
plot(iterations,sTime,ylims=(155,158),ylabel="Shufflingtimes",xlabel="Iteration",label="Individual times")
plot!(iterations,fill(StatsBase.mean(sTime),10),label="Averagetime",markercolor=:red)
plot(iterations,tTime,ylims=(25,28),ylabel="Trainingtimes",xlabel="Iteration",label="Individual times")
plot!(iterations,fill(StatsBase.mean(tTime),10),label="Averagetime",markercolor=:red)
avg_sTime=StatsBase.mean(sTime)
avg_tTime=StatsBase.mean(tTime)
