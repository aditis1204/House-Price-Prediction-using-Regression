using CSV
using Statistics
using LinearAlgebra
using DataFrames
data = CSV.read("data\\housingPriceData.csv")



price=data.price
bed=data.bedrooms
sqft=data.sqft_living
bedsq=bed.*bed
sqftsq=sqft.*sqft
bed_sqft=bed.*sqft
l=length(price)
l1=Int(floor(0.8*length(price)))
df = [data[:,3] data[:,5] bedsq sqftsq bed_sqft]
l1=Int(floor(0.8*length(price)))


#creating arrays for mean and std devn
x0=ones(l)
x = Array{Float64,1}(undef, 5)
stdde = Array{Float64,1}(undef, 5)
x = x.*0

for j in 1:5
    x[j]=mean(df[:,j])
    stdde[j]=std(df[:,j])
end

#standadizing values
standard = zeros(l,5)
for i in 1:5
    for j in 1:l
        standard[j,i]=((df[j,i]-x[i])/stdde[i])
    end
end


#new input feature vector
X = cat(x0, standard[:,1], standard[:,2], standard[:,3], standard[:,4], standard[:,5], dims=2)


Y=price
X_train=X[1:l1,:]
Y_train=Y[1:l1,:]
X_test=X[l1+1:end,:]
Y_test=Y[l1+1:end,:]

# Define a function to calculate cost function
function costFunction(X, Y, B)
    l = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*l)
    return cost
end
# to find rmse and r square
function rmse_r(Ypred,Y)
    l=length(Y)
    rmse=((sum((Ypred-Y).^2))/l)^0.5
    print("RMSE = ",rmse)
    print("\n")
    Ymean=mean(Y) # Y is original price
    r=1-(sum((Ypred-Y).^2))/sum((Y.-Ymean).^2)
    print("R Square = ",r)
    print("\n")
end
# # Initial coefficients
B = zeros(6, 1)
# Calcuate the cost with intial model parameters B=[0,0,0]
intialCost = costFunction(X, Y, B)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    l = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X' * loss)/l
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

learningRate = 0.01
newB, costHistory = gradientDescent(X, Y, B, learningRate, 1100)
# calculaing new values
Yhat_train= X_train*newB
Yhat_test= X_test*newB
Yhat_all=X*newB
#CSV.write("Path where your CSV file will be stored\\File Name.csv", df)#
#RMS CALCULATION

print("RMSE & R  Test data\n")
rmse_r(Yhat_test,Y_test)

df = DataFrame()

CSV.write("data\\1b.csv",  DataFrame(Yhat_all), writeheader=false)





newB


