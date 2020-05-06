using CSV
using Statistics
using LinearAlgebra
using DataFrames
data = CSV.read("data\\housingPriceData.csv")

#reading values
price=data.price
bed=data.bedrooms
bath=data.bathrooms
sqft=data.sqft_living
l=length(price)
l1=Int(floor(0.8*length(price)))


# finding mean and std devn.
x0=ones(l)
x = Array{Float64,1}(undef, 5)
stdde = Array{Float64,1}(undef, 5)
x = x.*0

for j in 3:5
    x[j]=mean(data[:,j])
    stdde[j]=std(data[:,j])
end
standa = zeros(l,5)


for i in 3:5
    for j in 1:l
        standa[j,i]=((data[j,i]-x[i])/stdde[i])
        end
end

X = cat(x0, standa[:,3], standa[:,4], standa[:,5], dims=2)
X_train = X[1:l1,:]
X_test = X[l1+1:end,:]
Z=price[l1+1:end,:] #test price values
Y=price[1:l1]

#function for rms and r square
function rmse_r(Yhat,Y)
	l=length(Y)
    rmse=((sum((Yhat-Y).^2))/l)^0.5
    print("RMSE is= ",rmse)
    print("\n")
    Ymean=mean(Y) # Y is original price
    r=1-(sum((Yhat-Y).^2))/sum((Y.-Ymean).^2)
    print("R Square is= ",r)
    print("\n")
end

# Define a function to calculate cost function
function costFunction(X, Y, B)
    #l = length(Y)
    cost = sum(((X* B) - Y).^2)/(2*l1)
    return cost
end


# # Initial coefficients
B = zeros(4, 1)
# Calcuate the cost with intial model parameters B=[0,0,0]
intialCost = costFunction(X_train, Y, B)


function gradientDescent(X_train, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X_train * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X_train' * loss)/l1
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X_train, Y, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

learningRate = 0.01
newB, costHistory = gradientDescent(X_train, Y, B, learningRate, 1200)
Yhat_test=X_test*newB
Yhat_train=X_train*newB
Yhat_all=X*newB
# rms & rsquare
print("RMSE & R  Test data\n")
rmse_r(Yhat_test,Z)

df = DataFrame()

CSV.write("data\\1a.csv",  DataFrame(Yhat_all), writeheader=false)

