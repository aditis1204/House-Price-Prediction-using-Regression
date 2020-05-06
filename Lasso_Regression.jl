using CSV
using Statistics
using LinearAlgebra
using DataFrames

data = CSV.read("data\\housingPriceData.csv")

price=data.price
bed=data.bedrooms
bath=data.bathrooms
sqft=data.sqft_living
l=length(price)
l1=Int(floor(0.6*length(price)))
l2= Int(floor(0.2*length(price)))


x0=ones(l)
x = Array{Float64,1}(undef, 5)
stdde = Array{Float64,1}(undef, 5)
x = x.*0

#rmse & r square
function rmse_r(Ypred,Y)
    l=length(Y)
    rmse=((sum((Ypred-Y).^2))/l)^0.5
    print("RMSE is= ",rmse)
    print("\n")
    Ymean=mean(Y) # Y is original price
    r=1-(sum((Ypred-Y).^2))/sum((Y.-Ymean).^2)
    print("R Square is= ",r)
    print("\n")
end



for j in 3:5
    x[j]=mean(data[:,j])
    stdde[j]=std(data[:,j])
end
norm = zeros(l,5)


for i in 3:5
    for j in 1:l
        norm[j,i]=((data[j,i]-x[i])/stdde[i])
        end
end


X = cat(x0, norm[:,3], norm[:,4], norm[:,5], dims=2)

X_train = X[1:l1,:]

X_valid = X[l1+1:l1+l2,:]
X_test = X[l1+l2+1:end,:]
Y_test=price[l1+l2+1:end,:] #test price values
Y_valid=price[l1+1:l1+l2,:]
X_train = X[1:l1,:]


Y_train=price[1:l1]


function costFunction(X, Y, B, lam)
     #l = length(Y)
     cost = (sum(((X * B) - Y).^2)/2) +(sum(broadcast(abs, B)))*lam/2
     return cost
end


# Define a function to perform gradient descent
# 'lam' is the regularization parameter
function gradientDescent(X, Y, B, learningRate, numIterations, lam)
    l = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: 
        for i in 1:4
            if(B[i]==0)
                B[i]=1
            end
        end
        gradient = ((X' * loss).+(lam.*(broadcast(abs, B)./B)))/l   # differential of mod(x) is mod(x)/x
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
    end
    return B
end

#using gd o find regularization coeff.
cost_init=10^20  # random initail cost value
newB =zeros(4, 1)
lambda=0 
lam=10^6
while lam>0.001
     learningRate = 0.01
     # # Initial coefficients
     B = [0.1,0.1,0.1,0.1] 
     B1 = gradientDescent(X_train, Y_train, B, learningRate, 1200,lam) 
     cost=costFunction(X_valid, Y_valid, B1, lam)
        if(cost<cost_init)
            cost_init=cost
            newB=B1
            lambda=lam
        end
     lam=lam/2
end


newb=reshape(newB, :, 1)





Ypred_test=X_test*newb
Ypred_train=X_train*newb
Ypred_all=X*newb




# Visualize the learning: how the loss decreased.
#plot(costHistory)
df = DataFrame()

print("RMSE & R for Test data\n")
rmse_r(Ypred_test,Y_test)
print("RMSE & R for Training data\n")
rmse_r(Ypred_train,Y_train)
#R square value

# Visualize the learning: how the loss decreased.
#plot(costHistory)
df = DataFrame()
Ypred_all=X*newb
CSV.write("data\\2b.csv",  DataFrame(Ypred_all), writeheader=false)


























