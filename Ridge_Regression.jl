using CSV
using Statistics
using LinearAlgebra
using DataFrames
data = CSV.read("data\\housingPriceData.csv")

price=data.price
bed=data.bedrooms
bath=data.bathrooms
sqft=data.sqft_living
m=length(price)
m1=Int(floor(0.6*length(price)))
m2= Int(floor(0.2*length(price)))




x0=ones(m)
x = Array{Float64,1}(undef, 5)
stdde = Array{Float64,1}(undef, 5)
x = x.*0




for j in 3:5
    x[j]=mean(data[:,j])
    stdde[j]=std(data[:,j])
end
norm = zeros(m,5)


for i in 3:5
    for j in 1:m
        norm[j,i]=((data[j,i]-x[i])/stdde[i])
        end
end


X = cat(x0, norm[:,3], norm[:,4], norm[:,5], dims=2)

X_train = X[1:m1,:]
print(m1)
X_valid = X[m1+1:m1+m2+1,:]
Y_valid=price[m1+1:m1+m2+1,:]
X_test = X[m1+m2+2:end,:]
Y_test=price[m1+m2+2:end,:] #test price values

X_train = X[1:m1,:]
Y_train= price[1:m1,:]




rmse_regpara= [0.0001 ;0.001; 0.01; 0.1; 1; ]
rmse_regpara=rmse_regpara.*0

function rmse_r(Ypred,Y)
	m=length(Y)
    rmse=((sum((Ypred-Y).^2))/m)^0.5
    print("RMSE is= ",rmse)
    print("\n")
    Ymean=mean(Y) # Y is original price
    r=1-(sum((Ypred-Y).^2))/sum((Y.-Ymean).^2)
    print("R Square is= ",r)
    print("\n")
end


# Define a function to calculate cost function
function costFunction(X_train, Y_train, B,regpara)
    #m = length(Y)
    cost = sum(((X_train * B) - Y_train).^2)+sum(regpara*(B.^2)/(2*m1))
    #print(cost," ")
    return cost
end





# # Initial coefficients
B = zeros(4, 1)




function gradientDescent(X_train, Y_train, B, learningRate, numIterations, regpara)
    costHistory = zeros(numIterations)
    #m = length(Y_train)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X_train * B) - Y_train
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = ((X_train' * loss)+regpara.*B)/m1
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X_train, Y_train, B, regpara)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B #, costHistory
end





learningRate = 0.01
lamda=[0.1;0.2;0.4;0.6;0.8]

regpara=lamda[1]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_test= X_valid*newB
rmse_regpara[1]=((sum((Ypred_test-Y_valid).^2)))^0.5





learningRate = 0.01
regpara=lamda[2]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_test= X_valid*newB
rmse_regpara[2]=((sum((Ypred_test-Y_valid).^2)))^0.5




learningRate = 0.01
regpara=lamda[3]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_test= X_valid*newB
rmse_regpara[3]=((sum((Ypred_test-Y_valid).^2)))^0.5

    





learningRate = 0.01
regpara=lamda[4]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_test= X_valid*newB
rmse_regpara[4]=((sum((Ypred_test-Y_valid).^2)))^0.5


learningRate = 0.01
regpara=lamda[5]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_test= X_valid*newB
rmse_regpara[5]=((sum((Ypred_test-Y_valid).^2)))^0.5





z=argmin(rmse_regpara)
real_regpara=rmse_regpara[z]
print(z)
learningRate = 0.01
regpara=lamda[z]
newB = gradientDescent(X_train,  Y_train, B, learningRate, 1000,regpara)
Ypred_testreal=X_test*newB
Ypred_train=X_train*newB
Ypred_all=X*newB


print("RMSE & R for Test data\n")
rmse_r(Ypred_testreal,Y_test)
print("RMSE & R for Training data\n")
rmse_r(Ypred_train,Y_train)
#R square value
Ypred_all=X*newB
# Visualize the learning: how the loss decreased.
#plot(costHistory)
df = DataFrame()

CSV.write("data\\2a.csv",  DataFrame(Ypred_all), writeheader=false)










