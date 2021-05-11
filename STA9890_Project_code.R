rm(list=ls())

# Install / load required libraries
library(tidyverse)
library(glmnet)
library(MASS)
library(ggplot2)
library(gridExtra)
library(randomForest)

########
## 1. ##
########
# Bring in dataset
data           <- read.csv('/Users/Downloads/STA9890 project data.csv', header = TRUE)
data$X         <- NULL

# Removing impertinent variable
#data             <- subset(data, select = -c(timeunit,MTLp2A,MTLp3A,MTLp4A,MTLp5A,MTLp6A,MTLp7A,MTLp8A,MTLp9A,MTLp10A,MTLp11A,MTLp12A,MTLp13A,MTLp14A,MTLp15A,MTLp16A))
#data             <- data %>% drop_na()
#write.csv(data, 'STA9890 project data.csv')

#########
## 2.a ##
#########
# Isolating response variable & removing from main data
y                = data$LBL
data             = subset(data, select = -c(LBL))

#############
## 2.b & c ##
#############
# Dimensions of the data
n                = dim(data)[1]   # n = 8966
p                = dim(data)[2]   # p = 60
# summary(data)
# str(data)

########
## 3. ##
########
# Convert regressor to matrix
X                = data.matrix(data) 
y                = data.matrix(y)

# Normalized
mu = as.vector(apply(X, 2, 'mean'))
mu.y = as.vector(apply(y, 2, 'mean'))
for (i in c(1:n)){
  X[i,]  =   (X[i,] - mu)/mu
  y[i,]  =   (y[i,] - mu.y)/mu.y
}

#################
## 3.a, b, c,d ##
#################
# Setting seed, split data, repetition count
set.seed(2)
n.train          = floor(0.8 * n)
n.test           = n - n.train
M                = 100

# R-Squared vectors
# rid = ridge
Rsq.test.rid     = rep(0,M)  
Rsq.train.rid    = rep(0,M)

# las = lasso
Rsq.test.las     = rep(0,M)  
Rsq.train.las    = rep(0,M)

# en = elastic net
Rsq.test.en      = rep(0,M)  
Rsq.train.en     = rep(0,M)

# rf = random forest 
Rsq.test.rf      = rep(0,M) 
Rsq.train.rf     = rep(0,M)

# rid
rid.time = system.time(for (m in c(1:M)) {
  
  # Randomly split the data into train and test
  shuffled_indexes = sample(n)
  train            = shuffled_indexes[1:n.train]
  test             = shuffled_indexes[(1+n.train):n]
  X.train          = X[train, ]
  y.train          = y[train]
  X.test           = X[test, ]
  y.test           = y[test]
  
  
  # fit ridge and calculate and record the train and test R squares 
  a=0 # ridge
  rid.cv           = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  rid.fit          = glmnet(X.train, y.train, alpha = a, lambda = rid.cv$lambda.min)
  y.train.hat      = predict(rid.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       = predict(rid.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]  = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rid[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.rid=%.2f,  Rsq.test.rid=%.2f| Rsq.train.rid=%.2f,  Rsq.train.rid=%.2f| \n", m,  Rsq.test.rid[m], Rsq.test.rid[m],  Rsq.train.rid[m], Rsq.train.rid[m]))
})
rid.time = rid.time[3]/10
plot(rid.cv, main = 'Ridge') # 10-fold CV curves

# en
en.time  = system.time(for (m in c(1:M)) {
  shuffled_indexes = sample(n)
  train            = shuffled_indexes[1:n.train]
  test             = shuffled_indexes[(1+n.train):n]
  X.train          = X[train, ]
  y.train          = y[train]
  X.test           = X[test, ]
  y.test           = y[test]
  
  
  # fit en and calculate and record the train and test R squares 
  a=0.5 # elastic-net 0<a<1
  en.cv            = cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  en.fit           = glmnet(X.train, y.train, intercept = FALSE, alpha = a, lambda = en.cv$lambda.min)
  y.train.hat      = predict(en.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       = predict(en.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.en[m]  = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.en=%.2f,  Rsq.test.en=%.2f| Rsq.train.en=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.en[m], Rsq.test.en[m],  Rsq.train.en[m], Rsq.train.en[m]))
})
en.time  = en.time[3]/10
plot(en.cv, main = 'Elastic-net') # 10-fold CV curves

# las
las.time = system.time(for (m in c(1:M)) {
  shuffled_indexes = sample(n)
  train            = shuffled_indexes[1:n.train]
  test             = shuffled_indexes[(1+n.train):n]
  X.train          = X[train, ]
  y.train          = y[train]
  X.test           = X[test, ]
  y.test           = y[test]
  
  
  # fit ridge and calculate and record the train and test R squares 
  a=1 # lasso
  las.cv           = cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  las.fit          = glmnet(X.train, y.train, intercept = FALSE, alpha = a, lambda = las.cv$lambda.min)
  y.train.hat      = predict(las.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       = predict(las.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.las[m]  = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.las[m] = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.las=%.2f,  Rsq.test.las=%.2f| Rsq.train.las=%.2f,  Rsq.train.las=%.2f| \n", m,  Rsq.test.las[m], Rsq.test.las[m],  Rsq.train.las[m], Rsq.train.las[m]))
})
las.time = las.time[3]/10
plot(las.cv, main = 'Lasso') # 10-fold CV curves

# rf
rf.time  = system.time(for (m in c(1:M)) {
  shuffled_indexes = sample(n)
  train            = shuffled_indexes[1:n.train]
  test             = shuffled_indexes[(1+n.train):n]
  X.train          = X[train, ]
  y.train          = y[train]
  X.test           = X[test, ]
  y.test           = y[test]
  
  
  # fit rf and calculate and record the train and test R squares 
  rf.fit           = randomForest(x = X.train, y = y.train, mtry = floor(sqrt(p)), ntree = 500, importance = TRUE)
  y.train.hat      = predict(rf.fit, newdata = X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       = predict(rf.fit, newdata = X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rf[m]   = 1-mean((y.test - y.test.hat)^2)/mean((y.test - mean(y.test))^2)
  Rsq.train.rf[m]  = 1-mean((y.train - y.train.hat)^2)/mean((y.train - mean(y.train))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.rf=%.2f| Rsq.train.rf=%.2f,  Rsq.train.rf=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.rf[m],  Rsq.train.rf[m], Rsq.train.rf[m]))
})
rf.time  = rf.time[3]/10

CV.df = data.frame(row.names = c('rg','en','las','rf'),
                   c(rid.time,en.time,las.time,rf.time), 
                   c(quantile(Rsq.test.rid, prob = 0.05),
                     quantile(Rsq.test.en, prob = 0.05),
                     quantile(Rsq.test.las, prob = 0.05),
                     quantile(Rsq.test.rf, prob = 0.05)
                     ),
                   c(quantile(Rsq.test.rid, prob = 0.95),
                     quantile(Rsq.test.en, prob = 0.95),
                     quantile(Rsq.test.las, prob = 0.95),
                     quantile(Rsq.test.rf, prob = 0.95)
                   )
                   )
colnames(CV.df) =  c("time", "lower", "upper")
# save
# write.csv(CV.df, 'CV_results.csv')


#########
## 4.b ##
#########
Rsq.df = data.frame(c(rep("train", 4*M), rep("test", 4*M)), 
                    
                    c(rep("rid",M),rep("en",M), 
                      rep("las",M),rep("rf",M), 
                      rep("rid",M),rep("en",M), 
                      rep("las",M),rep("rf",M)), 
                    
                    c(Rsq.train.rid, Rsq.train.en, Rsq.train.las, Rsq.train.rf, 
                      Rsq.test.rid, Rsq.test.en, Rsq.test.las, Rsq.test.rf))

colnames(Rsq.df) =  c("type", "method", "R_Squared")
Rsq.df

# save
# write.csv(Rsq.df, 'Rsq.csv')

# Changing order of factor levels
Rsq.df$method   = factor(Rsq.df$method, levels = c("rid", "en", "las", "rf"))
Rsq.df$type     = factor(Rsq.df$type, levels = c("train", "test"))

Rsq.df.boxplot  = ggplot(Rsq.df) + aes(x = method, y = R_Squared, fill = method) + 
  geom_boxplot() + facet_wrap(~ type, ncol = 2) + 
  labs(title = expression('Boxplots of R'[train]^{2}*' and R'[test]^{2}*' for Four Methods '),x = "", y = expression('R'^2),fill = "Method") + ylim(0.65, 1)
Rsq.df.boxplot 

Rsq.df.boxplot2 = ggplot(Rsq.df) + aes(x = type, y = R_Squared, fill = type) +
  geom_boxplot() + facet_wrap(~ method, ncol = 4) +
  labs(title = expression('Boxplots of R'[train]^{2}*' and R'[test]^{2}*' for each Methods '),x = "", y = expression('R'^2),fill = "Type") + ylim(0.65, 1)
Rsq.df.boxplot2

#########
## 4.c ##
#########
# 10-fold CV curves for Ridge, EN, Lasso
par(mfrow=c(1,3))
plot(rid.cv)
title("10-fold CV Curve - rid", line = 3)
plot(en.cv)
title("10-fold CV Curve - en", line = 3)
plot(las.cv)
title("10-fold CV Curve - las", line = 3)

#########
## 4.d ##
#########
# train and test residuals
shuffled_indexes = sample(n)
train            = shuffled_indexes[1:n.train]
test             = shuffled_indexes[(1+n.train):n]
X.train          = X[train, ]
y.train          = y[train]
X.test           = X[test, ]
y.test           = y[test]

# fit ridge
a=0 # ridge
rid.cv           = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
rid.fit          = glmnet(X.train, y.train, alpha = a, lambda = rid.cv$lambda.min)
y.train.hat      = predict(rid.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       = predict(rid.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.train.rid    = y.train - y.train.hat
Res.test.rid     = y.test - y.test.hat

res.df.rid       = data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n),
                              c(Res.train.rid, Res.test.rid))
colnames(res.df.rid) =     c("type", "NO.", "residuals")

# fit en
a=0.5 # en
en.cv            = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
en.fit           = glmnet(X.train, y.train, alpha = a, lambda = en.cv$lambda.min)
y.train.hat      = predict(en.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       = predict(en.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.train.en     = y.train - y.train.hat
Res.test.en      = y.test - y.test.hat

res.df.en        = data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n),
                              c(Res.train.en, Res.test.en))
colnames(res.df.en) =     c("type", "NO.", "residuals")

# fit las
a=1 # las
las.cv           = cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
las.fit          = glmnet(X.train, y.train, alpha = a, lambda = las.cv$lambda.min)
y.train.hat      = predict(las.fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       = predict(las.fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.train.las    = y.train - y.train.hat
Res.test.las     = y.test - y.test.hat

res.df.las       = data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n),
                              c(Res.train.las, Res.test.las))
colnames(res.df.las) =     c("type", "NO.", "residuals")

# fit rf 
rf.fit           = randomForest(x = X.train, y = y.train, mtry = floor(sqrt(p)), ntree = 500, importance = TRUE)
y.train.hat      = predict(rf.fit, newdata = X.train) # y.train.hat=X.train %*% fit$beta + fit$a0
y.test.hat       = predict(rf.fit, newdata = X.test) # y.test.hat=X.test %*% fit$beta  + fit$a0
Res.train.rf     = y.train - y.train.hat
Res.test.rf      = y.test - y.test.hat

res.df.rf       = data.frame(c(rep("train", n.train),rep("test", n.test)), c(1:n),
                              c(Res.train.rf, Res.test.rf))
colnames(res.df.rf) =     c("type", "NO.", "residuals")


Res.df = data.frame(
  c(rep("Rid",n), rep("En",n), rep("Las",n), rep("Rf",n)),
  rbind(res.df.rid, res.df.en, res.df.las, res.df.rf)
  )

colnames(Res.df) =  c("method", "type", "NO.", "residual")
Res.df

# save
#write.csv(Res.df, 'Res.csv')

# Changing order of factor levels
Res.df$method  = factor(Res.df$method, levels = c("Rid", "En", "Las", "Rf"))
Res.df$type    = factor(Res.df$type, levels = c("train", "test"))

Res.df.boxplot = ggplot(Res.df) + aes(x = method, y = residual, fill = method) + 
  geom_boxplot() + facet_wrap(~ type, ncol = 2) + 
  labs(title = expression('Boxplots of Train and Test Residuals'),x = "", y = "Residuals",fill = "method")
Res.df.boxplot 

############
## 5.a, b ##
############
# Fit all data

# Ridge
a=0 # ridge
Rid.time         = system.time(
  for (i in 1:1) {
    Rid.cv           = cv.glmnet(X, y, intercept = FALSE, alpha = a, nfolds = 10)
    Rid.fit          = glmnet(X, y, intercept = FALSE, alpha = a, lambda = Rid.cv$lambda.min)
    }
  )

y.test.hat           = predict(Rid.fit, newx = X, type = "response")
Rsq.Rid              = 1-mean((y - y.test.hat)^2)/mean((y - mean(y.test))^2)
beta.hat.Rid         = Rid.fit$beta[ ,Rid.fit$lambda==Rid.cv$lambda.min]

# EN
a=0.5 # elastic-net 0<a<1
En.time         = system.time(
  for (i in 1:1) {
    En.cv            = cv.glmnet(X, y, intercept = FALSE, alpha = a, nfolds = 10)
    En.fit           = glmnet(X, y, intercept = FALSE, alpha = a, lambda = En.cv$lambda.min)
    }
  )
y.test.hat           = predict(En.fit, newx = X, type = "response")
Rsq.En               = 1-mean((y - y.test.hat)^2)/mean((y - mean(y.test))^2)
beta.hat.En          = En.fit$beta[ ,En.fit$lambda==En.cv$lambda.min]

# Lasso
a=1 # lasso
Las.time         = system.time(
  for (i in 1:1) {
    Las.cv           = cv.glmnet(X, y, intercept = FALSE, alpha = a, nfolds = 10)
    Las.fit          = glmnet(X, y, intercept = FALSE, alpha = a, lambda = Las.cv$lambda.min)
    }
  )
y.test.hat           = predict(Las.fit, newx = X, type = "response")
Rsq.Las              = 1-mean((y - y.test.hat)^2)/mean((y - mean(y.test))^2)
beta.hat.Las         = Las.fit$beta[ ,Las.fit$lambda==Las.cv$lambda.min]

# Rf
Rf.time          = system.time(
  for (i in 1:1) {
    Rf.fit           = randomForest(X, y, mtry = floor(sqrt(p)), importance = TRUE)
      }
  )
y.train.hat          = predict(Rf.fit, newdata = X) 
y.test.hat           = predict(Rf.fit, newdata = X)
Rsq.Rf               = 1-mean((y - y.test.hat)^2)/mean((y - mean(y.test))^2)
beta.hat.Rf          = Rf.fit$importance[,1]

R2.All               = c(Rsq.Rid, Rsq.En, Rsq.Las, Rsq.Rf)
cost.time            = c(Rid.time[3], En.time[3], Las.time[3], Rf.time[3])
CV.df                = cbind(CV.df, cost.time, R2.All)
colnames(CV.df) =  c("time", "lower", "upper","R2", "cost.time")

#write.csv(CV.df, 'CV_results.csv')


betaS.Rid            = data.frame(c(1:p), as.vector(beta.hat.Rid))
colnames(betaS.Rid)  = c( "feature", "value")

betaS.En             = data.frame(c(1:p), as.vector(beta.hat.En))
colnames(betaS.En)   = c( "feature", "value")

betaS.Las            = data.frame(c(1:p), as.vector(beta.hat.Las))
colnames(betaS.Las)  = c( "feature", "value")

betaS.Rf             = data.frame(c(1:p), as.vector(beta.hat.Rf))
colnames(betaS.Rf)   = c( "feature", "value")

# Changing order of factor levels
betaS.Rid$feature    =  factor(betaS.Rid$feature, levels = betaS.En$feature[order(betaS.En$value, decreasing = TRUE)])
betaS.En$feature     =  factor(betaS.En$feature,  levels = betaS.En$feature[order(betaS.En$value, decreasing = TRUE)])
betaS.Las$feature    =  factor(betaS.Las$feature, levels = betaS.En$feature[order(betaS.En$value, decreasing = TRUE)])
betaS.Rf$feature     =  factor(betaS.Rf$feature,  levels = betaS.En$feature[order(betaS.En$value, decreasing = TRUE)])

rgPlot =  ggplot(betaS.Rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  labs(title = expression('Estimated coefficients - Ridge'))

enPlot =  ggplot(betaS.En, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  labs(title = expression('Estimated coefficients - EN'))

lsPlot =  ggplot(betaS.Las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  labs(title = expression('Estimated coefficients - Lasso'))

rfPlot =  ggplot(betaS.Rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  labs(title = expression('Importance of Variable - RF'))


grid.arrange(rgPlot, enPlot, lsPlot, rfPlot,nrow = 4) 

###########
### END ###
###########
  




