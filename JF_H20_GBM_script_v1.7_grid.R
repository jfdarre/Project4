# H2O GBM script version 1
library(caret)
library(plyr)
library(dplyr)
library(data.table)  
library(h2o)

cat("reading the train and test data (with data.table) \n")
train0 <- fread("../data/train3.csv",stringsAsFactors = T)
test   <- fread("../data/test3.csv",stringsAsFactors = T)
store  <- fread("./input/store.csv",stringsAsFactors = T)
train0 <- train0[Sales > 0,]  ## We are not judged on 0 sales records in test set


train0 <- merge(train0,store,by="Store")
test   <- merge(test,store,by="Store")


## more care should be taken to ensure the dates of test can be projected from train
## decision trees do not project well, so you will want to have some strategy here, if using the dates
train0[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# competition feature
train0$Competition <- (sqrt(max(train0$CompetitionDistance, na.rm = TRUE)-train0$CompetitionDistance))*
  (((train0$year - train0$CompetitionOpenSinceYear) * 12) - (train0$CompetitionOpenSinceMonth-train0$month))

test$Competition <- (sqrt(max(test$CompetitionDistance, na.rm = TRUE)-test$CompetitionDistance))*
  (((test$year - test$CompetitionOpenSinceYear) * 12) - (test$CompetitionOpenSinceMonth-test$month))

## log transformation to not be as sensitive to high sales
## decent rule of thumb: 
##     if the data spans an order of magnitude, consider a log transform
train0[,logSales:=log1p(Sales)]
valid = train0[year == 2015 & month >= 6,]
train = train0[year <  2015 | month <  6,]
dim(valid)
dim(train)


# Set appropriate variables to factors
for (j in c("Store", "DayOfWeek", "Promo", 
            "year", "month", "day", "PromoFirstDate",
#            "day_of_year", "week_of_year", "DayBeforeClosed", "DayAfterClosed",
            "State", "PromoSecondDate", 
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear")) {
  train[[j]] <- as.factor(train[[j]])
  valid[[j]] <- as.factor(valid[[j]])
  test[[j]]  <- as.factor(test[[j]])
}


## Useful functions:
rmse = function(predictions, targets) {
  return(((predictions - targets)/targets) ** 2)
}

sumup = function(model, trainHex, train) {
  train_pred = as.data.frame(h2o.predict(model,trainHex))
  train_pred <- expm1(train_pred[,1])
  train$pred = train_pred
  train$rmse = rmse(train_pred, train$Sales)
  train2 = filter(train, month %in% c(8,9))
  total_rmse = sqrt(sum(train$rmse)/nrow(train))
  print("Total RMSE:")
  print(total_rmse)
  partial_rmse = sqrt(sum(train2$rmse)/nrow(train2))
  print("RMSE on Aug/Sep:")
  print(partial_rmse)
  temp = as.data.frame(rbind(summary(train_pred), summary(train$Sales), summary(train2$pred), summary(train2$Sales)))
  temp$sd = c(round(sd(train_pred)), round(sd(train$Sales)), round(sd(train2$pred)), round(sd(train2$Sales)))
  print("Stats of predictions vs. actual:")
  print(temp)
}


## Use H2O's random forest
## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='5G', assertion = FALSE)

## create validation and training set
trainHex <- as.h2o(train)
validHex <- as.h2o(valid)
testHex  <- as.h2o(test)

## Load data into cluster from R
features   = read.csv('./H2O_submits/h2o_GBM_20_03_800_top100_varimp.csv')
features   = features$variable
features   = as.character(features)

####################################################################################
results = data.frame()
index   = data.frame()
for (sample_rate in c(0.2,0.5,0.8)) {
  for (col_sample_rate in c(0.2,0.5,0.8)) {
    for (max_depth in c(2, 6, 15)) {
      for (learn_rate in c(0.2, 0.5, 0.8)) {
        model_id        = paste0('GBM_sr',sample_rate*100,'_csr',col_sample_rate*100,'_md',max_depth,'_lr',learn_rate*100)
        model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1/",model_id,"_model")
        varimp_path     = paste0("../Grid_v1/",model_id,"_varimp")
        valid_pred_path = paste0("../Grid_v1/",model_id,"_valid_pred")
        train_pred_path = paste0("../Grid_v1/",model_id,"_train_pred")
        test_pred_path  = paste0("../Grid_v1/",model_id,"_test_pred")
        
        gbmHex <- h2o.gbm(  x                = features,
                            y                = "logSales",
                            training_frame   = trainHex,
                            model_id         = model_id,
                            nbins_cats       = 1115,
                            sample_rate      = sample_rate,
                            col_sample_rate  = col_sample_rate,
                            max_depth        = max_depth,
                            learn_rate       = learn_rate,
                            seed             = 12345678,
                            ntrees           = 50,
                            validation_frame = validHex)
        
        h2o.saveModel(gbmHex, path = model_path, force = FALSE)
        
        varimp     = data.frame(h2o.varimp(gbmHex))
        test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
        train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
        valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
        train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
        valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
        index      = rbind(index, data.frame(sample = sample_rate,
                                             col_sp = col_sample_rate,
                                             depth  = max_depth,
                                             learn  = learn_rate))
        results    = rbind(results, data.frame(valid = valid_rmse, train = train_rmse))

        write.csv(varimp, varimp_path,row.names=F)
        write.csv(test_pred, test_pred_path,row.names=F)
        write.csv(train_pred, train_pred_path,row.names=F)
        write.csv(valid_pred, valid_pred_path,row.names=F)
      }
    }
  }
}
results     = cbind(results, index)
write.csv(results, "../Grid_v1/results",row.names=F)

####################################################################################
filter(results[order(results$valid),], depth == 6)

####################################################################################
results2 = data.frame()
index   = data.frame()
for (sample_rate in c(0.2,0.8)) {
  for (col_sample_rate in c(0.2,0.8)) {
    for (max_depth in c(2, 6)) {
      for (learn_rate in c(0.2, 0.5, 0.8)) {
        model_id        = paste0('GBM_sr',sample_rate*100,'_csr',col_sample_rate*100,'_md',max_depth,'_lr',learn_rate*100)
        model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1_1/",model_id,"_model")
        varimp_path     = paste0("../Grid_v1_1/",model_id,"_varimp")
        valid_pred_path = paste0("../Grid_v1_1/",model_id,"_valid_pred")
        train_pred_path = paste0("../Grid_v1_1/",model_id,"_train_pred")
        test_pred_path  = paste0("../Grid_v1_1/",model_id,"_test_pred")
        
        gbmHex <- h2o.gbm(  x                = features,
                            y                = "logSales",
                            training_frame   = trainHex,
                            model_id         = model_id,
                            nbins_cats       = 1115,
                            sample_rate      = sample_rate,
                            col_sample_rate  = col_sample_rate,
                            max_depth        = max_depth,
                            learn_rate       = learn_rate,
                            #seed             = 12345678,
                            ntrees           = 100,
                            validation_frame = validHex)
        
        h2o.saveModel(gbmHex, path = model_path, force = FALSE)
        
        varimp     = data.frame(h2o.varimp(gbmHex))
        test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
        train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
        valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
        train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
        valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
        index      = rbind(index, data.frame(sample = sample_rate,
                                             col_sp = col_sample_rate,
                                             depth  = max_depth,
                                             learn  = learn_rate))
        results2    = rbind(results2, data.frame(valid = valid_rmse, train = train_rmse))

        write.csv(varimp, varimp_path,row.names=F)
        write.csv(test_pred, test_pred_path,row.names=F)
        write.csv(train_pred, train_pred_path,row.names=F)
        write.csv(valid_pred, valid_pred_path,row.names=F)
      }
    }
  }
}
results2     = cbind(results2, index)
write.csv(results2, "../Grid_v1_1/results",row.names=F)

####################################################################################
filter(results2[order(results2$valid),], depth == 2)

####################################################################################
results3 = data.frame()
index   = data.frame()
for (sample_rate in c(0.2,0.5)) {
  for (col_sample_rate in c(0.2,0.5)) {
    for (max_depth in c(2, 6)) {
      for (learn_rate in c(0.1, 0.01)) {
        model_id        = paste0('GBM_sr',sample_rate*100,'_csr',col_sample_rate*100,'_md',max_depth,'_lr',learn_rate*100)
        model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1_1/",model_id,"_model")
        varimp_path     = paste0("../Grid_v1_1/",model_id,"_varimp")
        valid_pred_path = paste0("../Grid_v1_1/",model_id,"_valid_pred")
        train_pred_path = paste0("../Grid_v1_1/",model_id,"_train_pred")
        test_pred_path  = paste0("../Grid_v1_1/",model_id,"_test_pred")
        
        gbmHex <- h2o.gbm(  x                = features,
                            y                = "logSales",
                            training_frame   = trainHex,
                            model_id         = model_id,
                            nbins_cats       = 1115,
                            sample_rate      = sample_rate,
                            col_sample_rate  = col_sample_rate,
                            max_depth        = max_depth,
                            learn_rate       = learn_rate,
                            #seed             = 12345678,
                            ntrees           = 200,
                            validation_frame = validHex)
        
        h2o.saveModel(gbmHex, path = model_path, force = FALSE)
        
        varimp     = data.frame(h2o.varimp(gbmHex))
        test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
        train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
        valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
        train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
        valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
        index      = rbind(index, data.frame(sample = sample_rate,
                                             col_sp = col_sample_rate,
                                             depth  = max_depth,
                                             learn  = learn_rate))
        results3    = rbind(results3, data.frame(valid = valid_rmse, train = train_rmse))

        write.csv(varimp, varimp_path,row.names=F)
        write.csv(test_pred, test_pred_path,row.names=F)
        write.csv(train_pred, train_pred_path,row.names=F)
        write.csv(valid_pred, valid_pred_path,row.names=F)
      }
    }
  }
}
results3     = cbind(results3, index)
write.csv(results3, "../Grid_v1_1/results",row.names=F)

####################################################################################
filter(results3[order(results3$valid),], depth == 6)

####################################################################################
results4 = data.frame()
index   = data.frame()
for (max_depth in c(2, 6)) {
  for (learn_rate in c(0.1, 0.01)) {
    model_id        = paste0('GBM_md',max_depth,'_lr',learn_rate*100)
    model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1_1/",model_id,"_model")
    varimp_path     = paste0("../Grid_v1_1/",model_id,"_varimp")
    valid_pred_path = paste0("../Grid_v1_1/",model_id,"_valid_pred")
    train_pred_path = paste0("../Grid_v1_1/",model_id,"_train_pred")
    test_pred_path  = paste0("../Grid_v1_1/",model_id,"_test_pred")
    
    gbmHex <- h2o.gbm(  x                = features,
                        y                = "logSales",
                        training_frame   = trainHex,
                        model_id         = model_id,
                        nbins_cats       = 1115,
                        max_depth        = max_depth,
                        learn_rate       = learn_rate,
                        #seed             = 12345678,
                        ntrees           = 1000,
                        validation_frame = validHex)
    
    h2o.saveModel(gbmHex, path = model_path, force = FALSE)
    
    varimp     = data.frame(h2o.varimp(gbmHex))
    test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
    train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
    valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
    train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
    valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
    index      = rbind(index, data.frame(sample = sample_rate,
                                         col_sp = col_sample_rate,
                                         depth  = max_depth,
                                         learn  = learn_rate))
    results4    = rbind(results4, data.frame(valid = valid_rmse, train = train_rmse))

    write.csv(varimp, varimp_path,row.names=F)
    write.csv(test_pred, test_pred_path,row.names=F)
    write.csv(train_pred, train_pred_path,row.names=F)
    write.csv(valid_pred, valid_pred_path,row.names=F)
  }
}
results4     = cbind(results4, index)
write.csv(results4, "../Grid_v1_1/results",row.names=F)

####################################################################################
filter(results4[order(results4$valid),], depth == 6)

####################################################################################
temp            = read.csv('../Grid_v1_1/GBM_md2_lr1_test_pred')
temp            = data.frame(Id = test$Id, Sales = temp$x)
write.csv(temp, "../Grid_v1_2/GBM_dp2_lr1_nt5000_v1_1",row.names=F)
head(temp)
####################################################################################
results4 = data.frame()
index   = data.frame()
for (max_depth in c(4)) {
  for (learn_rate in c(0.01)) {
    model_id        = paste0('GBM_md',max_depth,'_lr',learn_rate*100)
    model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1_3/",model_id,"_model")
    varimp_path     = paste0("../Grid_v1_3/",model_id,"_varimp")
    valid_pred_path = paste0("../Grid_v1_3/",model_id,"_valid_pred")
    train_pred_path = paste0("../Grid_v1_3/",model_id,"_train_pred")
    test_pred_path  = paste0("../Grid_v1_3/",model_id,"_test_pred")
    
    gbmHex <- h2o.gbm(  x                = features,
                        y                = "logSales",
                        training_frame   = trainHex,
                        model_id         = model_id,
                        nbins_cats       = 1115,
                        max_depth        = max_depth,
                        learn_rate       = learn_rate,
                        #seed             = 12345678,
                        ntrees           = 5000,
                        validation_frame = validHex)
    
    h2o.saveModel(gbmHex, path = model_path, force = FALSE)
    
    varimp     = data.frame(h2o.varimp(gbmHex))
    test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
    train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
    valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
    train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
    valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
    index      = rbind(index, data.frame(depth  = max_depth,
                                         learn  = learn_rate))
    results4    = rbind(results4, data.frame(valid = valid_rmse, train = train_rmse))

    write.csv(varimp, varimp_path,row.names=F)
    write.csv(test_pred, test_pred_path,row.names=F)
    write.csv(train_pred, train_pred_path,row.names=F)
    write.csv(valid_pred, valid_pred_path,row.names=F)
  }
}
results4     = cbind(results4, index)
write.csv(results4, "../Grid_v1_3/results",row.names=F)
results4
summary(gbmHex)
####################################################################################


####################################################################################

temp_test     = read.csv('../Grid_v1_1/GBM_md2_lr1_test_pred')
temp_train    = read.csv('../Grid_v1_1/GBM_md2_lr1_train_pred')
temp_valid    = read.csv('../Grid_v1_1/GBM_md2_lr1_valid_pred')
train_rmse = sqrt(sum(rmse(temp_train, train$Sales))/nrow(train))
valid_rmse = sqrt(sum(rmse(temp_valid, valid$Sales))/nrow(valid))

temp_test     = read.csv('../Grid_v1_1/GBM_md6_lr1_test_pred')
temp_train    = read.csv('../Grid_v1_1/GBM_md6_lr1_train_pred')
temp_valid    = read.csv('../Grid_v1_1/GBM_md6_lr1_valid_pred')
train_rmse = sqrt(sum(rmse(temp_train, train$Sales))/nrow(train))
valid_rmse = sqrt(sum(rmse(temp_valid, valid$Sales))/nrow(valid))

####################################################################################
results5 = data.frame()
index   = data.frame()
for (max_depth in c(4)) {
  for (learn_rate in c(0.01)) {
    model_id        = paste0('GBM_md',max_depth,'_lr',learn_rate*100)
    model_path      = paste0("/Users/jfdarre/Documents/NYCDS/Project4/Grid_v1_4/",model_id,"_model")
    varimp_path     = paste0("../Grid_v1_4/",model_id,"_varimp")
    valid_pred_path = paste0("../Grid_v1_4/",model_id,"_valid_pred")
    train_pred_path = paste0("../Grid_v1_4/",model_id,"_train_pred")
    test_pred_path  = paste0("../Grid_v1_4/",model_id,"_test_pred")
    
    gbmHex <- h2o.gbm(  x                = features,
                        y                = "logSales",
                        training_frame   = trainHex,
                        model_id         = model_id,
                        nbins_cats       = 1115,
                        max_depth        = max_depth,
                        learn_rate       = learn_rate,
                        #seed             = 12345678,
                        ntrees           = 8000,
                        validation_frame = validHex)
    
    h2o.saveModel(gbmHex, path = model_path, force = FALSE)
    
    varimp     = data.frame(h2o.varimp(gbmHex))
    test_pred  = expm1(as.data.frame(h2o.predict(gbmHex,testHex))[,1])
    train_pred = expm1(as.data.frame(h2o.predict(gbmHex,trainHex))[,1])
    valid_pred = expm1(as.data.frame(h2o.predict(gbmHex,validHex))[,1])
    train_rmse = sqrt(sum(rmse(train_pred, train$Sales))/nrow(train))
    valid_rmse = sqrt(sum(rmse(valid_pred, valid$Sales))/nrow(valid))
    index      = rbind(index, data.frame(depth  = max_depth,
                                         learn  = learn_rate))
    results5    = rbind(results5, data.frame(valid = valid_rmse, train = train_rmse))

    write.csv(varimp, varimp_path,row.names=F)
    write.csv(test_pred, test_pred_path,row.names=F)
    write.csv(train_pred, train_pred_path,row.names=F)
    write.csv(valid_pred, valid_pred_path,row.names=F)
  }
}
results5     = cbind(results5, index)
write.csv(results5, "../Grid_v1_4/results",row.names=F)
results5
####################################################################################
temp            = read.csv('../Grid_v1_4/GBM_md4_lr1_test_pred')
temp            = data.frame(Id = test$Id, Sales = temp$x)
write.csv(temp, "../Grid_v1_4/GBM_dp4_lr1_nt8000_v1_1",row.names=F)
head(temp)
summary(temp)
summary(gbmHex)
