g=c("dplyr","car","hflights","lubridate","tidyr","xlsx","stringr",
    "esquisse","vcd","ggplot2","nortest","sas7bdat","psych",
    "stringi","tree","cvTools","randomForest","knitr","xtable",
    "gbm","forecast","caret","ranger","data.table")
lapply(g, library, character.only = TRUE)

#####
st_train=read.csv("C:/Users/DELL/Desktop/Aniket/edvancer/R/Projects/Retail/store_train.csv",stringsAsFactors = F)
st_test=read.csv("C:/Users/DELL/Desktop/Aniket/edvancer/R/Projects/Retail/store_test.csv",stringsAsFactors = F)
setdiff(colnames(st_train),colnames(st_test))
#store
table(st_train$store)

st_test$store=NA

st_train$data="train"
st_test$data="test"

st=rbind(st_train,st_test)

glimpse(st)

st=st %>%
  select(-Id)

z=sapply(st,function(x) is.character(x))
z=z[z==T]
z

table(st$storecode)
is.data.table(st)
setDT(st)
st=st[, SC_M := as.integer(grepl("^METRO", storecode))]
st=st[, SC_N := as.integer(grepl("^NCNTY", storecode))]

glimpse(st)

st$Areaname=strsplit(st$Areaname,split = ", ")
st$Areaname=sapply( st$Areaname, "[", 2 )
table(st$Areaname)

setDT(st)
st=st[, AN_AF := as.integer(grepl("^[A-F]", Areaname))]
st=st[, AN_GL := as.integer(grepl("^[G-L]", Areaname))]
st=st[, AN_MR := as.integer(grepl("^[M-R]", Areaname))]
st=st[, AN_SZ := as.integer(grepl("^[S-Z]", Areaname))]

st=st %>%
  select(-countyname,-storecode,-Areaname,-countytownname)

glimpse(st)

round(prop.table(table(st$store_Type,st$store),1),2)
st=st %>%
  mutate(ST_G=as.numeric(store_Type=="Grocery Store"),
         ST_S3=as.numeric(store_Type=="Supermarket Type3"),
         ST_S2=as.numeric(store_Type=="Supermarket Type2"),
         ST_S1=as.numeric(store_Type=="Supermarket Type1")) %>%
  select(-state_alpha,-store_Type)

#REMOVING NA VALUES.
sapply(st, function(x) sum(is.na(x)))

for(col in names(st)){
  
  if(sum(is.na(st[,col]))>0 & !(col %in% c("data","store"))){
    
    st[is.na(st[,col]),col]=mean(st[,col],na.rm=T)
  }
  
}

glimpse(st)

##FILTERING DATA INTO TRAIN AND TESTS FOR PREDICTION.

st_train=st %>%
  filter(data=="train") %>%
  select(-data)

st_test=st %>%
  filter(data=="test") %>%
  select(-data,-store)

#####

set.seed(66)
s=sample(1:nrow(st_train),0.8*nrow(st_train))
st_train1=st_train[s,]
st_train2=st_train[-s,]

fit=lm(store~.,data=st_train1)
vif(fit)

vars=attributes(alias(fit)$Complete)$dimnames[[1]]
vars

fit=lm(store~.-SC_N-ST_S1,data=st_train1)
vif(fit)
sort(vif(fit),decreasing = T)

fit=lm(store~.-SC_N-ST_S1-AN_MR,data=st_train1)
vif(fit)
sort(vif(fit),decreasing = T)

fit=lm(store~.-SC_N-ST_S1-AN_MR-sales0,data=st_train1)
vif(fit)
sort(vif(fit),decreasing = T)

fit=lm(store~.-SC_N-ST_S1-AN_MR-sales0-sales2,data=st_train1)
vif(fit)
sort(vif(fit),decreasing = T)

fit=lm(store~.-SC_N-ST_S1-AN_MR-sales0-sales2-sales3,data=st_train1)
vif(fit)
sort(vif(fit),decreasing = T)

#####

log_fit=glm(store~.-SC_N-ST_S1-AN_MR-sales0-sales2-sales3,
            data=st_train1,family="binomial")

log_fit=step(log_fit)

formula(log_fit)

# log_fit=glm(store ~ SC_M + AN_GL + CTN_GL + CTN_MR,data=st_train1,family="binomial")
log_fit=glm(store ~ country + CouSub + SC_M + AN_GL,data=st_train1,family="binomial")
summary(log_fit)

library(pROC)

val.score=predict(log_fit,newdata = st_train2,type='response')

auc(roc(st_train2$store,val.score))
#Area under the curve: 0.8373

#####PREDICTION ON ENTIRE TRAINING DATA
#####

log_fit_final=glm(store ~ country + CouSub + SC_M + AN_GL,data=st_train,family="binomial")

log_fit_final=step(log_fit_final)

formula(log_fit_final)

log_fit_final=glm(store ~ country + CouSub + SC_M + AN_GL,
                  data=st_train,family="binomial")

variable=predict(log_fit_final,st_train,type="response")
library(ROCR)
RP=prediction(variable,st_train$store)
RPE=performance(RP,"tpr","fpr")
plot(RPE,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

summary(log_fit_final)

test.prob.score= predict(log_fit_final,newdata = st_test,type='response')
write.table(test.prob.score,"Aniket_Rele_P2_part2.csv",row.names = F,col.names = "store")

variable=predict(log_fit_final,st_train,type="response")
library(ROCR)
RP=prediction(variable,st_train$store)
RPE=performance(RP,"tpr","fpr")
plot(RPE,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

st_train$score=predict(fit,st_train,type="response")
ggplot(st_train,aes(x=score,y=store,color=factor(store)))+geom_point()+geom_jitter()



#CONFUSION MATRIX TO CALCULATE ACCURACY.
table(ActualValue=st_train2$store,PredictedValue=val.score>0.5) #VALUES FROM 0.3-0.8 ARE SAME
Accuracy=(518+346)/(518+50+88+346) #0.8622754
Sn=518/(518+50)                    #0.9119718
Sp=346/(88+346)                    #0.797235

Ks=(518/(518+50))-(88/(88+346))    #0.7092069
Precision=518/(518+88)             #0.8547855
Recall=518/(518+50)                #0.9119718

# TO OBTAIN CUTOFF PROPERLY
variable=predict(log_fit,st_train1,type="response")
library(ROCR)
RPD=prediction(variable,st_train1$store)
RPE=performance(RPD,"tpr","fpr")
plot(RPE,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))
#HERE THE CUTOFFS ARE FROM 0.3-0.8 AS SEEN IN THE GRAPH
