#####Installing libraries
install.packages(c("dplyr","car","hflights","lubridate","tidyr","xlsx","stringr",
                   "esquisse","vcd","ggplot2","nortest","sas7bdat","psych",
                   "stringi","tree","cvTools","randomForest","knitr","xtable",
                   "gbm","forecast","caret","ranger","data.table"))
g=c("dplyr","car","hflights","lubridate","tidyr","xlsx","stringr",
    "esquisse","vcd","ggplot2","nortest","sas7bdat","psych",
    "stringi","tree","cvTools","randomForest","knitr","xtable",
    "gbm","forecast","caret","ranger","data.table")
lapply(g, library, character.only = TRUE)

#####
setwd("C:\\Users\\DELL\\Desktop\\Aniket\\edvancer\\R\\Projects\\Banking")

bank_train=read.csv("bank-full_train.csv",stringsAsFactors = F)
bank_test=read.csv("bank-full_test.csv",stringsAsFactors = F)

setdiff(colnames(bank_train),colnames(bank_test))
#y

bank_test$y=NA

bank_train$data="train"
bank_test$data="test"

bank=rbind(bank_train,bank_test)

glimpse(bank)

bank=bank %>%
  select(-ID)

table(bank$age)
quantile(bank$age)
bank$age=ifelse(bank$age <= 33, 1, 
                ifelse(bank$age <= 39, 2, 
                       ifelse(bank$age <= 48, 3, 4)))

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

bank=CreateDummies(bank,"job")
bank=CreateDummies(bank,"marital")
bank=CreateDummies(bank,"education")
bank=CreateDummies(bank,"contact")
bank=CreateDummies(bank,"month")
bank=CreateDummies(bank,"poutcome")

table(bank$y)
bank$y=(bank$y=="yes") + 0

table(bank$default)
bank$default=(bank$default=="yes") +0

table(bank$loan)
bank$loan=(bank$loan=="yes") +0

table(bank$housing)
bank$housing=(bank$housing=="yes") +0

glimpse(bank)

z=sapply(bank,function(x) is.character(x))
z=z[z==T]
z

# table(bank$pdays)
# quantile(bank$pdays)
# bank$pdays=ifelse(bank$pdays <= -1, 1, 
#                   ifelse(bank$pdays <= -1, 2, 
#                          ifelse(bank$pdays <= -1, 3, 4)))
# 
# table(bank$campaign)
# quantile(bank$campaign)
# bank$campaign=ifelse(bank$campaign <= 1, 1, 
#                      ifelse(bank$campaign <= 2, 2, 
#                             ifelse(bank$campaign <= 3, 3, 4)))
# 
# table(bank$duration)
# quantile(bank$duration)
# bank$duration=ifelse(bank$duration <= 103, 1, 
#                      ifelse(bank$duration <= 180, 2, 
#                             ifelse(bank$duration <= 319, 3, 4)))
# 
# glimpse(bank)
# 
# table(bank$balance)
# quantile(bank$balance)
# bank$balance=ifelse(bank$balance <= 72, 1, 
#                     ifelse(bank$balance <= 448, 2, 
#                            ifelse(bank$balance <= 1428, 3, 4)))
# 
# table(bank$day)
# quantile(bank$day)
# bank$day=ifelse(bank$day <= 8, 1, 
#                 ifelse(bank$day <= 16, 2, 
#                        ifelse(bank$day <= 21, 3, 4)))

glimpse(bank)
#####CHECKING FOR NA VALUES.
lapply(bank,function(x) sum(is.na(x)))

#####FILTERING DATA
bank_train=bank %>% 
  filter(data=='train') %>% 
  select(-data)

bank_test=bank%>% 
  filter(data=='test') %>% 
  select(-data,-y)

#####
set.seed(2211)
s=sample(1:nrow(bank_train),0.8*nrow(bank_train))
bank_train1=bank_train[s,]
bank_train2=bank_train[-s,]

fit=lm(y~.,data=bank_train1)
vif(fit)
sort(vif(fit),decreasing = T)

fit=lm(y~.-month_may-job_blue_collar,data=bank_train1)
vif(fit)
sort(vif(fit),decreasing = T)

#####
log_fit=glm(y~.-month_may-job_blue_collar,data=bank_train1,family="binomial")
log_fit=step(log_fit)

formula(log_fit)

log_fit=glm(y ~ balance + housing + loan + day + duration + campaign + job_student + 
              job_housemaid + job_retired + job_admin. + job_technician + 
              job_management + marital_married + education_primary + 
              education_tertiary + contact_unknown + month_mar + month_sep + 
              month_oct + month_jan + month_feb + month_apr + month_nov + 
              month_jun + month_aug + month_jul + poutcome_other + poutcome_failure + 
              poutcome_unknown,data=bank_train1,family="binomial")

summary(log_fit)

#####
library(pROC)
val.score=predict(log_fit,newdata = bank_train2,type='response')
auc(roc(bank_train2$y,val.score))
#Area under the curve: 0.9027

#####
log_fit_final=glm(y ~ balance + housing + loan + day + duration + campaign + job_student + 
                    job_housemaid + job_retired + job_admin. + job_technician + 
                    job_management + marital_married + education_primary + 
                    education_tertiary + contact_unknown + month_mar + month_sep + 
                    month_oct + month_jan + month_feb + month_apr + month_nov + 
                    month_jun + month_aug + month_jul + poutcome_other + poutcome_failure + 
                    poutcome_unknown,data=bank_train,family = "binomial")

formula(log_fit_final)
summary(log_fit_final)

variable=predict(log_fit_final,bank_train,type="response")

test.prob.score= as.numeric(predict(log_fit_final,newdata = bank_test,type='response')>0.3)
test.prob.score=ifelse(test.prob.score==1,'Yes','No')
write.table(test.prob.score,"Aniket_Rele_P5_part2.csv",row.names = F,col.names="y")
#####
library(ROCR)
RP=prediction(variable,bank_train$y)
RPE=performance(RP,"tpr","fpr")
plot(RPE,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

# 0.3 THRESHOLD OBTAINED FROM GRAPH
#CONFUSION MATRIX
table(ActualValue=bank_train2$y,PredictedValue=val.score>0.3)

ks=(5286/(5286+299))-(355/(355+390))
round(ks,2)
#0.47

library(ggplot2)
bank_train$score=predict(log_fit_final,bank_train,type="response")
ggplot(bank_train,aes(x=score,y=y,color=factor(y)))+geom_point()+geom_jitter()

k=read.csv("Aniket_Rele_P5_part2.csv")
table(k$y)
