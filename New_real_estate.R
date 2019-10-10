#####Installing libraries
g=c("dplyr","car","hflights","lubridate","tidyr","xlsx","stringr",
    "esquisse","vcd","ggplot2","nortest","sas7bdat","psych",
    "stringi","tree","cvTools","randomForest","knitr","xtable",
    "gbm","forecast","caret","ranger","data.table")
lapply(g, library, character.only = TRUE)

#####
setwd("C:\\Users\\DELL\\Desktop\\Aniket\\edvancer\\R\\Projects\\Real Estate")

housing_train=read.csv("housing_train.csv",sep = ",",stringsAsFactors = F)

housing_test= read.csv("housing_test.csv",sep = ",",stringsAsFactors = F)

setdiff(colnames(housing_train),colnames(housing_test))

housing_test$Price=NA

housing_train$data='train'
housing_test$data='test'

housing_all=rbind(housing_train,housing_test)

library(dplyr)

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

glimpse(housing_all)

#REMOVING NA's
sapply(housing_all,function(x) sum(is.na(x)))

for(col in names(housing_all)){
  
  if(sum(is.na(housing_all[,col]))>0 & !(col %in% c("data","Price"))){
    
    housing_all[is.na(housing_all[,col]),col]=mean(housing_all[,col],na.rm=T)
  }
  
}

glimpse(housing_all)

housing_all=housing_all %>%
  select(-Address)

table(housing_all$Suburb)
table(housing_all$CouncilArea)
table(housing_all$SellerG)

housing_all=housing_all %>%
  mutate(CouncilArea=ifelse(CouncilArea=="","None",CouncilArea))

housing_all=CreateDummies(housing_all,"Suburb",80)
housing_all=CreateDummies(housing_all,"CouncilArea",300)
housing_all=CreateDummies(housing_all,"SellerG",50)

glimpse(housing_all)

table(housing_all$YearBuilt)
housing_all=housing_all %>%
  mutate(yb_18c=as.numeric(YearBuilt %in% c(1800:1899)),
         yb_19c=as.numeric(YearBuilt %in% c(1900:1999)),
         yb_20c=as.numeric(YearBuilt %in% c(2000:2999))) %>%
  select(-YearBuilt)

table(housing_all$Type)
housing_all= housing_all %>%
  mutate(t_h=as.numeric(Type=="h"),
         t_t=as.numeric(Type=="t"),
         t_u=as.numeric(Type=="u")) %>%
  select(-Type)

table(housing_all$Method)
housing_all=housing_all %>%
  mutate(m1=as.numeric(Method=="PI"),
         m2=as.numeric(Method=="S"),
         m3=as.numeric(Method=="SP"),
         m4=as.numeric(Method=="VB")) %>%
  select(-Method)

glimpse(housing_all)

###FILTERING
housing_train=housing_all %>%
  filter(data=="train") %>%
  select(-data)

housing_test=housing_all %>%
  filter(data=="test") %>%
  select(-data,-Price)

#####Splitting the data
set.seed(787)
#787
s=sample(1:nrow(housing_train),0.8*nrow(housing_train))

housing_train1=housing_train[s,]
housing_train2=housing_train[-s,]

fit=lm(Price~.,data=housing_train1)
vif(fit)

##remove aliased variables
vars=attributes(alias(fit)$Complete)$dimnames[[1]]
vars

fit=lm(Price~.-t_u-m2,data=housing_train1)
vif(fit)
sort(vif(fit),decreasing = T )

fit=step(fit)

formula(fit)

fit=lm(Price ~ Rooms + Distance + Postcode + Bedroom2 + Bathroom + Car + 
         Landsize + BuildingArea + Suburb_KeilorEast + Suburb_Prahran + 
         Suburb_SurreyHills + Suburb_Kensington + Suburb_Toorak + 
         Suburb_Doncaster + Suburb_AscotVale + Suburb_Thornbury + 
         Suburb_Hampton + Suburb_Balwyn + Suburb_MalvernEast + Suburb_Camberwell + 
         Suburb_PortMelbourne + Suburb_BrightonEast + Suburb_Hawthorn + 
         Suburb_BalwynNorth + Suburb_Kew + Suburb_Brighton + 
         Suburb_Essendon + Suburb_SouthYarra + Suburb_StKilda + 
         Suburb_Preston + Suburb_Richmond + Suburb_Reservoir + CouncilArea_Banyule + 
         CouncilArea_PortPhillip + CouncilArea_Yarra + CouncilArea_Maribyrnong + 
         CouncilArea_Stonnington + CouncilArea_Moreland + 
         CouncilArea_Boroondara + SellerG_Douglas + 
         SellerG_Williams + SellerG_Kay + SellerG_Miles + 
         SellerG_Greg + SellerG_RT + 
         SellerG_Marshall + SellerG_hockingstuart + SellerG_Jellis + 
         yb_18c + t_h + t_t + m1 + m3 + m4,data=housing_train1)


summary(fit)

val.pred=predict(fit,newdata=housing_train2)

errors=housing_train2$Price-val.pred

rmse=errors**2 %>% mean() %>% sqrt()
rmse

Score =round(212467/rmse,2)
Score
# 0.5321782

##PLOT TO SEE PREDICTION IN GRAPHICAL MANNER.
plot(housing_train2$Price,type="l",lty=1.8,col="green",xlim=c(0,300),ylim=c(0e+00,3e+06),
     xaxs="i",yaxs="i")
lines(val.pred,type="l",col="blue")

### model for predcition on the entire data
fit.final=lm(Price~.-t_u-m2,data=housing_train)
vif(fit.final)
sort(vif(fit.final),decreasing = T)

fit.final=step(fit.final)

formula(fit.final)

fit.final=lm(Price ~ Rooms + Distance + Postcode + Bedroom2 + Bathroom + Car + 
               Landsize + BuildingArea + Suburb_KeilorEast + Suburb_Prahran + 
               Suburb_SurreyHills + Suburb_Kensington + 
               Suburb_Toorak + Suburb_Doncaster + 
               Suburb_Thornbury + Suburb_Hampton + Suburb_Balwyn + 
               Suburb_MalvernEast + Suburb_Camberwell + Suburb_PortMelbourne + 
               Suburb_BrightonEast + Suburb_Hawthorn + Suburb_BalwynNorth + 
               Suburb_Kew + Suburb_Brighton + Suburb_Essendon + Suburb_SouthYarra + 
               Suburb_StKilda + Suburb_Preston + Suburb_Richmond + Suburb_Reservoir + 
               CouncilArea_Banyule + CouncilArea_PortPhillip + CouncilArea_Yarra + 
               CouncilArea_Maribyrnong + CouncilArea_Stonnington + 
               CouncilArea_Moreland + CouncilArea_Boroondara + CouncilArea_None + 
               SellerG_Douglas + SellerG_Williams + SellerG_Kay + 
               SellerG_Miles + SellerG_Greg + SellerG_RT + SellerG_Marshall +
               SellerG_Jellis + yb_18c + t_h + t_t + m1 + m3 + m4,data=housing_train)

summary(fit.final)

test.pred=predict(fit.final,newdata=housing_test)

write.table(test.pred,"Aniket_Rele_P1_part2.csv",row.names=F,col.names = "Price")
