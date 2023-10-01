# Análise de Sobrevivência
# Dataset https://www.kaggle.com/datasets/davinwijaya/employee-turnover?resource=download

"""
stag
Experience (time)

event
Employee turnover

gender
Employee's gender, female(f), or male(m)

age
Employee's age (year)

industry
Employee's Industry

profession
Employee's profession

traffic
From what pipelene employee came to the company. You contacted the company directly (after learning from advertising, knowing the company's brand, etc.) 
- advert You contacted the company directly on the recommendation of your friend 
- NOT an employee of this company-recNErab You contacted the company directly on the recommendation of your friend - an employee of this company 
- referal You have applied for a vacancy on the job site 
- youjs The recruiting agency brought you to the employer 
- KA Invited by the Employer, we knew him before the employment 
- friends The employer contacted you on the recommendation of a person who knows you 
- rabrecNErab The employer reached you through your resume on the job site 
- empjs

coach
Presence of a coach (training) on probation

head_gender
head (supervisor) gender

greywage
The salary does not seem to the tax authorities. Greywage in Russia or Ukraine means that the employer (company) pay just a tiny bit amount of salary above the white-wage (white-wage means minimum wage)

way
Employee's way of transportation

extraversion
Extraversion score

independ
Independend score

selfcontrol
Selfcontrol score

anxiety
Anxiety score

novator
Novator score
"""

# Define o diretório
setwd('C:/Users/R')
getwd()

# Instala os pacotes
install.packages('caret')
install.packages('fastDummies')
install.packages("ROCR")

# Carrega os pacotes
library(caret)
library(stringr)
library(fastDummies)
library(ROCR)

# Carrega o dataset
dataset = read.table('turnover.csv', header = TRUE, sep = ',')
#View(dataset)
str(dataset)

table(dataset$profession)


###### Pré Processamento ######

### Limpeza ###

# Exlusão colunas
colunas_excluir = c('industry', 'coach', 'greywage', 'traffic')
dataset = dataset[, -which(names(dataset) %in% colunas_excluir)]
str(dataset)


# Exclusão de registros ruídos
table(dataset$age)
dataset = subset(dataset, age>=18, drop = TRUE)


# Troca valor 'Finan\xf1e' for 'Financial' da coluna profession
table(dataset$profession)
dataset$profession = str_replace(dataset$profession, 'Finan\xf1e', 'Financial')
unique(dataset$profession)


# Agrupa profissoes
dataset$group_profession = ifelse(dataset$profession %in% c('Financial', 'Accounting'), 'Financial',
                              ifelse(dataset$profession %in% c('Commercial', 'Sales'), 'Business',
                                ifelse(dataset$profession %in% c('Consult', 'manage'), 'Management',
                                  ifelse(dataset$profession %in% c('Engineer', 'IT'), 'Technology',
                                    ifelse(dataset$profession %in% c('HR', 'Teaching'), 'HR',
                                      ifelse(dataset$profession == 'Law', 'Law',
                                        ifelse(dataset$profession %in% c('Marketing', 'PR'), 'Relations', 'etc')))))))

table(dataset$groupe_profession)


# Transforma variáveis numéricas que são de categoria em dummys
str(dataset)
dataset2 = dummy_cols(dataset, select_columns = c('gender', 'head_gender'), remove_first_dummy = TRUE)
dataset2 = dummy_cols(dataset2, select_columns = c('group_profession', 'way'))
dataset2 = dataset2[, !names(dataset2) %in% c('profession', 'group_profession', 'gender', 'head_gender', 'way')]
str(dataset2)

### Transformação ###

# Função que transforma a variável em tipo fator
to.factor =  function(df, variables){
  for( variable in variables){
    df[[variable]] = as.factor(df[[variable]])
  }
  return(df)
}

# Função que normaliza a variável numérica
scale.feautures = function(df, variables){
  for (variable in variables){
    df[[variable]] = scale(df[[variable]], center = T, scale = T)  #xscaled = (x – média) / desvio padrão
  }
  return(df)
}

# Normaliza as variáveis do tipo número
str(dataset2)
numeric.vars = c('stag', 'age', 'extraversion', 'independ', 'selfcontrol', 'anxiety', 'novator')
vars_num_scaled = scale.feautures(dataset2, numeric.vars)

todas_colunas = colnames(vars_num_scaled)
print(todas_colunas)

# Variáveis do tipo fator
categorical.vars = c('event','gender_m', 'head_gender_m', 'group_profession_Business',
                     'group_profession_etc', 'group_profession_Financial', 'group_profession_HR', 
                     'group_profession_Law', 'group_profession_Management', 'group_profession_Relations',
                     'group_profession_Technology', 'way_bus' ,'way_car', 'way_foot')

# Junta os dois dfs de número padronizado e fator
dataset2 = to.factor(df = vars_num_scaled, variables = categorical.vars)
str(dataset2)
View(dataset2)

###### Criação do modelo de predição ######

# Separa os dados de treino e teste
indexes = sample(1:nrow(dataset2), size = 0.7 * nrow(dataset2))
dataset_train = dataset2[indexes,]
dataset_test = dataset2[-indexes,]
str(dataset_train)
str(dataset_test)
class(dataset_train)
class(dataset_test)

# Separa as variáveis preditores(features_vars) da variável resposta(feature_class)
test_features_vars = subset(dataset_test, select = -event)
test_features_class = dataset_test$event
summary(test_features_class)
dim(test_features_class)

# Constrói modelo de regressão logística
formula.init = 'event ~ .'
formula.init = as.formula(formula.init)
help(glm)
modelo_v1 = glm(formula = formula.init, data = dataset_train, family = 'binomial')

summary(modelo_v1)

# Faz previsões e analisa resultado
help(predict)
previsoes = predict(modelo_v1, dataset_test , type = "response")
previsoes = round(previsoes)

# Matriz de Confusão

# Prepara dados para matriz
test_features_class <- as.vector(test_features_class)
levels(previsoes) <- levels(test_features_class)

help("confusionMatrix")
matriz = confusionMatrix(table(data = previsoes, reference = test_features_class), positive = '1')
matriz

# Seleção das melhores features para o modelo
formula <- "event ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = dataset_train, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)

# Plot de importância das features
plot(importance)

# Cria um novo modelo com features selecionadas
formula2 = 'event ~ age + way_car + way_bus + way_foot + anxiety + stag + extraversion + independ + selfcontrol + anxiety + novator'
formula2 = as.formula(formula2)
modelo_v2 = glm(formula = formula2, data = dataset_train, family = 'binomial')

summary(modelo_v2)
summary(modelo_v1)

previsoes2 = predict(modelo_v2, dataset_test, type = 'response')
previsoes2 = round(previsoes2)

matriz2 = confusionMatrix(table(data = previsoes2, reference = test_features_class), positive = '1')
matriz2


matriz
matriz2
summary(modelo_v1)
summary(modelo_v2)


# Modelo final
modelo_final = modelo_v1
previsoes3 = predict(modelo_final, test_features_vars, type = 'response')
previsoes_finais = prediction(previsoes3, test_features_class)

# Função para Plot ROC
funcao_roc = function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
}

# Plot
par(mfrow = c(1,2))
funcao_roc(previsoes_finais, title = 'Curva ROC')

