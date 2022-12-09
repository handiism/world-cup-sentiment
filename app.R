library(tidyverse)
library(e1071)
library(caret)
library(wordcloud)
library(dplyr)
library(tm)
library(ggplot2)
library(shiny)


dataset <- read.csv("/home/handiism/Developments/R/ds/world-cup-qatar.csv")

ui <- fluidPage(
  titlePanel("World Cup Qatar Sentiment Analysis"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("n",
                  "Dataset Size",
                  value = nrow(dataset),
                  min = 20,
                  max = nrow(dataset))

    ),
    mainPanel(
      tabsetPanel(type = "tabs",
                  tabPanel("Data",dataTableOutput("table")),
                  tabPanel("Pie Chart", plotOutput("piechart")),
                  tabPanel("Word Cloud", plotOutput("wordcloud",width = "100%")),
                  tabPanel("Confusion Matrix", verbatimTextOutput("metrics")),
      )
    )
  )
)

server <- function(input, output) {
  r <- reactive({
    n <- input$n
  })
  
  output$wordcloud <- renderPlot({
    nDataset <- dataset[sample(nrow(dataset),input$n),]
    wordcloud(nDataset$tweet, max.words = 100, random.color = TRUE, scale = c(2.2,1),colors = colors())
  })
  
  output$piechart <- renderPlot({
    nDataset <- dataset[sample(nrow(dataset),input$n),]
    proportion <- round(as.numeric(prop.table(table(nDataset$sentiment))),2)
    sentiment <- c("Negative", "Neutral", "Positive")
    pie(proportion, labels = proportion,
        main = "Sentiment Pie Chart",
        col = rainbow(length(proportion)))
    legend("topright",sentiment,fill = rainbow(length(proportion)))
  })
  
  output$table <- renderDataTable({
    nDataset <- dataset[sample(nrow(dataset),input$n),]
    colnames(nDataset) <- c("ID","Tweet","Sentiment")
    nDataset
  },options = list(
    pageLength = 10
  ))
  
  output$metrics <- renderPrint({
    data <- dataset[sample(nrow(dataset),input$n),]
    
    dtm <- DocumentTermMatrix(data$tweet)
    
    # In the sense of the sparse argument to removeSparseTerms(), sparsity
    # refers to the threshold of relative document frequency for a term,
    # , above which the term will be removed.
    
    # Sparsity refers to a matrix of numbers that includes many zeros or values
    # that will not significantly impact a calculation.
    dtm <- removeSparseTerms(dtm, 0.999)
    
    convert <- function(x) {
      y <- ifelse(x > 0, 1, 0)
      y <- factor(y, levels = c(0, 1), labels = c("No", "Yes"))
      y
    }
    
    dataNaive <- apply(dtm, 2, convert)
    
    dataset <- as.data.frame(as.matrix(dataNaive))
    dataset$sentiment <- as.factor(data$sentiment)
    
    split <- sample(2, nrow(dataset), prob = c(0.75, 0.25), replace = TRUE)
    train <- dataset[split == 1,]
    test <- dataset[split == 2,]
    
    # Bayes' theorem menawarkan suatu formula untuk menghitung nilai
    # probability dari suatu event dengan memanfaatkan pengetahuan
    # sebelumnya dari kondisi terkait; atau sering kali dikenal
    # dengan istilah conditional probability.
    control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
    classifier <- naiveBayes(train, train$sentiment, laplace = 1, trControl = control, tuneLength = 7)
    
    prediction <- predict(classifier, type = 'class', newdata = test)
    confusionMatrix(prediction, test$sentiment)
  })
}

shinyApp(ui, server)