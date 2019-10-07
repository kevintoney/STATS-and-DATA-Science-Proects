#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(leaflet)
library(dplyr)
library(leaflet.extras)
library(ggmap)
library(ggplot2)
library(shinyWidgets)
library(stringr)
#library(shinyapps)



#rm(list=ls())
#setwd("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/NFL Careers/NFLBirthState/")
pl1 <- read.csv("playersLonLat.csv", header=T, sep=",", na.strings = "")
pl2<- read.csv("playersLonLat2.csv", header=T, sep=",", na.strings = "")
pl3<- read.csv("playersLonLat3.csv", header=T, sep=",", na.strings = "")
pl4<- read.csv("playersLonLat4.csv", header=T, sep=",", na.strings = "")
players <- rbind(pl1,pl2,pl3,pl4)
tbl <- with(players, table(birth_state))

#playersSub <- players
#playersSub <- playersSub[!is.na(playersSub$lat),]

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Birth States of NFL Players"),
  
  #User dropbox
  sidebarLayout(position="right",
                sidebarPanel("Filters",
                             selectInput("year", "Choose Draft Year", choices=sort(unique(players$draft_year))),
                             searchInput("lastname",   label = "Search for a Player :", 
                                         placeholder = "Last Name", 
                                         btnSearch = icon("search"), 
                                         btnReset = icon("remove"))),
                mainPanel( 
                  #this will create a space for us to display my map
                  h4("Map of Players' Birth States"),
                  leafletOutput(outputId = "mymap"),
                  br(),
                  h4("Number of Players Born in Each State"),
                  plotOutput("plotGG"),
                  h5("Citation of GGMap for the Interactive Map"),
                  p("D. Kahle and H. Wickham. ggmap: Spatial Visualization with
                    ggplot2. The R Journal, 5(1), 144-161. URL
                    http://journal.r-project.org/archive/2013-1/kahle-wickham.pdf")
                  )
                  )
                )


# Define server logic required to draw a histogram
server <- function(input, output) {
  
  playersSub <- reactive({
    if(input$lastname ==""){
      a <- subset(players, draft_year == input$year)
      a <- droplevels(a)
      return(a)
    } else{
      
      c <- subset(players, last_name %in% str_to_title(input$lastname))
      c <- droplevels(c)
      return(c)
    }
  })
  
  
  output$mymap <- renderLeaflet({
    leaflet(playersSub()) %>% 
      setView(lng = -99, lat = 45, zoom = 4)  %>% #setting the view over ~ center of North America
      addTiles() %>%
      addCircles(data = playersSub(), lat = ~ latitude, lng = ~ longitude, weight = 5, radius = 20000, fill=T, fillOpacity = 1,
                 popup = paste("Name:", playersSub()$name, "<br>",
                               "Birth City:", playersSub()$birth_city, "<br>",
                               "Position:", playersSub()$position, "<br>",
                               "Draft Team:", playersSub()$draft_team, "<br>",
                               "Draft Round:", playersSub()$draft_round, "<br>",
                               "First Year:", playersSub()$draft_year, "<br>",
                               "Career Length:", playersSub()$careerLength, "<br>",
                               "Google Search:", paste0("<a href='","https://www.google.com/search?q=", playersSub()$first_name, "+", playersSub()$last_name,"'>","Google", "</a>", sep="")))
  })
  
  output$plotGG <- renderPlot({
    ggplot(as.data.frame(tbl), aes(factor(birth_state), Freq))+geom_col(position = 'dodge')+theme(axis.text.x=element_text(angle=90, vjust=0.5)) + xlab("State") + ylab("Count")
  })
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)

