library(ggmap)


#rm(list=ls())
setwd("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/NFL Careers/")
players <- read.csv("players_2013-12-12.csv", header=T, sep=",", na.strings = "")
register_google('AIzaSyAxC18SN0lS8lDTIbzcHHUii8hYo0ceTYA', account_type = "standard")
players$careerLength <- players$year_end - players$year_start

playersSub <- players[!is.na(players$birth_city),]
playersSub <- playersSub[16803:20932,]
lonlat <- geocode(as.character(playersSub$birth_city))
playersSub$latitude <- lonlat$lat
playersSub$longitude <- lonlat$lon
playersSub <- playersSub[!is.na(playersSub$lat),]
write.csv(playersSub, "NFLBirthState/playersLonLat4.csv")