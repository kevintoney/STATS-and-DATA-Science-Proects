locateCity <- function(city, state) {
  url = paste('http://nominatim.openstreetmap.org/search?',
              'city=',  gsub(' ', '%20', city), 
              '&state=', gsub(' ', '%20', state),
              '&country=USA',
              '&limit=1&format=json',
              sep=""
  )
  resOSM = RJSONIO::fromJSON(url)
  if(length(resOSM) > 0) {
    return(c(resOSM[[1]]$lon, resOSM[[1]]$lat))
  } else return(rep(NA,2)) 
}

#version 2
setwd("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/NFL Careers/")
players <- read.csv("players_2013-12-12.csv", header=T, sep=",", na.strings = "")
register_google('AIzaSyAxC18SN0lS8lDTIbzcHHUii8hYo0ceTYA', account_type = "standard")

playersSub <- players[!is.na(players$birth_city),]


getGeoDetails <- function(address){   
  api <- 'AIzaSyAxC18SN0lS8lDTIbzcHHUii8hYo0ceTYA'
  register_google(api, account_type = "standard")
  #use the gecode function to query google servers
  geo_reply = geocode(address, output='all', messaging=TRUE, override_limit=TRUE)
  #now extract the bits that we need from the returned list
  answer <- data.frame(lat=NA, long=NA, accuracy=NA, formatted_address=NA, address_type=NA, status=NA)
  answer$status <- geo_reply$status
  #if we are over the query limit - want to pause for an hour
  while(geo_reply$status == "OVER_QUERY_LIMIT"){
    print("OVER QUERY LIMIT - Pausing for 1 hour at:") 
    time <- Sys.time()
    print(as.character(time))
    Sys.sleep(60*60)
    geo_reply = geocode(address, output='all', messaging=TRUE, override_limit=TRUE)
    answer$status <- geo_reply$status
  }
  #return Na's if we didn't get a match:
  if (geo_reply$status != "OK"){
    return(answer)
  }   
  #else, extract what we need from the Google server reply into a dataframe:
  answer$lat <- geo_reply$results[[1]]$geometry$location$lat
  answer$long <- geo_reply$results[[1]]$geometry$location$lng   
  if (length(geo_reply$results[[1]]$types) > 0){
    answer$accuracy <- geo_reply$results[[1]]$types[[1]]
  }
  answer$address_type <- paste(geo_reply$results[[1]]$types, collapse=',')
  answer$formatted_address <- geo_reply$results[[1]]$formatted_address
  return(answer)
}

mapData <- data.frame(NULL)

for(i in 1:100){
  mapData[i,] <- getGeoDetails(as.character(playersSub$birth_city[i]))
}