library(data.table)

setwd("~/Intern/Projekte/Blog/RGF")

files <- list.files("data", pattern = ".RData")

rdata_to_csv <- function(path) {
  
  e <- new.env(parent = emptyenv())
  load(paste0("data/", path), envir = e)
  objs <- ls(envir = e, all.names = TRUE)
  
  for (i in objs) {
    .x <- get(i, envir = e)
    if (any(class(.x) == "data.frame")) {
      fwrite(.x, paste0("data/", substr(path, 1, nchar(path) - 6), "_", i,".csv"))
    }
  }
  rm(e)
}

sapply(files, rdata_to_csv)
