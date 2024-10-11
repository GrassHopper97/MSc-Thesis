
##Import libraires

library(rgdal)
library(gstat)
library(geoR)
library(sf)
library(sp)
library(terra)
library(ggplot2)
library(gridExtra)
library(grid)
library(MASS)

## Prediction Grid for the kriging

extn_d = extent(c(800, 1100, 3180, 3500)) ## Change it to desire values of UTM Coordinates
pix_siz = 25 ##Spatial Resolution in Km
pix_siz_d = pix_siz/103 # in Degrees

grid_3 = raster(extn_d, res = c(pix_siz,pix_siz))

## Convert it to Spatial DataFrame object and add desired crs

grid_3sp = as(grid_3,"SpatialPixelsDataFrame")
proj4string(grid_3sp) = CRS("+proj=utm +zone=43 +datum=WGS84 +units=km +no_defs")
colnames(grid_3sp@coords)  = c("lon","lat")

### Statistical Analysis, Variogram Generation and Kriging is shown here with one days data, can be re iterate for the whole months data

f1 = "/data/private/GPM/Data_1/July_2023_with_elev/processed_1_July_23.csv" ## Dataset

## Data Transformation

d1 = read.csv(f1)

# Perform Box-Cox transformation on rainfall data
rain <- d1$RAINFALL.DAILY.CUMULATIVE..0.5.mm.or.more.
bc_rn <- boxcox(rain ~ 1)
lambda <- bc_rn$x[which.max(bc_rn$y)]

# Transform the rainfall data using the optimal lambda
rain_transformed <- (rain ^ lambda - 1) / lambda
  
d1$Rainfall_Transformed <- rain_transformed


# Perform Box-Cox transformation on elevation data
 elev <- d1$Elevation
 bc_el <- boxcox(elev ~ 1)
 lambda_el <- bc_el$x[which.max(bc_el$y)]
  
# Transform the elevation data using the optimal lambda
  el_transformed <- (elev^ lambda_el - 1) / lambda_el

  d1$Elevation_Transformed <- el_transformed


## Convert the dataset to Spatial Data Frame

coordinates1 = data.frame(lon = d1$Longitude, lat = d1$Latitude)
coords1 = SpatialPoints(coordinates1)# make it Spatial Pixels
d1_modb = subset(d1, select = -c(Longitude,Latitude))
sd_1 = SpatialPointsDataFrame(coords1,d1_modb)
class(sd_1)
proj4string(sd_1) =CRS("+init=epsg:4326")
desired_crs <- "+proj=utm +zone=43 +datum=WGS84 +units=km +no_defs" #should be same with the prediction grid
sd_1_prjtd <- spTransform(sd_1, CRS(desired_crs))
proj4string(sd_1_prjtd)


## Variogram Modelling

cord = coordinates(sd_1_prjtd)
rn = sd_1_prjtd$RAINFALL.DAILY.CUMULATIVE..0.5.mm.or.more.
rn = data.frame(rn)
gdt_1 = as.geodata(data.frame(cord, rainfall = rn))
jit_am = 1e-3 ## Need to remove the similar coordinates. In our case, it was needed few times
gdt_1 = jitterDupCoords(gdt_1, max = jit_am)

vf = variog(gdt_1, lambda = 0.141414141414141, trend = "cte") ## get the lambda value from the earlier step
plot(vf)
vgm_rn_lk = likfit(geodata = gdt_1, trend = "1st",cov.model = "exponential", ini.cov.pars = c(3,50),nugget = 1, lambda = 0.141414141414141, lik.method = "ML", messages = FALSE)
vgm_rn_lk

## Kriging

grid_coords <- coordinates(grid_3sp)
grid_df <- data.frame(grid_coords)
names(grid_df) <- c("lon", "lat") 

krig_lkft_1 <- krige.conv(gdt_1, locations = grid_df, krige = krige.control(obj.model = vgm_rn_lk))

krig_rain_lkft_1 <- krig_lkft_1$predict
# Create a new raster based on the grid and assign the kriging values
pr_rain_lkft_1 <- rasterFromXYZ(cbind(grid_df, krig_rain_lkft_1))
proj4string(pr_rain_lkft_1) <- proj4string(grid_3sp)
plot(pr_rain_lkft_1)


## Regression

train.x <- data.frame(el_transformed)  # Explanatory variable
RESPONSE <- rain_transformed  # Target variable

# Set up the train control
myControl <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

GLM <- train(train.x,
             RESPONSE,
             method = "glm",
             trControl = myControl,
             preProc = c('center', 'scale'))  # Center and scale the data

# Print the model summary
print(GLM)
var = predict(GLM) #predicted vzlues from GLM model

## use the outputs from GLM model to generate variogram and perform kriging. For Kriging with the regression outputs, the trend should be linear, not constant. 

### Some Related tests and checks

corr_mat = cor(rain_transformed, elev, use = 'complete') ## Checking correlation between these two variables
scatter_plot <- ggplot(d1, aes(x = elev, y = rain_transformed)) +
  geom_point() +
  theme_minimal() +
  labs(x = "Ele", y = "Rain")
scatter_plot


