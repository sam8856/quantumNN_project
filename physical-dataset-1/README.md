#README
##Dataset
The choosen dataset contains 21 columns and 4214 examples. The name of the dataset is `Solar energy power generation dataset` and it can be found under [kaggle][1](13.02.2023-08:36-MEZ). The first 20 parameters are physical desription of the environment while the last parameter the power output of the solar panel(s) quantifies. The parameters are set as followed:
ParamNumber|Description|simpleNN|reducedNN|autoencoder|
---|---|---|---|---|
1 |temperature at 2m above ground | x | |
2 |relative humidity at 2m above ground | x | |
3 |mean sea level pressure | x | |
4 |total precipitation surface | x | |
5 |snowfall amount surface | x | |
6 |total cloud cover surface | x | |
7 |high cloud cover, high cld lay | x | |
8 |medium cloud cover, mid cld lay | x | |
9 |low cloud cover, low cld lay | x | |
10 |shortwave radiation backwards surface | x | |
11 |windspeed at 10m above ground | x | |
12 |wind direction at 10m above ground | x | |
13 |windspeed at 80 above ground | x | |
14 |wind direction at 80m above ground | x | |
15 |windspeed at 900m above ground | x | |
16 |wind direction at 900m above ground | x | |
17 |wind gust at 10m above ground | x | |
18 |angle of incidence (sun) | x | |
19 |zenith | x | |
20 |azimuth | x | |
21 |generated power output in kW | x | |
    
Due to the missing description and informations in the dataset, not all parameters are completely clear in their meaning, for example the surface-parameters. These parameters are **all** used for a first approach model, which is expected to work quite bad. Afterwards an reduced model will be used and an autoencoder is build up, both first as a *normal* ANN. Later we will try to do the same with QNN's and compare the outcome. If there are difference in learning, we will consider coming up with a mix Q/N-Boost combination of both model types.






[1]: https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset