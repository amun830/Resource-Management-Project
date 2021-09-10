#################### Resource Consent Project [Contamination in the Onehunga Aquifer] ####################

-   These functions are intended to aid our investigation into the effect of extraction from the Onehunga Aquifer 
    on the pressure and copper concentration, and hence help make an educated recommendation to the Auckland Regional 
    Council regarding the outcome of Watercare’s consent application.

##### Summary #####
-   We created our LPM model of the aquifer pressure and copper concentration and optimised the unknown parameters by 
    minimising the sum of squares.
    
-   From this we then considered five different "what-if" scenarios:
    -   Stop operation in Onehunga Aquifer.
    -   Decrease extraction rate (5ML/day).
    -   Decrease extraction rate less (7.5ML/day).
    -   Continue operating at current rate (20ML/day).
    -   Double extraction rate (40 ML/day).

-   We then conduct uncertanty analysis by computing parameter posterior distribution of pressure and copper models (normal). 

-   This is used to fit to data and hence create samples for each senario. Then plot.

