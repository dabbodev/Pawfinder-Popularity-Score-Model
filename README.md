# Petfinder Pawpularity Score Model
 
This keras model uses a pre-trained resnet50 model at it's base in conjunction with 15 independant convolution-based binary feature encoders to detect various meta-tagged features possibly related to the popularity of photos of shelter animals online. Once the binary layer is encoded it is fed into a rendering layer that creates a base score and modifier value from the results of the binary detection layer. A final layer weights these together to create a final predicted "paw"pularity score.
