# PGGAN

# method 1
 Constructing a whole network. Training lower layers weight and freeze higher layers weight. Gradually training higher layer. (failed)

# method 2
Using transfer learning concept. We trained low pixel network and then output model. Next, add new layer into output trained model and train,and so on. (failed)

# warning 
Because of dead line, we abandoned this approach. Code are fixed many time.
Although the code may execute wrong, we just want to prove that we make an error in this way. 
