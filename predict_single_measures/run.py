from .train_resnet_for_single_measure import train_for_single_measure





#for measure in range(9):
#    for lr in [5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]:
#        if not (measure == 0 and lr == 5e-3):
#            print(measure)
#            print(lr)
#            train_for_single_measure(measure = measure, path = "predict_single_measures/results/", lr = lr, gamma = 0.95, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0001,epochs = 50)

train_for_single_measure(measure = 0, path = "predict_single_measures/results/", lr = 5e-4, gamma = 0.7, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0001,epochs = 50)
train_for_single_measure(measure = 0, path = "predict_single_measures/results/", lr = 5e-4, gamma = 0.6, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0001,epochs = 50)
train_for_single_measure(measure = 0, path = "predict_single_measures/results/", lr = 5e-4, gamma = 0.5, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0001,epochs = 50)