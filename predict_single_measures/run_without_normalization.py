from .train_resnet_for_single_measure import train_for_single_measure





#for measure in range(9):
#    for lr in [5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]:
#        print(measure)
#        print(lr)
#        train_for_single_measure(measure = measure, path = "predict_single_measures/results_no_norm/", lr = lr, gamma = 0.95, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 50)

#train_for_single_measure(measure = 0, path = "predict_single_measures/results_no_norm/", lr = 5e-4, gamma = 0.7, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 50, dropout = False)
#train_for_single_measure(measure = 0, path = "predict_single_measures/results_no_norm/", lr = 5e-4, gamma = 0.6, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 50, dropout = False)
#train_for_single_measure(measure = 0, path = "predict_single_measures/results_no_norm/", lr = 5e-4, gamma = 0.5, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 50, dropout = False)

#for measure in range(1,9):
#    train_for_single_measure(measure = measure, path = "predict_single_measures/results_no_norm_small_dataset/", lr = 1e-3, gamma = 0.95, batchsize = 1, optimizer = "ADAM", weight_decay = 0.0,epochs = 200, dropout = False)

#for measure in range(9):
#    for lr in [1e-3,5e-4,1e-4]:
#        train_for_single_measure(measure = measure, path = "predict_single_measures/results_no_norm_full_dataset/", lr = lr, gamma = 0.92, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 50, dropout = False)


for measure in range(5,9):
    for lr in [1e-3]:
        train_for_single_measure(measure = measure, path = "predict_single_measures/results_no_norm_full_dataset/", lr = lr, gamma = 0.9, batchsize = 100, optimizer = "ADAM", weight_decay = 0.0,epochs = 15, dropout = False)
