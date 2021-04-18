from exp_main import *
from data.NYU.corrupt import *

# main()

for pos in range(25):
    train_model = train(start_pos=1)
    test(model_path =train_model, search_opt=True)