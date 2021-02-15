
import joblib

def load_sklearn_joblib_model(path):
    # Load from file
    model = joblib.load(path)
    # make n_jobs = 1, to avoid oversubcription.
    # because esemble models like RF,GBM are already parallelized even for predict method.
    # ensemble models build the trees parallely using all cores.
    if 'n_jobs' in model.get_params().keys():
        n_jobs_param = {'n_jobs':1}
        model.set_params(**n_jobs_param)
    return model

