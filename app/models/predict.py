
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed, cpu_count
from sklearn.utils import gen_batches
from functools import partial
import numpy as np
import asyncio


def joblib_predict(model, X):
    """
    simple predict method using joblib 
    """
    predictions = model.predict(X)
    return predictions.tolist()

async def predict_async(**kwargs):
    """
    async version of joblib_predict
    """
    loop = asyncio.get_running_loop()
    # func_args = {'model': clf, 'X': X_infer}
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, partial(joblib_predict, **kwargs))
        # print('thread pool', len(result))
    return result


def joblib_predict_parallel(model, X, 
                            jb_kwargs={'prefer': "threads"}):
    """
    parallel predict method using joblib 
    """
    n_jobs = max(cpu_count(), 1)
    slices = gen_batches(len(X), len(X)//n_jobs)
    parallel = Parallel(n_jobs=n_jobs, **jb_kwargs)
    results = parallel(delayed(model.predict)(X[s]) for s in slices)
    return np.vstack(results).flatten().tolist()

async def predict_parallel_async(model, X):
    """
    async version of joblib_predict_parallel
    """

    loop = asyncio.get_event_loop()
    n_jobs = max(cpu_count(), 1)
    slices = gen_batches(len(X), len(X)//n_jobs)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        tasks = [loop.run_in_executor(pool, model.predict, X[s]) for s in slices]
    completed, pending = await asyncio.wait(tasks)
    results = [t.result() for t in completed]
    return np.vstack(results).flatten().tolist()




