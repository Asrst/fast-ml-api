### Inference

- results for loading test are present in `tests/load_test_results/` & their subsequent prediction functions in `app/models/predict.py`
- The simple Joblib prediction functions `joblib_predict` & `predict_async` gave best load test results.
- The parallel prediction function `joblib_predict_parallel` didn't scale well because of smaller batches size.
- Read further for detailed explanations of How & Why.

</div>

<div class="cell code" data-execution_count="6" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:06:58.887072Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:06:58.884620Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:06:58.887669Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:06:58.883912Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:58.859523&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:06:58.887801&quot;,&quot;duration&quot;:2.8278e-2}" data-tags="[]">

``` python
import sklearn
sklearn.__version__, joblib.__version__
```

<div class="output execute_result" data-execution_count="6">

    ('0.23.2', '1.0.0')

</div>

</div>

<div class="cell code" data-execution_count="7" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:06:58.944222Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:06:58.936498Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:06:58.944793Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:06:58.935771Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:58.908529&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:06:58.944927&quot;,&quot;duration&quot;:3.6398e-2}" data-tags="[]">

``` python
saved_models = [pt for pt in os.listdir('.') if pt.endswith('joblib')]
saved_models
```

<div class="output execute_result" data-execution_count="7">

    ['RF.joblib', 'SVC.joblib', 'LR.joblib', 'NB.joblib']

</div>

</div>

<div class="cell code" data-execution_count="8" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:06:59.016928Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:06:59.010007Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:06:59.016428Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:06:59.009137Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:58.969645&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:06:59.017144&quot;,&quot;duration&quot;:4.7499e-2}" data-tags="[]">

``` python
import joblib

def load_sklearn_joblib_model(path, n_jobs = 1):
    # Load from file
    model = joblib.load(path)

    # make n_jobs = 1, to avoid oversubcription.
    # because esemble models like RF,GBM are already parallelized even for predict method.
    # ensemble models build the trees parallely using all cores.
    if 'n_jobs' in model.get_params().keys():
        n_jobs_param = {'n_jobs': n_jobs}
        model.set_params(**n_jobs_param)
        
    return model
```

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:59.042815&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:06:59.073491&quot;,&quot;duration&quot;:3.0676e-2}" data-tags="[]">

### Will Parallelizing the predict method (n\_jobs \> 1) for sklearn improve inference time ?

  - We can use Joblib to parallelize the predict method of an algo if it
    is not already.
  - If it is already parallelized, we can use n\_jobs parameter.

### How to Tell if an algo used sklearn is already parallelized ?

  - A Hack: using CPUs/second. What that means ?

If you ever used %%time magic command, it will give two outputs

`CPU times: user 42.2 ms, sys: 6.23 ms, total: 48.4 ms Wall time: 47 ms`

CPU times: Total time your cpus took (all cores). Wall time: Time took
to run the function (Wait time to get the result).

`CPU/second = CPU times / Wall time`

  - if CPU/s =\~ 1 , it means the algo is running only on single core
    using it 100%.

  - if CPU/s \>\> 1 (significantly), multiple cores are being used i.e.
    parallelized.

  - if CPU/s \<\< 1: The lower the number, the more of its time the
    process spent waiting (for the network, or the harddrive, or locks,
    or other processes to release the CPU, or just sleeping). E.g. if
    CPU/s is 0.75, 25% of the time was spent waiting.

To explore more on this:

<https://pythonspeed.com/articles/blocking-cpu-or-io/>

</div>

<div class="cell code" data-execution_count="9" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:06:59.128996Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:06:59.127376Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:06:59.129812Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:06:59.126500Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:59.097068&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:06:59.129982&quot;,&quot;duration&quot;:3.2914e-2}" data-tags="[]">

``` python
def timer(f):
    """
    A simple python decorator
    to calculate CPU & Wall times
    of any function
    """
    
    import time
    def timed(*args, **kw):
        cs, ws = time.process_time(), time.time()
        result = f(*args, **kw)
        ce, we = time.process_time(), time.time()
        ct, wt = ce-cs, we-ws
        print(f"func: {f.__name__}, CPU/s: {ct/wt:.4f}")
        print(f"CPUtimes: {ct:.4f} s , Walltime: {wt:.4f} s") 
        return result
    return timed


@timer
def joblib_predict(model, X):
    predictions = model.predict(X)
    return predictions.tolist()
```

</div>

<div class="cell code" data-execution_count="10" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:07:51.584468Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:06:59.201586Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:07:51.585207Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:06:59.200782Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:06:59.162624&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:51.585389&quot;,&quot;duration&quot;:52.422765}" data-tags="[]">

``` python
# generate fake data for inference
X_infer, y_infer = datasets.make_classification(n_samples=20000, n_features=10)
print('Input:', X_infer.shape)


for load_path in saved_models:
    print('\nModel: ',load_path)
    clf = load_sklearn_joblib_model(load_path)
    time.sleep(3)
    preds = joblib_predict(clf, X_infer)
    time.sleep(10)
    del clf, preds
```

<div class="output stream stdout">

    Input: (20000, 10)
    
    Model:  RF.joblib
    func: joblib_predict, CPU/s: 0.9998
    CPUtimes: 0.2574 s , Walltime: 0.2574 s
    
    Model:  SVC.joblib
    func: joblib_predict, CPU/s: 1.9948
    CPUtimes: 0.0143 s , Walltime: 0.0071 s
    
    Model:  LR.joblib
    func: joblib_predict, CPU/s: 2.7214
    CPUtimes: 0.0047 s , Walltime: 0.0017 s
    
    Model:  NB.joblib
    func: joblib_predict, CPU/s: 1.0004
    CPUtimes: 0.0043 s , Walltime: 0.0043 s

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:51.623302&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:51.660635&quot;,&quot;duration&quot;:3.7333e-2}" data-tags="[]">

#### From the above results:

  - LR - predict is parallelized.
  - GaussianNB - No n\_jobs & predict method is not parallelized.
  - LinearSVC - No n\_jobs & predict method runs in parallel.
  - RandomForest - parallelized, but can be controlled using n\_jobs.

</div>

<div class="cell code" data-execution_count="11" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:07:51.750605Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:07:51.744091Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:07:51.750029Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:07:51.743178Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:51.697595&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:51.750802&quot;,&quot;duration&quot;:5.3207e-2}" data-tags="[]">

``` python
from sklearn.utils import gen_batches
from joblib import Parallel, delayed, cpu_count
print('num of cpus:', cpu_count())

@timer
def joblib_predict_parallel(model, X, 
                            jb_kwargs = {'prefer': "threads",
                                         'require': "sharedmem"}):
    n_jobs = max(cpu_count(), 1)
    slices = gen_batches(len(X), len(X)//n_jobs)
    parallel = Parallel(n_jobs=n_jobs, **jb_kwargs)
    results = parallel(delayed(model.predict)(X[s]) for s in slices)
    return np.vstack(results).flatten().tolist()
```

<div class="output stream stdout">

    num of cpus: 4

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:51.776650&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:51.801921&quot;,&quot;duration&quot;:2.5271e-2}" data-tags="[]">

  - Since, The predict method of NB is not parallelized, I will try to
    use Joblib's Parallel & delayed functions to make it Parallel across
    all the cores.

  - What happens if try to parallelize the code which is already runs in
    all cores ? It results in oversubcriptions of threads, increasing
    overhead. So the function runs even slower.

</div>

<div class="cell code" data-execution_count="12" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:07:54.974618Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:07:51.857825Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:07:54.975733Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:07:51.857224Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:51.826524&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:54.975897&quot;,&quot;duration&quot;:3.149373}" data-tags="[]">

``` python
load_path = 'NB.joblib'
print('\nModel: ', load_path)
clf = load_sklearn_joblib_model(load_path)
time.sleep(3)
preds = joblib_predict_parallel(clf, X_infer)
assert len(preds) == len(X_infer)
del clf, preds
```

<div class="output stream stdout">

``` 

Model:  NB.joblib
func: joblib_predict_parallel, CPU/s: 0.1467
CPUtimes: 0.0157 s , Walltime: 0.1071 s
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:55.002445&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:07:55.027460&quot;,&quot;duration&quot;:2.5015e-2}" data-tags="[]">

  - One thing to notice: CPU/s is increased only slightly (i.e only
    using all cores partially), Parallelized NB still takes way more
    time than original one (0.10 \> 0.0047) for batch size of 20k.

  - This is due to the overhead of slicing data & accumulating the
    results. Also, sklearn models are mostly optimized for larger
    batches. So let us check at what is minimum batch size to use to
    leverage power of parallelzing.

</div>

<div class="cell code" data-execution_count="13" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.051111Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:07:55.089779Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.050141Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:07:55.088887Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:07:55.053356&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.051326&quot;,&quot;duration&quot;:339.99797}" data-tags="[]">

``` python
for ns in [10, 100, 10**3, 10**4, 10**5, 10**6, 10**7]:
    load_path = 'NB.joblib'
    print('Model: ', load_path)
    X_infer, _ = datasets.make_classification(n_samples=ns, n_features=10)
    
    # direct predict
    print('\nInput:', X_infer.shape, 'Parallel: ', False)
    clf = load_sklearn_joblib_model(load_path)
    time.sleep(5)
    preds = joblib_predict(clf, X_infer)
    time.sleep(10)
    del clf, preds
    
    # parallel predict threads
    print('\nInput:', X_infer.shape, 'Parallel: ', 'threads')
    clf = load_sklearn_joblib_model(load_path)
    time.sleep(5)
    preds = joblib_predict_parallel(clf, X_infer)
    assert len(preds) == len(X_infer)
    time.sleep(10)
    del clf, preds
    
    # parallel predict processes
    print('\nInput:', X_infer.shape, 'Parallel: ', 'processes')
    clf = load_sklearn_joblib_model(load_path)
    time.sleep(5)
    preds = joblib_predict_parallel(clf, X_infer, jb_kwargs = {'prefer': 'processes'})
    assert len(preds) == len(X_infer)
    print('-'*50)
    time.sleep(10)
    del clf, preds
```

<div class="output stream stdout">

    Model:  NB.joblib
    
    Input: (10, 10) Parallel:  False
    func: joblib_predict, CPU/s: 1.0362
    CPUtimes: 0.0004 s , Walltime: 0.0004 s
    
    Input: (10, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 0.0757
    CPUtimes: 0.0078 s , Walltime: 0.1034 s
    
    Input: (10, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.8672
    CPUtimes: 0.0135 s , Walltime: 0.0156 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (100, 10) Parallel:  False
    func: joblib_predict, CPU/s: 1.0074
    CPUtimes: 0.0006 s , Walltime: 0.0006 s
    
    Input: (100, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 0.0625
    CPUtimes: 0.0065 s , Walltime: 0.1035 s
    
    Input: (100, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.9156
    CPUtimes: 0.0093 s , Walltime: 0.0101 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (1000, 10) Parallel:  False
    func: joblib_predict, CPU/s: 1.0058
    CPUtimes: 0.0006 s , Walltime: 0.0006 s
    
    Input: (1000, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 0.1046
    CPUtimes: 0.0108 s , Walltime: 0.1034 s
    
    Input: (1000, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.9527
    CPUtimes: 0.0134 s , Walltime: 0.0141 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (10000, 10) Parallel:  False
    func: joblib_predict, CPU/s: 1.0012
    CPUtimes: 0.0024 s , Walltime: 0.0024 s
    
    Input: (10000, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 0.1427
    CPUtimes: 0.0147 s , Walltime: 0.1028 s
    
    Input: (10000, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.9013
    CPUtimes: 0.0129 s , Walltime: 0.0143 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (100000, 10) Parallel:  False
    func: joblib_predict, CPU/s: 0.9906
    CPUtimes: 0.0438 s , Walltime: 0.0442 s
    
    Input: (100000, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 0.4356
    CPUtimes: 0.0454 s , Walltime: 0.1043 s
    
    Input: (100000, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.1493
    CPUtimes: 0.0203 s , Walltime: 0.1359 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (1000000, 10) Parallel:  False
    func: joblib_predict, CPU/s: 0.9995
    CPUtimes: 0.3963 s , Walltime: 0.3965 s
    
    Input: (1000000, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 1.5257
    CPUtimes: 0.3618 s , Walltime: 0.2371 s
    
    Input: (1000000, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.3004
    CPUtimes: 0.0982 s , Walltime: 0.3269 s
    --------------------------------------------------
    Model:  NB.joblib
    
    Input: (10000000, 10) Parallel:  False
    func: joblib_predict, CPU/s: 0.9996
    CPUtimes: 2.2422 s , Walltime: 2.2430 s
    
    Input: (10000000, 10) Parallel:  threads
    func: joblib_predict_parallel, CPU/s: 2.3908
    CPUtimes: 3.4988 s , Walltime: 1.4634 s
    
    Input: (10000000, 10) Parallel:  processes
    func: joblib_predict_parallel, CPU/s: 0.3858
    CPUtimes: 0.6902 s , Walltime: 1.7892 s
    --------------------------------------------------

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.115772&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.176841&quot;,&quot;duration&quot;:6.1069e-2}" data-tags="[]">

  - only at batch size \>= 10\*\*6, the parallelizing is showing some
    use. But this large batch sizes are not realistic to get in real world cases.
  - But one important thing is CPU/s is \<\< 1 for lower/mini batches
    i.e indicating that CPU is waiting most of the time, can we levarge
    that somehow ?

</div>

<div class="cell code" data-execution_count="14" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.444701Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:35.305284Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.444185Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:35.304444Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.238859&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.444872&quot;,&quot;duration&quot;:0.206013}" data-tags="[]">

``` python
X_infer, _ = datasets.make_classification(n_samples=100000, n_features=10)
X_infer.shape
```

<div class="output execute_result" data-execution_count="14">

    (100000, 10)

</div>

</div>

<div class="cell code" data-execution_count="15" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.554598Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:35.531084Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.555559Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:35.530392Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.485028&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.555765&quot;,&quot;duration&quot;:7.0737e-2}" data-tags="[]">

``` python
clf = load_sklearn_joblib_model(load_path)
r = joblib_predict(model = clf, X = X_infer)
# print(len(r))
```

<div class="output stream stdout">

    func: joblib_predict, CPU/s: 1.0338
    CPUtimes: 0.0202 s , Walltime: 0.0195 s

</div>

</div>

<div class="cell code" data-execution_count="16" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.690004Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:35.685082Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.689333Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:35.684221Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.618623&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.690255&quot;,&quot;duration&quot;:7.1632e-2}" data-tags="[]">

``` python
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio

def blocking_io():
    # File operations (such as logging) can block the
    # event loop: run them in a thread pool.
    with open('/dev/urandom', 'rb') as f:
        return f.read(100)


async def joblib_predict_async(**kwargs):
    loop = asyncio.get_running_loop()
    # func_args = {'model': clf, 'X': X_infer}
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, partial(joblib_predict, **kwargs))
        # print('thread pool', len(result))
    return result
```

</div>

<div class="cell code" data-execution_count="17" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.852241Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:35.825286Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.854557Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:35.824401Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.753536&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.854725&quot;,&quot;duration&quot;:0.101189}" data-tags="[]">

``` python
clf = load_sklearn_joblib_model(load_path)
r = await joblib_predict_async(model = clf, X = X_infer)
# print(len(r))
```

<div class="output stream stdout">

    func: joblib_predict, CPU/s: 1.0408
    CPUtimes: 0.0227 s , Walltime: 0.0218 s

</div>

</div>

<div class="cell code" data-execution_count="18" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:35.948833Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:35.946562Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:35.949689Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:35.945899Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.895753&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:35.949840&quot;,&quot;duration&quot;:5.4087e-2}" data-tags="[]">

``` python
@timer
async def predict_parallel_async(model, X):
    loop = asyncio.get_event_loop()
    n_jobs = max(cpu_count(), 1)
    slices = gen_batches(len(X), len(X)//n_jobs)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        tasks = [loop.run_in_executor(pool, model.predict, X[s]) for s in slices]

    completed, pending = await asyncio.wait(tasks)
    results = []
    for t in completed:
        results.extend(t.result().tolist())
    return results
```

</div>

<div class="cell code" data-execution_count="19" data-execution="{&quot;shell.execute_reply&quot;:&quot;2021-02-14T21:13:36.064444Z&quot;,&quot;iopub.execute_input&quot;:&quot;2021-02-14T21:13:36.042411Z&quot;,&quot;iopub.status.idle&quot;:&quot;2021-02-14T21:13:36.068785Z&quot;,&quot;iopub.status.busy&quot;:&quot;2021-02-14T21:13:36.041751Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:35.992290&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:36.068929&quot;,&quot;duration&quot;:7.6639e-2}" data-tags="[]">

``` python
s= time.time()
clf = load_sklearn_joblib_model(load_path)
r = await predict_parallel_async(model = clf, X = X_infer)
print(time.time() - s)
print(len(r))
```

<div class="output stream stdout">

    func: predict_parallel_async, CPU/s: 2.2230
    CPUtimes: 0.0000 s , Walltime: 0.0000 s
    0.02070903778076172
    100000

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:36.110689&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:36.151819&quot;,&quot;duration&quot;:4.113e-2}" data-tags="[]">

  - Asyncio also didn't help to improve inference times at smaller
    batches (expected as model.predict is cpu bound.)

  - Joblib natively wont support async/await.

### WHY PARALLELIZING DIDN'T HELP ?

#### Training:

1.  Both Thread Based & Process Based Parallelism can help. Why ?

<!-- end list -->

  - Because Training is a iterative process, the observarions must be
    used for train the algo multiple times. So in this case, With little
    overhead of creating process & sharing data the training can be
    faster.

  - Also Training is done on large amount of data, so large batch sizes
    helps. (sklearn & numpy are optimized for matrix/larger batch
    calculations)

#### Predictions:

  - Prediction is a single step process (we only use observation once to
    get prediction), So using process based parallelism adds
    unneccessary overhead and will not help in most cases.

  - Usually predictions/inference happens on smaller set of data. Thread
    based parallelism can help to speed up the algo but if inference
    data is small then overhead of creating threads can overtake normal
    execution time. so If inference is running on smaller batches/single
    observation it may not improve the results.

  - <https://github.com/scikit-learn/scikit-learn/issues/7448>

  - <https://github.com/scikit-learn/scikit-learn/pull/16310>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2021-02-14T21:13:36.193886&quot;,&quot;end_time&quot;:&quot;2021-02-14T21:13:36.235429&quot;,&quot;duration&quot;:4.1543e-2}" data-tags="[]">

### Solution to Faster & Scalable Machine Learning Inference APIs:

This can be tricky to answer without excat suitation but here are some
options to try.

1.  Make code/algo simpler so that it can run the predictions for
    single/mini-batch observation(s) in minimal possible time. Further,
    combine this ASGI server (reduces I/O time) with multiple workers
    for better request latency.

some framework(s) doing this:

1.  onnx (supports sklearn, xgboost & lgbm)
2.  river

<https://cloudblogs.microsoft.com/opensource/2020/12/17/accelerate-simplify-scikit-learn-model-inference-onnx-runtime/>

OR

1.  Use a Queue to collect the request, make predictions in batches &
    return response for all at once. Further, this can be also combined
    with ASGI server for better latency.

some framework(s) doing this:

1.  tf-serving (tensorflow)
2.  BentoML (all major ml frameworks)
3.  Clipper

OR

3.Other Alternative(s):

  - Using Distrubuted systems (Spark, Dask)
  - Humming Bird (<https://github.com/microsoft/hummingbird>)
  - Processor specific accelrators (like Intel's OpenVINO)
  - Using GPU based infernece (TensorRT)

</div>
