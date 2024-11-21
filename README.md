# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

<h2>Parallel Check Output</h2>

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/content/mod3-mattmaitland1/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-mattmaitland1/minitorch/fast_ops.py (163)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        # Calculate total size                                           |
        size = 1                                                         |
        for i in range(len(out_shape)):                                  |
            size *= out_shape[i]                                         |
                                                                         |
        # Main loop                                                      |
        for i in prange(size):-------------------------------------------| #0
            out_index = np.empty(len(out_shape), np.int32)               |
            in_index = np.empty(len(in_shape), np.int32)                 |
                                                                         |
            to_index(i, out_shape, out_index)                            |
            broadcast_index(out_index, out_shape, in_shape, in_index)    |
                                                                         |
            o = index_to_position(out_index, out_strides)                |
            j = index_to_position(in_index, in_strides)                  |
                                                                         |
            out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (178) is hoisted out of the
parallel loop labelled #0 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (179) is hoisted out of the
parallel loop labelled #0 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/content/mod3-mattmaitland1/minitorch/fast_ops.py (215)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-mattmaitland1/minitorch/fast_ops.py (215)
---------------------------------------------------------------------|loop #ID
    def _zip(                                                        |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        b_storage: Storage,                                          |
        b_shape: Shape,                                              |
        b_strides: Strides,                                          |
    ) -> None:                                                       |
        # Fast path for aligned tensors                              |
        if (                                                         |
            len(out_strides) == len(a_strides) == len(b_strides)     |
            and np.array_equal(out_strides, a_strides)               |
            and np.array_equal(out_strides, b_strides)               |
            and np.array_equal(out_shape, a_shape)                   |
            and np.array_equal(out_shape, b_shape)                   |
        ):                                                           |
            for i in prange(len(out)):-------------------------------| #1
                out[i] = fn(a_storage[i], b_storage[i])              |
            return                                                   |
                                                                     |
        # Calculate total elements                                   |
        total_size = 1                                               |
        for dim in range(len(out_shape)):                            |
            total_size *= out_shape[dim]                             |
                                                                     |
        # Main parallel loop                                         |
        for i in prange(total_size):---------------------------------| #2
            # Create thread-local index buffers                      |
            out_idx = np.empty(len(out_shape), np.int32)             |
            a_idx = np.empty(len(out_shape), np.int32)               |
            b_idx = np.empty(len(out_shape), np.int32)               |
                                                                     |
            # Convert flat index to coordinates                      |
            to_index(i, out_shape, out_idx)                          |
                                                                     |
            # Get output position                                    |
            out_pos = index_to_position(out_idx, out_strides)        |
                                                                     |
            # Handle broadcasting and get input positions            |
            broadcast_index(out_idx, out_shape, a_shape, a_idx)      |
            a_pos = index_to_position(a_idx, a_strides)              |
                                                                     |
            broadcast_index(out_idx, out_shape, b_shape, b_idx)      |
            b_pos = index_to_position(b_idx, b_strides)              |
                                                                     |
            # Apply function                                         |
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #1, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (246) is hoisted out of the
parallel loop labelled #2 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_idx = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (247) is hoisted out of the
parallel loop labelled #2 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: a_idx = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (248) is hoisted out of the
parallel loop labelled #2 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: b_idx = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/content/mod3-mattmaitland1/minitorch/fast_ops.py (290)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-mattmaitland1/minitorch/fast_ops.py (290)
------------------------------------------------------------------|loop #ID
    def _reduce(                                                  |
        out: Storage,                                             |
        out_shape: Shape,                                         |
        out_strides: Strides,                                     |
        a_storage: Storage,                                       |
        a_shape: Shape,                                           |
        a_strides: Strides,                                       |
        reduce_dim: int,                                          |
    ) -> None:                                                    |
        # Get total elements                                      |
        ndim = len(out_shape)                                     |
        total = 1                                                 |
        for i in range(ndim):                                     |
            total *= out_shape[i]                                 |
                                                                  |
        # Process each position                                   |
        for i in prange(total):-----------------------------------| #3
            # Thread buffers                                      |
            out_idx = np.empty(ndim, np.int32)                    |
            a_idx = np.empty(ndim, np.int32)                      |
                                                                  |
            # Map to output                                       |
            to_index(i, out_shape, out_idx)                       |
            out_pos = index_to_position(out_idx, out_strides)     |
                                                                  |
            # Setup input indices                                 |
            for j in range(ndim):                                 |
                a_idx[j] = out_idx[j]                             |
                                                                  |
            # First value                                         |
            a_idx[reduce_dim] = 0                                 |
            curr_pos = index_to_position(a_idx, a_strides)        |
            acc = a_storage[curr_pos]                             |
                                                                  |
            # Reduce remaining                                    |
            for j in range(1, a_shape[reduce_dim]):               |
                a_idx[reduce_dim] = j                             |
                curr_pos = index_to_position(a_idx, a_strides)    |
                acc = fn(acc, a_storage[curr_pos])                |
                                                                  |
            out[out_pos] = acc                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (308) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: out_idx = np.empty(ndim, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/content/mod3-mattmaitland1/minitorch/fast_ops.py (309) is hoisted out of the
parallel loop labelled #3 (it will be performed before the loop is executed and
reused inside the loop):
   Allocation:: a_idx = np.empty(ndim, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/content/mod3-mattmaitland1/minitorch/fast_ops.py (335)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-mattmaitland1/minitorch/fast_ops.py (335)
----------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                      |
    out: Storage,                                                                 |
    out_shape: Shape,                                                             |
    out_strides: Strides,                                                         |
    a_storage: Storage,                                                           |
    a_shape: Shape,                                                               |
    a_strides: Strides,                                                           |
    b_storage: Storage,                                                           |
    b_shape: Shape,                                                               |
    b_strides: Strides,                                                           |
) -> None:                                                                        |
    """NUMBA tensor matrix multiply function.                                     |
                                                                                  |
    Should work for any tensor shapes that broadcast as long as                   |
                                                                                  |
    ```                                                                           |
    assert a_shape[-1] == b_shape[-2]                                             |
    ```                                                                           |
                                                                                  |
    Optimizations:                                                                |
                                                                                  |
    * Outer loop in parallel                                                      |
    * No index buffers or function calls                                          |
    * Inner loop should have no global writes, 1 multiply.                        |
                                                                                  |
                                                                                  |
    Args:                                                                         |
    ----                                                                          |
        out (Storage): storage for `out` tensor                                   |
        out_shape (Shape): shape for `out` tensor                                 |
        out_strides (Strides): strides for `out` tensor                           |
        a_storage (Storage): storage for `a` tensor                               |
        a_shape (Shape): shape for `a` tensor                                     |
        a_strides (Strides): strides for `a` tensor                               |
        b_storage (Storage): storage for `b` tensor                               |
        b_shape (Shape): shape for `b` tensor                                     |
        b_strides (Strides): strides for `b` tensor                               |
                                                                                  |
    Returns:                                                                      |
    -------                                                                       |
        None : Fills in `out`                                                     |
                                                                                  |
    """                                                                           |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                        |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                        |
                                                                                  |
    # TODO: Implement for Task 3.2.                                               |
    batch_size = max(a_shape[0], b_shape[0])                                      |
    rows, cols = a_shape[1], b_shape[2]                                           |
    reduce_dim = a_shape[2]                                                       |
                                                                                  |
    for p in prange(batch_size * rows):-------------------------------------------| #4
        batch = p // rows                                                         |
        row = p % rows                                                            |
                                                                                  |
        a_start = batch * a_batch_stride + row * a_strides[1]                     |
        b_start = batch * b_batch_stride                                          |
        out_pos = batch * out_strides[0] + row * out_strides[1]                   |
                                                                                  |
        for j in range(cols):                                                     |
            temp = 0.0                                                            |
            for k in range(reduce_dim):                                           |
                temp += (                                                         |
                    a_storage[a_start + k * a_strides[2]]                         |
                    * b_storage[b_start + k * b_strides[1] + j * b_strides[2]]    |
                )                                                                 |
            out[out_pos + j * out_strides[2]] = temp                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


<h2>Proof of speed-ups on large matrix operations:</h2>
Output from timing.py provided on Ed

```
Timing summary
Size: 64
    fast: 0.00319
    gpu: 0.00689
Size: 128
    fast: 0.01185
    gpu: 0.01476
Size: 256
    fast: 0.05736
    gpu: 0.05232
Size: 512
    fast: 0.36776
    gpu: 0.21865
Size: 1024
    fast: 2.16130
    gpu: 0.97939
```




<h2>Simple</h2>
<h3>CPU</h3>

```
Epoch  0  loss  4.578356361272406 correct 35
Epoch  10  loss  1.7062716711960901 correct 49
Epoch  20  loss  0.7548648765179116 correct 50
Epoch  30  loss  1.165145762695747 correct 50
Epoch  40  loss  0.31989225248948977 correct 50
Epoch  50  loss  0.4923159757054414 correct 49
Epoch  60  loss  0.1846076928537417 correct 50
Epoch  70  loss  0.5249055893221868 correct 50
Epoch  80  loss  0.4989057046060035 correct 50
Epoch  90  loss  1.2035592411641383 correct 50
Epoch  100  loss  0.4852882360093281 correct 50
Epoch  110  loss  0.20736721785782752 correct 50
Epoch  120  loss  0.11712409404791907 correct 50
Epoch  130  loss  0.4916660337015577 correct 50
Epoch  140  loss  0.4476628114081715 correct 50
Epoch  150  loss  0.5761846933961331 correct 50
Epoch  160  loss  0.33567191062379453 correct 50
Epoch  170  loss  0.02939205663709698 correct 50
Epoch  180  loss  0.13302064011795164 correct 50
Epoch  190  loss  0.27191794833776817 correct 50
Epoch  200  loss  0.05360278132742095 correct 50
Epoch  210  loss  0.3084510114870961 correct 50
Epoch  220  loss  0.31221589448262943 correct 50
Epoch  230  loss  0.022666373358692957 correct 50
Epoch  240  loss  0.6453295268957658 correct 50
Epoch  250  loss  0.13381890854641026 correct 50
Epoch  260  loss  0.04955189432510262 correct 50
Epoch  270  loss  0.0017283421708740654 correct 50
Epoch  280  loss  0.15830590343381148 correct 50
Epoch  290  loss  0.051692884242693055 correct 50
Epoch  300  loss  0.00921760567102152 correct 50
Epoch  310  loss  0.2720878825264728 correct 50
Epoch  320  loss  0.1420993549799111 correct 50
Epoch  330  loss  0.12595560319970167 correct 50
Epoch  340  loss  0.00048541452135227525 correct 50
Epoch  350  loss  0.11299686624956252 correct 50
Epoch  360  loss  0.26396835089781256 correct 50
Epoch  370  loss  0.1437720169930806 correct 50
Epoch  380  loss  0.10337068506977445 correct 50
Epoch  390  loss  0.16458938797279377 correct 50
Epoch  400  loss  0.026783442730087162 correct 50
Epoch  410  loss  0.13634214317044782 correct 50
Epoch  420  loss  0.10390307700399071 correct 50
Epoch  430  loss  0.07297417086620361 correct 50
Epoch  440  loss  0.024265855708064762 correct 50
Epoch  450  loss  0.12220076986657841 correct 50
Epoch  460  loss  0.012558141185876321 correct 50
Epoch  470  loss  0.015760327718579614 correct 50
Epoch  480  loss  0.019712020738753087 correct 50
Epoch  490  loss  0.01240613924867935 correct 50


Average execution time: 0.17 seconds
```

<h3>GPU</h3>

```
Epoch  0  loss  4.328733722271717 correct 46
Epoch  10  loss  1.54015448042516 correct 49
Epoch  20  loss  1.0230725840084884 correct 50
Epoch  30  loss  0.31919869434197523 correct 50
Epoch  40  loss  1.0577159259480904 correct 50
Epoch  50  loss  0.41252422667307864 correct 50
Epoch  70  loss  0.15117826453846495 correct 50
Epoch  80  loss  0.16480411825505598 correct 50
Epoch  90  loss  0.10700053325744553 correct 50
Epoch  100  loss  0.21593501341556914 correct 50
Epoch  110  loss  0.2591596419739543 correct 50
Epoch  120  loss  0.0964420109687326 correct 50
Epoch  130  loss  0.08477008739903363 correct 50
Epoch  140  loss  0.15804292589347407 correct 50
Epoch  150  loss  0.17897755517284936 correct 50
Epoch  160  loss  0.1974796455134642 correct 50
Epoch  170  loss  0.043724540249585934 correct 50
Epoch  180  loss  0.162722596238679 correct 50
Epoch  190  loss  0.04507685817821262 correct 50
Epoch  200  loss  0.14813674738874072 correct 50
Epoch  210  loss  0.01747738190148397 correct 50
Epoch  220  loss  0.009485262294388543 correct 50
Epoch  230  loss  0.07947713333765838 correct 50
Epoch  240  loss  0.1261814157494011 correct 50
Epoch  250  loss  0.019303777891863036 correct 50
Epoch  260  loss  0.03592203020088125 correct 50
Epoch  270  loss  0.008271791294604681 correct 50
Epoch  280  loss  0.050775302101082956 correct 50
Epoch  290  loss  0.08013428890265524 correct 50
Epoch  300  loss  0.06734876677994998 correct 50
Epoch  310  loss  0.025640662523513675 correct 50
Epoch  320  loss  0.009005584463185572 correct 50
Epoch  330  loss  0.04536574747310296 correct 50
Epoch  340  loss  0.11989379872250858 correct 50
Epoch  350  loss  0.0360546982449031 correct 50
Epoch  360  loss  0.08406089162158599 correct 50
Epoch  370  loss  0.0487740432655334 correct 50
Epoch  380  loss  0.06913829575344856 correct 50
Epoch  390  loss  0.0010390841226030347 correct 50
Epoch  400  loss  0.05887034884186106 correct 50
Epoch  410  loss  0.014219223242209564 correct 50
Epoch  420  loss  0.029471231677801968 correct 50
Epoch  430  loss  0.0012733250145469145 correct 50
Epoch  440  loss  0.0066095304068968744 correct 50
Epoch  450  loss  0.0782706642715219 correct 50
Epoch  460  loss  0.043989155793528086 correct 50
Epoch  470  loss  0.0002522698540611754 correct 50
Epoch  480  loss  0.05526628888541616 correct 50
Epoch  490  loss  0.00048704932993141187 correct 50

Average execution time: 1.81 seconds
```


<h2>Xor</h2>
<h3>CPU</h3>

```
Epoch  0  loss  6.18063664967868 correct 25
Epoch  10  loss  4.547030460359165 correct 47
Epoch  20  loss  3.9671677971462134 correct 43
Epoch  30  loss  3.1415450697322007 correct 43
Epoch  40  loss  1.3000545543689843 correct 47
Epoch  50  loss  2.6244321582195393 correct 48
Epoch  60  loss  1.4647208386751516 correct 48
Epoch  70  loss  1.296517653124287 correct 48
Epoch  80  loss  1.760230392943282 correct 48
Epoch  90  loss  2.599621931504696 correct 50
Epoch  100  loss  0.7187708048567257 correct 49
Epoch  110  loss  1.0654516553960895 correct 50
Epoch  120  loss  0.9347168704565577 correct 50
Epoch  130  loss  0.777445162141619 correct 50
Epoch  140  loss  0.29216965345877094 correct 50
Epoch  150  loss  0.9736698967949945 correct 50
Epoch  160  loss  0.3711007317670324 correct 50
Epoch  170  loss  0.9270477381803555 correct 50
Epoch  180  loss  0.6029597150945812 correct 50
Epoch  190  loss  0.9516882700507074 correct 50
Epoch  200  loss  0.4981775485498159 correct 50
Epoch  210  loss  0.7827678095613726 correct 50
Epoch  220  loss  0.3853122137526555 correct 50
Epoch  230  loss  1.1306826534872203 correct 50
Epoch  240  loss  0.2734386363382678 correct 50
Epoch  250  loss  0.3236934274123386 correct 50
Epoch  260  loss  0.19381908891163707 correct 50
Epoch  270  loss  0.6515308805022457 correct 50
Epoch  280  loss  0.4816980013666892 correct 50
Epoch  290  loss  0.7636038417573414 correct 50
Epoch  300  loss  0.5500105891015064 correct 50
Epoch  310  loss  0.41254405152835416 correct 50
Epoch  320  loss  0.5982384931678482 correct 50
Epoch  330  loss  0.5078656367426685 correct 50
Epoch  340  loss  0.5573788793643973 correct 50
Epoch  350  loss  0.553613386416495 correct 50
Epoch  360  loss  0.28030263017762824 correct 50
Epoch  370  loss  0.2198867376267029 correct 50
Epoch  380  loss  0.209006520295603 correct 50
Epoch  390  loss  0.2851365919684275 correct 50
Epoch  400  loss  0.13791201056149202 correct 50
Epoch  410  loss  0.5066067536136559 correct 50
Epoch  420  loss  0.08047682972118148 correct 50
Epoch  430  loss  0.03613442263586975 correct 50
Epoch  440  loss  0.19607686795667123 correct 50
Epoch  450  loss  0.1473272373454733 correct 50
Epoch  460  loss  0.34009872742172675 correct 50
Epoch  470  loss  0.3639386666574981 correct 50
Epoch  480  loss  0.05554261798865613 correct 50
Epoch  490  loss  0.11064292577320922 correct 50

Average execution time: 0.17 seconds

```


<h3>GPU</h3>

```
Epoch  0  loss  4.423298595389291 correct 30
Epoch  10  loss  3.7026982552618173 correct 47
Epoch  20  loss  2.394555588912163 correct 48
Epoch  30  loss  3.129464833103821 correct 48
Epoch  40  loss  2.4125282134477866 correct 48
Epoch  50  loss  0.9029081480584527 correct 48
Epoch  60  loss  2.495927042902499 correct 48
Epoch  70  loss  1.2398590010685329 correct 48
Epoch  80  loss  0.3729361325781324 correct 49
Epoch  90  loss  1.1960266614654893 correct 50
Epoch  100  loss  1.0527856526197068 correct 50
Epoch  110  loss  1.1807305766314942 correct 50
Epoch  120  loss  0.8715872226919001 correct 50
Epoch  130  loss  1.82408955717997 correct 50
Epoch  140  loss  1.6341946725007284 correct 49
Epoch  150  loss  0.7694535265995193 correct 50
Epoch  160  loss  0.49428449718000333 correct 50
Epoch  170  loss  0.8053897129016221 correct 50
Epoch  180  loss  0.21733888576679092 correct 50
Epoch  190  loss  0.03082369749257224 correct 50
Epoch  200  loss  0.3116142789999846 correct 50
Epoch  210  loss  0.4055085641142201 correct 50
Epoch  220  loss  0.7932202475943454 correct 50
Epoch  230  loss  0.7378729820018748 correct 50
Epoch  240  loss  0.0458942726635197 correct 50
Epoch  250  loss  0.29218371786522446 correct 50
Epoch  260  loss  0.6295335107076002 correct 50
Epoch  270  loss  0.37980334593467 correct 50
Epoch  280  loss  0.1763769030378672 correct 50
Epoch  290  loss  0.6344166642312044 correct 50
Epoch  300  loss  0.5386118734225022 correct 50
Epoch  310  loss  0.21224054168265843 correct 50
Epoch  320  loss  0.558334652306085 correct 50
Epoch  330  loss  0.3335270900866154 correct 50
Epoch  340  loss  0.41569578547340447 correct 50
Epoch  350  loss  0.26308763070952923 correct 50
Epoch  360  loss  0.43858824201683533 correct 50
Epoch  370  loss  0.42637660440985004 correct 50
Epoch  380  loss  0.2603284914561728 correct 50
Epoch  390  loss  0.20703544399868615 correct 50
Epoch  400  loss  0.20418865090505176 correct 50
Epoch  410  loss  0.4677443048649338 correct 50
Epoch  420  loss  0.11276558818945655 correct 50
Epoch  430  loss  0.42230954654076636 correct 50
Epoch  440  loss  0.17861013241581933 correct 50
Epoch  450  loss  0.09822847166498297 correct 50
Epoch  460  loss  0.08567013442785967 correct 50
Epoch  470  loss  0.0604707499714868 correct 50
Epoch  480  loss  0.3236906455773331 correct 50
Epoch  490  loss  0.38983390144034447 correct 50

Average execution time: 1.79 seconds
```



<h2>Split</h2>
<h3>CPU</h3>

```
Epoch  0  loss  6.343115155818081 correct 36
Epoch  10  loss  3.5641394232383394 correct 40
Epoch  20  loss  3.1471355545411117 correct 43
Epoch  30  loss  3.371758131887922 correct 43
Epoch  40  loss  2.5774770364465485 correct 46
Epoch  50  loss  4.468114397745539 correct 47
Epoch  60  loss  2.681807211090392 correct 48
Epoch  70  loss  2.02900137907765 correct 48
Epoch  80  loss  1.6115974977059035 correct 47
Epoch  90  loss  2.130827032796261 correct 50
Epoch  100  loss  1.2764132081960269 correct 49
Epoch  110  loss  1.4546418319380288 correct 50
Epoch  120  loss  2.2506333552848585 correct 50
Epoch  130  loss  1.6169093915026052 correct 50
Epoch  140  loss  0.6165791479713398 correct 50
Epoch  150  loss  0.7664307481858387 correct 50
Epoch  160  loss  1.1705831373042295 correct 50
Epoch  170  loss  1.2268826996763693 correct 50
Epoch  180  loss  0.44309251830151003 correct 50
Epoch  190  loss  0.9490930283451576 correct 50
Epoch  200  loss  0.3410404137164442 correct 49
Epoch  210  loss  1.507876444323958 correct 50
Epoch  220  loss  1.0426014913814772 correct 50
Epoch  230  loss  0.16336106592030783 correct 49
Epoch  240  loss  0.635308855685925 correct 50
Epoch  250  loss  0.4205247846716026 correct 50
Epoch  260  loss  0.9348565127499444 correct 50
Epoch  270  loss  0.4472352979473496 correct 50
Epoch  280  loss  1.0284128555879513 correct 50
Epoch  290  loss  0.40000222660058726 correct 50
Epoch  300  loss  1.4240345881279446 correct 50
Epoch  310  loss  0.13611593772681613 correct 50
Epoch  320  loss  0.5054051661891038 correct 50
Epoch  330  loss  0.3123539357314727 correct 50
Epoch  340  loss  0.6897448536852182 correct 50
Epoch  350  loss  0.10648446738716469 correct 50
Epoch  360  loss  0.6500250569625683 correct 50
Epoch  370  loss  0.6195469275625468 correct 50
Epoch  380  loss  0.26063015809441836 correct 50
Epoch  390  loss  0.4166262353228125 correct 50
Epoch  400  loss  0.2769171025258265 correct 50
Epoch  410  loss  0.31949684526053845 correct 50
Epoch  420  loss  0.12213755237244218 correct 50
Epoch  430  loss  0.5112404166783913 correct 50
Epoch  440  loss  0.10384795644860731 correct 50
Epoch  450  loss  0.18904292657807717 correct 50
Epoch  460  loss  0.39728238968150315 correct 50
Epoch  470  loss  0.6342080524963118 correct 50
Epoch  480  loss  0.2042712324348394 correct 50
Epoch  490  loss  0.018872636000514857 correct 50

Average execution time: 0.17 seconds
```



<h3>GPU</h3>

```
Epoch  0  loss  5.620474598537574 correct 26
Epoch  10  loss  4.07744839994591 correct 45
Epoch  20  loss  3.1850190046324984 correct 47
Epoch  30  loss  4.895892625628052 correct 47
Epoch  40  loss  1.4256250552512904 correct 49
Epoch  50  loss  2.623032498753559 correct 49
Epoch  60  loss  1.351391529139111 correct 49
Epoch  70  loss  1.2758388542882164 correct 48
Epoch  80  loss  1.1129227084651305 correct 49
Epoch  90  loss  0.7458437666772983 correct 49
Epoch  100  loss  0.8621987120070249 correct 49
Epoch  110  loss  1.2571785925773085 correct 49
Epoch  120  loss  0.5572562765433674 correct 49
Epoch  130  loss  1.9089885441017904 correct 49
Epoch  140  loss  0.38517652029047245 correct 49
Epoch  150  loss  0.6062481571298529 correct 49
Epoch  160  loss  0.5246197358109332 correct 49
Epoch  170  loss  0.5610385296995858 correct 49
Epoch  180  loss  1.9682696477576016 correct 50
Epoch  190  loss  0.1411118983801618 correct 49
Epoch  200  loss  0.449273966006736 correct 49
Epoch  210  loss  0.4935652538941351 correct 49
Epoch  220  loss  0.0639871161471596 correct 50
Epoch  230  loss  0.32016531414601157 correct 49
Epoch  240  loss  1.593258105360777 correct 50
Epoch  250  loss  0.15594425604099085 correct 49
Epoch  260  loss  1.6447793074444437 correct 50
Epoch  270  loss  0.4737926972160773 correct 49
Epoch  280  loss  0.047018544512116914 correct 50
Epoch  290  loss  0.43696379575179894 correct 50
Epoch  300  loss  0.35534638323132567 correct 49
Epoch  310  loss  0.22766363434995954 correct 50
Epoch  320  loss  0.22422554644863418 correct 49
Epoch  330  loss  0.08723068909315565 correct 50
Epoch  340  loss  0.6559710378437691 correct 49
Epoch  350  loss  0.2294671384648165 correct 50
Epoch  360  loss  0.23744415994169793 correct 50
Epoch  370  loss  0.17056025372431488 correct 49
Epoch  380  loss  0.18436318464921175 correct 50
Epoch  390  loss  1.088813580695094 correct 50
Epoch  400  loss  1.178528412880976 correct 50
Epoch  410  loss  0.24571933522350145 correct 49
Epoch  420  loss  0.8805817864213917 correct 49
Epoch  430  loss  0.0331155015320554 correct 50
Epoch  440  loss  1.0941639335651077 correct 50
Epoch  450  loss  0.3975916861790319 correct 50
Epoch  460  loss  0.395166870469416 correct 50
Epoch  470  loss  1.1081242606314146 correct 50
Epoch  480  loss  0.9102436183957963 correct 49
Epoch  490  loss  0.11339411022482951 correct 50

Average execution time: 1.78 seconds
```

<h2>Bigger Dataset (Xor with 200 layers)</h2>
<h3>CPU</h3>

```
Epoch  0  loss  13.157290284522018 correct 31
Epoch  10  loss  2.367584929948194 correct 45
Epoch  20  loss  2.551091082129693 correct 40
Epoch  30  loss  3.6514952823863718 correct 47
Epoch  40  loss  4.936957955178266 correct 47
Epoch  50  loss  4.413718695136273 correct 43
Epoch  60  loss  3.256881752098558 correct 44
Epoch  70  loss  2.5319856535920304 correct 47
Epoch  80  loss  2.4372158424705788 correct 46
Epoch  90  loss  1.7461981346891375 correct 42
Epoch  100  loss  2.275246488138517 correct 46
Epoch  110  loss  1.6839479701114806 correct 48
Epoch  120  loss  3.2109770596075338 correct 47
Epoch  130  loss  2.5421472819676625 correct 47
Epoch  140  loss  0.9858811871851281 correct 50
Epoch  150  loss  2.002906611567263 correct 50
Epoch  160  loss  1.2451025533794362 correct 49
Epoch  170  loss  1.0907364417090304 correct 50
Epoch  180  loss  1.8139457563175512 correct 50
Epoch  190  loss  0.39485275078201476 correct 50
Epoch  200  loss  1.3342582572564503 correct 50
Epoch  210  loss  1.660521448976852 correct 50
Epoch  220  loss  1.6440568925625385 correct 50
Epoch  230  loss  0.803738625375574 correct 50
Epoch  240  loss  0.1660701636227033 correct 50
Epoch  250  loss  0.1320316240985911 correct 48
Epoch  260  loss  1.5604724382725184 correct 49
Epoch  270  loss  2.333285998759588 correct 45
Epoch  280  loss  0.07098899432800587 correct 50
Epoch  290  loss  0.9440985030024802 correct 50
Epoch  300  loss  0.8823030118339977 correct 50
Epoch  310  loss  0.24221239978939665 correct 50
Epoch  320  loss  0.19075922954254526 correct 50
Epoch  330  loss  0.798656158600011 correct 50
Epoch  340  loss  0.06711049154800762 correct 48
Epoch  350  loss  0.7148981792243141 correct 50
Epoch  360  loss  0.7529949291213514 correct 50
Epoch  370  loss  0.34656032650806845 correct 50
Epoch  380  loss  0.3605878713479615 correct 50
Epoch  390  loss  0.14903953230650316 correct 50
Epoch  400  loss  0.14723279587607443 correct 50
Epoch  410  loss  0.6702896213579385 correct 50
Epoch  420  loss  1.9199896602005122 correct 45
Epoch  430  loss  0.5691632585864909 correct 50
Epoch  440  loss  0.37528743735170256 correct 50
Epoch  450  loss  2.9989254720267247 correct 47
Epoch  460  loss  0.24042254451029388 correct 50
Epoch  470  loss  0.7451078644987581 correct 50
Epoch  480  loss  0.055990854232039255 correct 49
Epoch  490  loss  0.7358433698236309 correct 50

Average execution time: 0.26 seconds
```

<h3>GPU</h3>

```
Epoch  0  loss  12.950910745207487 correct 37
Epoch  10  loss  2.3653773349694696 correct 47
Epoch  20  loss  1.8479193020340656 correct 48
Epoch  30  loss  0.8618432533489633 correct 45
Epoch  40  loss  1.7602737703960272 correct 47
Epoch  50  loss  0.321505395169406 correct 47
Epoch  60  loss  0.9289418172771458 correct 48
Epoch  70  loss  2.4914939054145595 correct 49
Epoch  80  loss  0.9678406173619124 correct 49
Epoch  90  loss  0.9656003337379953 correct 49
Epoch  100  loss  1.3245851317987434 correct 48
Epoch  110  loss  0.4631159661099411 correct 48
Epoch  120  loss  1.9953127964062622 correct 47
Epoch  130  loss  1.4683870252857756 correct 48
Epoch  140  loss  2.1158907412598897 correct 47
Epoch  150  loss  1.7286515569470402 correct 50
Epoch  160  loss  1.5035456320034262 correct 50
Epoch  170  loss  0.4233325357855705 correct 48
Epoch  180  loss  1.4135752202818064 correct 48
Epoch  190  loss  2.206067144056669 correct 50
Epoch  200  loss  0.2689812577096144 correct 49
Epoch  210  loss  0.27478178364777656 correct 48
Epoch  220  loss  1.622837645517204 correct 49
Epoch  230  loss  0.6878719254943914 correct 48
Epoch  240  loss  0.12630421120092322 correct 50
Epoch  250  loss  0.7086107241861255 correct 49
Epoch  260  loss  1.2625685368683097 correct 48
Epoch  270  loss  0.27627105660411516 correct 50
Epoch  280  loss  1.0054697463923445 correct 49
Epoch  290  loss  0.3050725158026184 correct 49
Epoch  300  loss  1.0357194271094254 correct 49
Epoch  310  loss  0.32284038557370476 correct 48
Epoch  320  loss  0.4299182255552706 correct 50
Epoch  330  loss  1.133302270014155 correct 50
Epoch  340  loss  0.20445621281386897 correct 50
Epoch  350  loss  0.9113119964436621 correct 50
Epoch  360  loss  0.5486080376788868 correct 50
Epoch  370  loss  0.17598628380628678 correct 50
Epoch  380  loss  2.0467754875150224 correct 47
Epoch  390  loss  0.5093239470240971 correct 50
Epoch  400  loss  0.40297256034915274 correct 50
Epoch  410  loss  0.5396050569934132 correct 50
Epoch  420  loss  0.3199376363030484 correct 49
Epoch  430  loss  0.5843966597729402 correct 50
Epoch  440  loss  0.1278086016931392 correct 50
Epoch  450  loss  0.6475596561783965 correct 50
Epoch  460  loss  0.2727294046750628 correct 50
Epoch  470  loss  0.6200723913125641 correct 50
Epoch  480  loss  0.09893252679864915 correct 50
Epoch  490  loss  0.052586164916684545 correct 50

Average execution time: 1.87 seconds
```

