# prog-gpu           10 / 21 pts
 


✔️ TP1 – Julia sets - 3pts
|
✔️ 1) Write a CPU version of the Julia set fractals. The user can change the seed by dragging the  mouse, and increment/decrement the precision by hitting special keys. 
✔️ 2) Write an equivalent GPU version of the Julia set fractals. Use 2D thread and 2D block  indexing. The user can toggle between the CPU mode and the GPU mode. 
✔️ 3) GPU version 2: use pinned host memory for faster data transfer.
|
KEYBIND: NUMPAD_1 NUMPAD_2 NUMPAD_3 (select question 1 2 3)           [+] [-] (changer la précision)



✔️ TP2 – RayTracer - 4pts
|
✔️ 1) Write a CPU version of the ray tracing algorithm. The user can change the number of spheres by hitting special keys and use the mouse to move the camera within the xy-plane.
✔️ 2) GPU version 1: each thread is responsible for one pixel of the output image. Use pinned host memory for fast data transfers.
✔️ 3) GPU version 2: load the sphere array into the constant memory of the GPU.
✔️ 4) GPU version 3: use streams for task parallelization. 4 streams may be a good choice. Each stream processes a separate slice of the image.
|
KEYBIND: NUMPAD_1 NUMPAD_2 NUMPAD_3 NUMPAD_4 (select question 1 2 3 4)              Mouse movement to move along the screen (X and Y)



✔️ TP 3 – Bugs - 3pts
|
✔️ 1) Write a CPU version of the Bug algorithm. The edges should wrap around so that you do not need to worry about borders and corners.
✔️ 2) GPU version 1. Allocate two grids on the device: grid1 and grid2. Copy the initial grid from the host to grid1. Then compute on the device from grid1 to grid2 and fetch grid2 back for the OpenGL rendering call. At the next frame, compute from grid2 back to grid1 and fetch grid1 back for the OpenGL rendering call, and so on.
✔️ 3) GPU version 2: use shared memory for faster memory access. In each thread block, preload a “tile”, i.e. a subgrid of cells, into shared memory and calculate the next generation of that tile (you need to figure out how to resolve the problem of computing the tile borders).
|
KEYBIND: NUMPAD_1 NUMPAD_2 NUMPAD_3 (select question 1 2 3)



❌ TP4 – Nbody - 3pts
|
❌ 1) Write a CPU version of the N-body algorithm. The precise parameter values of the algorithm do not really matter for us. If you are unsure, you can use G = 0.0000001, EPS² = 0.1, m[j] = random(1.0...5.0) which produces nice simulations. 
❌ 2) GPU version 1. Write a kernel which receives body masses, current positions and current velocities, and which computes new pos/vel. Just as in TP3, copy the initial values only once and compute on the device from pos1/vel1 to pos2/vel2, and next time from pos2/vel2 back to pos1/vel1. At each frame, fetch the proper data back to the host. 
❌ 3) GPU version 2: use shared memory for faster memory access. In each thread block, preload the data of NBTHREADS bodies into shared memory and start summing up partial accelerations. Then preload the next NBTHREADS bodies,



❌ TP5 – Kmeans - 3pts
|
❌ 1) Write a CPU version of the Kmeans algorithm. 
❌ 2) GPU version 1: write a kernel kernelAssign where phase 1 (assignment) is executed on the GPU. Copy the resulting pointlabel array back to the host and continue phase 2 (reduction) on the CPU. 
❌ 3) GPU version 2: design a second kernel, kernelReduce where phase 2 (reduction) is executed on the GPU. Execute kernelAssign followed by kernelReduce. Try to be faster than version 1 using CPU-reduction. 



❌ TP6 – Interop - 5pts
|
❌ 1) TP1 Julia
❌ 2) TP2 RayTracer
❌ 3) TP3 Bugs
❌ 4) TP4 Nbody
❌ 5) TP5 Kmeans