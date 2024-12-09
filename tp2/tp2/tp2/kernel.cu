#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OpenGL Graphics includes
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Ray tracer"

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU2_MODE 3
#define GPU3_MODE 4

GLuint imageTex;
GLuint imageBuffer;
float* debug;

#define INF 2e10f
struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;
    __host__ __device__ float hit(float cx, float cy, float* sh) {
        float dx = cx - x;
        float dy = cy - y;
        float dz2 = radius * radius - dx * dx - dy * dy;
        if (dz2 > 0) {
            float dz = sqrtf(dz2);
            *sh = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};

Sphere* spheres = NULL; 
Sphere* d_spheres = NULL;

// Constant memory
__constant__ Sphere d_spheres_const[1024];  // Max of 1024 spheres
int numSpheres = 5; // Current number of spheres

float cameraX = 0.0f, cameraY = 0.0f; // Camera position in the xy-plane
float ambient = 0.2f; // Ambient light intensity

/* Globals */
float scale = 0.003f;
float mx, my;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4* pixels;
float4* d_pixels;

// Variables for mouse dragging
bool isDragging = false;
int lastMouseX = 0, lastMouseY = 0;

void clean();
void init();

// Function to initialize spheres with random positions and colors
void initSpheres() {
    if (spheres != NULL) {
        free(spheres); // Free previous allocation if it exists
    }

    spheres = (Sphere*)malloc(numSpheres * sizeof(Sphere)); // Allocate memory for new number of spheres

    for (int i = 0; i < numSpheres; i++) {
        spheres[i].r = (float)rand() / RAND_MAX;
        spheres[i].g = (float)rand() / RAND_MAX;
        spheres[i].b = (float)rand() / RAND_MAX;
        spheres[i].radius = 0.01f + (float)rand() / RAND_MAX * 0.03f;
        spheres[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        spheres[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        spheres[i].z = -1.0f - (float)rand() / RAND_MAX * 2.0f;
    }
}

void initCPU() {
    pixels = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4));  // Allocate pixel buffer
    initSpheres();
}

void cleanCPU() {
    if (pixels != NULL) {
        free(pixels);
        pixels = NULL;
    }
    if (spheres != NULL) {
        free(spheres);
        spheres = NULL;
    }
}

void initGPU() {
    initSpheres();

    cudaMalloc((void**)&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4));
    cudaMallocHost((void**)&pixels, SCREEN_X * SCREEN_Y * sizeof(float4));

    cudaMalloc((void**)&d_spheres, numSpheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
}

void cleanGPU() {
    if (d_pixels != NULL) {
        cudaFree(d_pixels);
        d_pixels = NULL;
    }
    if (d_spheres != NULL) {
        cudaFree(d_spheres);
        d_spheres = NULL;
    }
    if (pixels != NULL) {
        cudaFreeHost(pixels);
        pixels = NULL;
    }
    if (spheres != NULL) {
        free(spheres);
        spheres = NULL;
    }
}

void initGPU2() {
    initSpheres();

    // Allocate and initialize pixel buffer for GPU2 mode
    cudaMalloc((void**)&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4));
    cudaMallocHost((void**)&pixels, SCREEN_X * SCREEN_Y * sizeof(float4));

    if (numSpheres <= 1024) {
        cudaMemcpyToSymbol(d_spheres_const, spheres, numSpheres * sizeof(Sphere), 0, cudaMemcpyHostToDevice);
    }
    else {
        fprintf(stderr, "Number of spheres exceeds the maximum allowed in constant memory!\n");
        numSpheres = 1024;
        clean();
        init();
    }
}

void cleanGPU2() {
    if (d_pixels != NULL) {
        cudaFree(d_pixels);
        d_pixels = NULL;
    }
    if (pixels != NULL) {
        cudaFreeHost(pixels);
        pixels = NULL;
    }
    if (spheres != NULL) {
        free(spheres);
        spheres = NULL;
    }
}

void initGPU3() {
    initSpheres();

    // Allocate and initialize pixel buffer for GPU2 mode
    cudaMalloc((void**)&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4));
    cudaMallocHost((void**)&pixels, SCREEN_X * SCREEN_Y * sizeof(float4));

    if (numSpheres <= 1024) {
        cudaMemcpyToSymbol(d_spheres_const, spheres, numSpheres * sizeof(Sphere), 0, cudaMemcpyHostToDevice);
    }
    else {
        fprintf(stderr, "Number of spheres exceeds the maximum allowed in constant memory!\n");
        numSpheres = 1024;
        clean();
        init();
    }
}

void cleanGPU3() {
    if (d_pixels != NULL) {
        cudaFree(d_pixels);
        d_pixels = NULL;
    }
    if (pixels != NULL) {
        cudaFreeHost(pixels);
        pixels = NULL;
    }
    if (spheres != NULL) {
        free(spheres);
        spheres = NULL;
    }
}

void clean() {
    if (mode == CPU_MODE) {
        cleanCPU();
    }
    else if (mode == GPU2_MODE) {
        cleanGPU2();
    }
    else if (mode == GPU3_MODE) {
        cleanGPU3();
    }
    else {
        cleanGPU();
    }
}

void init() {
    if (mode == CPU_MODE) {
        initCPU();
    }
    else if (mode == GPU2_MODE) {
        initGPU2();
    }
    else if (mode == GPU3_MODE) {
        initGPU3();
    }
    else {
        initGPU();
    }
}

__global__ void processImageSlice(int startRow, int endRow, int numSpheres, float4* pixels, float cameraX, float cameraY, float ambient)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y + startRow;

    if (i >= startRow && i < endRow && j < SCREEN_X)
    {
        // Calculate ray origin (camera ray) based on pixel position and camera offset
        float cx = (float)j / SCREEN_X * 2.0f - 1.0f + cameraX; 
        float cy = (float)i / SCREEN_Y * 2.0f - 1.0f + cameraY; 

        float closestZ = -INF;
        float shading = 0.0f;
        float r = 0, g = 0, b = 0;

        // Iterate through all spheres and check for ray-sphere intersections
        for (int k = 0; k < numSpheres; k++) {
            float hitShading;
            float hitZ = d_spheres_const[k].hit(cx, cy, &hitShading); // Hit test

            if (hitZ > closestZ) {
                closestZ = hitZ;
                shading = hitShading;

                r = ambient + (1.0f - ambient) * d_spheres_const[k].r * shading;
                g = ambient + (1.0f - ambient) * d_spheres_const[k].g * shading;
                b = ambient + (1.0f - ambient) * d_spheres_const[k].b * shading;
            }
        }

        int pixelIndex = i * SCREEN_X + j;
        pixels[pixelIndex] = make_float4(r, g, b, 1.0f);  // Store result in RGBA format
    }
}

void processImageWithStreams()
{
    const int numStreams = 4;
    const int sliceHeight = SCREEN_Y / numStreams;
    cudaStream_t streams[numStreams];

    // Allocate memory for streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 blockSize(16, 16); 
    dim3 gridSize((SCREEN_X + blockSize.x - 1) / blockSize.x, (sliceHeight + blockSize.y - 1) / blockSize.y);

    // Launch kernels for each slice using streams
    for (int i = 0; i < numStreams; i++) {
        int startRow = i * sliceHeight;
        int endRow = (i == numStreams - 1) ? SCREEN_Y : (startRow + sliceHeight);

        processImageSlice << <gridSize, blockSize, 0, streams[i] >> > (startRow, endRow, numSpheres, d_pixels, cameraX, cameraY, ambient);
    }

    // Synchronize streams
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy the pixel data back to pinned host memory
    cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numStreams; i++) {

        cudaStreamDestroy(streams[i]);
    }
}


// CPU ray tracing algorithm
void raytraceCPU() {
    for (int i = 0; i < SCREEN_Y; i++) {
        for (int j = 0; j < SCREEN_X; j++) {
            float cx = (float)j / SCREEN_X * 2.0f - 1.0f + cameraX;  
            float cy = (float)i / SCREEN_Y * 2.0f - 1.0f + cameraY; 
            float closestZ = -INF;
            float shading = 0.0f;
            float r = 0, g = 0, b = 0;  

            for (int k = 0; k < numSpheres; k++) {
                float hitShading;
                float hitZ = spheres[k].hit(cx, cy, &hitShading);

                if (hitZ > closestZ) {
                    closestZ = hitZ;
                    shading = hitShading;

                    r = ambient + (1.0f - ambient) * spheres[k].r * shading;
                    g = ambient + (1.0f - ambient) * spheres[k].g * shading;
                    b = ambient + (1.0f - ambient) * spheres[k].b * shading;
                }
            }

            float4* pixel = pixels + (i * SCREEN_X + j);
            pixel->x = r;
            pixel->y = g;
            pixel->z = b;
            pixel->w = 1.0f; 
        }
    }
}

__global__ void raytraceKernel(Sphere* spheres, int numSpheres, float4* pixels, float cameraX, float cameraY, float ambient) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= SCREEN_Y || j >= SCREEN_X) return;

    // Calculate ray origin (camera ray) based on pixel position and camera offset
    float cx = (float)j / SCREEN_X * 2.0f - 1.0f + cameraX;  
    float cy = (float)i / SCREEN_Y * 2.0f - 1.0f + cameraY; 

    float closestZ = -INF;
    float shading = 0.0f;
    float r = 0, g = 0, b = 0; 

    // Iterate through all spheres and check for ray-sphere intersections
    for (int k = 0; k < numSpheres; k++) {
        float hitShading;
        float hitZ = spheres[k].hit(cx, cy, &hitShading); // Hit test

        if (hitZ > closestZ) {
            closestZ = hitZ;
            shading = hitShading;

            r = ambient + (1.0f - ambient) * spheres[k].r * shading;
            g = ambient + (1.0f - ambient) * spheres[k].g * shading;
            b = ambient + (1.0f - ambient) * spheres[k].b * shading;
        }
    }

    int pixelIndex = i * SCREEN_X + j;
    pixels[pixelIndex] = make_float4(r, g, b, 1.0f);  // Store result in RGBA format
}

__global__ void raytraceKernelGPU2(int numSpheres, float4* pixels, float cameraX, float cameraY, float ambient) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= SCREEN_Y || j >= SCREEN_X) return;

    // Calculate ray origin (camera ray) based on pixel position and camera offset
    float cx = (float)j / SCREEN_X * 2.0f - 1.0f + cameraX; 
    float cy = (float)i / SCREEN_Y * 2.0f - 1.0f + cameraY; 

    float closestZ = -INF;
    float shading = 0.0f;
    float r = 0, g = 0, b = 0;  

    // Iterate through all spheres and check for ray-sphere intersections
    for (int k = 0; k < numSpheres; k++) {
        float hitShading;
        float hitZ = d_spheres_const[k].hit(cx, cy, &hitShading); // Hit test

        if (hitZ > closestZ) {
            closestZ = hitZ;
            shading = hitShading;

            r = ambient + (1.0f - ambient) * d_spheres_const[k].r * shading;
            g = ambient + (1.0f - ambient) * d_spheres_const[k].g * shading;
            b = ambient + (1.0f - ambient) * d_spheres_const[k].b * shading;
        }
    }

    int pixelIndex = i * SCREEN_X + j;
    pixels[pixelIndex] = make_float4(r, g, b, 1.0f);  // Store result in RGBA format
}

void raytraceGPU2() {
    // Load spheres to constant memory if the number of spheres is within limit
    if (numSpheres <= 1024) {
        cudaMemcpyToSymbol(d_spheres_const, spheres, numSpheres * sizeof(Sphere));
    }
    else {
        printf("Number of spheres exceeds the limit for constant memory.\n");
        numSpheres = 1024;
        clean();
        init();
        return;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((SCREEN_X + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (SCREEN_Y + threadsPerBlock.y - 1) / threadsPerBlock.y); 

    raytraceKernelGPU2 << <numBlocks, threadsPerBlock >> > (numSpheres, d_pixels, cameraX, cameraY, ambient);

    // Synchronize GPU to ensure kernel has finished execution
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost);
}

void raytraceGPU() {
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((SCREEN_X + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (SCREEN_Y + threadsPerBlock.y - 1) / threadsPerBlock.y); 

    raytraceKernel << <numBlocks, threadsPerBlock >> > (d_spheres, numSpheres, d_pixels, cameraX, cameraY, ambient);

    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost);
}

void calculate() {
    frame++;
    int timecur = glutGet(GLUT_ELAPSED_TIME);

    if (timecur - timebase > FPS_UPDATE) {
        char t[200];
        char* m = "";
        switch (mode)
        {
        case CPU_MODE: m = "CPU mode"; break;
        case GPU_MODE: m = "GPU mode"; break;
        case GPU2_MODE: m = "GPU2 mode"; break;
        case GPU3_MODE: m = "GPU3 mode"; break;
        }
        sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
        glutSetWindowTitle(t);
        timebase = timecur;
        frame = 0;
    }

    switch (mode)
    {
    case CPU_MODE: raytraceCPU(); break;
    case GPU_MODE: raytraceGPU(); break;
    case GPU2_MODE: raytraceGPU2(); break;
    case GPU3_MODE: processImageWithStreams(); break;
    }
}



void idle() {
    glutPostRedisplay();
}

void render() {
    calculate();
    switch (mode)
    {
    case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
    case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
    case GPU2_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
    case GPU3_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
    }
    glutSwapBuffers();
}

void toggleMode(int m) {
    clean();
    mode = m;
    init();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            isDragging = true;
            lastMouseX = x;
            lastMouseY = y;
        }
        else if (state == GLUT_UP) {
            isDragging = false;
        }
    }
    if (button == 3) scale /= 1.05f;
    else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y) {
    if (isDragging) {
        int dx = x - lastMouseX;
        int dy = y - lastMouseY;

        // Adjust camera position based on mouse movement
        cameraX += dx * 0.01f; // Sensitivity for x movement
        cameraY -= dy * 0.01f; // Sensitivity for y movement (inverted for screen coordinates)

        lastMouseX = x;
        lastMouseY = y;
    }
}

void processNormalKeys(unsigned char key, int x, int y) {
    if (key == 27) { clean(); exit(0); }
    else if (key == '1') toggleMode(CPU_MODE);
    else if (key == '2') toggleMode(GPU_MODE);
    else if (key == '3') toggleMode(GPU2_MODE);
    else if (key == '4') toggleMode(GPU3_MODE);
    else if (key == '+') {
        numSpheres += 10;
        clean();
        init();
    }
    else if (key == '-') {
        if (numSpheres > 10) {
            numSpheres -= 10;
            clean();
            init();
        }
    }
}

void processSpecialKeys(int key, int x, int y) {
    
}

void initGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(SCREEN_X, SCREEN_Y);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(TITLE);
    glViewport(0, 0, SCREEN_X, SCREEN_Y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, SCREEN_X, 0, SCREEN_Y);

    glutIdleFunc(idle);
    glutKeyboardFunc(processNormalKeys);
    glutSpecialFunc(processSpecialKeys);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);  
    glutDisplayFunc(render);

    init();
}

int main(int argc, char** argv) {
    initGL(argc, argv);
    glutMainLoop();
    return 0;
}
