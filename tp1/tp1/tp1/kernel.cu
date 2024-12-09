#include <stdio.h>

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
#define TITLE "Julia Fractals"

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU2_MODE 3

GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals */
float scale = 0.003f;
float mx = 0.0f, my = 0.0f; // mouse coordinates for seed
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;
int precision = 100; // Default precision for Julia set
float4* pixels;
float4* d_pixels;

// CUDA kernel for Julia set computation
__global__ void juliaKernel(float4* pixels, float sx, float sy, float scale, int precision) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (j < SCREEN_X && i < SCREEN_Y) {
        float x = scale * (j - SCREEN_X / 2);
        float y = scale * (i - SCREEN_Y / 2);
        float a_real = x;
        float a_imag = y;

        for (int iter = 0; iter < precision; iter++) {
            float a_real_squared = a_real * a_real - a_imag * a_imag; 
            float a_imag_squared = 2 * a_real * a_imag;
            a_real = a_real_squared + sx; 
            a_imag = a_imag_squared + sy; 

            if (a_real * a_real + a_imag * a_imag > 4.0f) {
                float value = 1.0f - (float)iter / precision; 
                pixels[i * SCREEN_X + j] = make_float4(value, value, value, 1.0f); 
                return;
            }
        }
        pixels[i * SCREEN_X + j] = make_float4(0.0f, 0.0f, 0.0f, 1.0f); 
    }
}

float juliaColor(float x, float y, float sx, float sy, int p) {
    float a_real = x;
    float a_imag = y;
    float seed_real = sx;
    float seed_imag = sy;

    for (int i = 0; i < p; i++) {
        float a_real_squared = a_real * a_real - a_imag * a_imag; 
        float a_imag_squared = 2 * a_real * a_imag; 
        a_real = a_real_squared + seed_real; 
        a_imag = a_imag_squared + seed_imag; 

        if (a_real * a_real + a_imag * a_imag > 4.0f) {
            return 1.0f - (float)i / p; 
        }
    }
    return 0.0f; 
}

void exampleCPU() {
    for (int i = 0; i < SCREEN_Y; i++) {
        for (int j = 0; j < SCREEN_X; j++) {
            float x = scale * (j - SCREEN_X / 2);
            float y = scale * (i - SCREEN_Y / 2);
            float4* p = pixels + (i * SCREEN_X + j);
            float color = juliaColor(x, y, mx, my, precision);
            p->x = color; 
            p->y = color; 
            p->z = color; 
            p->w = 1.0f;  
        }
    }
}

void calculate() {
    frame++;
    int timecur = glutGet(GLUT_ELAPSED_TIME);

    if (timecur - timebase > FPS_UPDATE) {
        char t[200];
        char* m = "";
        switch (mode) {
        case CPU_MODE: m = "CPU mode"; break;
        case GPU_MODE: m = "GPU mode"; break;
        case GPU2_MODE: m = "GPU2 mode"; break;
        }
        sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
        glutSetWindowTitle(t);
        timebase = timecur;
        frame = 0;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((SCREEN_X + blockSize.x - 1) / blockSize.x, (SCREEN_Y + blockSize.y - 1) / blockSize.y);

    switch (mode) {
    case CPU_MODE: exampleCPU(); break;
    case GPU_MODE:
        juliaKernel << <gridSize, blockSize >> > (d_pixels, mx, my, scale, precision);
        cudaDeviceSynchronize(); // Wait for the GPU to finish
        cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost); 
        break;
    case GPU2_MODE:
        juliaKernel << <gridSize, blockSize >> > (d_pixels, mx, my, scale, precision);
        cudaDeviceSynchronize(); // Wait for the GPU to finish
        cudaMemcpy(pixels, d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaMemcpyDeviceToHost); 
        break;
    }
}


void render() {
    calculate();
    glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels);
    glutSwapBuffers();
}

void idle() {
    glutPostRedisplay();
}

void initCPU() {
    pixels = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4));
}

void cleanCPU() {
    free(pixels);
}

void initGPU() {
    cudaMalloc((void**)&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4)); 
    pixels = (float4*)malloc(SCREEN_X * SCREEN_Y * sizeof(float4)); 
}

void cleanGPU() {
    cudaFree(d_pixels);
    free(pixels); 
}

void initGPU2() {
    cudaMalloc((void**)&d_pixels, SCREEN_X * SCREEN_Y * sizeof(float4)); 
    cudaHostAlloc((void**)&pixels, SCREEN_X * SCREEN_Y * sizeof(float4), cudaHostAllocMapped);
}

void cleanGPU2() {
    cudaFree(d_pixels); 
    cudaFreeHost(pixels);
}

void init() {
    if (mode == CPU_MODE) {
        initCPU();
    }
    else if (mode == GPU2_MODE) {
        initGPU2();
    }
    else {
        initGPU();
    }
}

void clean() {
    if (mode == CPU_MODE) {
        cleanCPU();
    }
    else if (mode == GPU2_MODE) {
        cleanGPU2();
    }
    else {
        cleanGPU();
    }
}

void toggleMode(int m) {
    clean();
    mode = m;
    init();
}

void mouse(int button, int state, int x, int y) {
    if (button <= 2 && state == GLUT_DOWN) {
        mx = (float)(scale * (x - SCREEN_X / 2));
        my = -(float)(scale * (y - SCREEN_Y / 2));
    }
    if (button == 3) scale /= 1.05f;
    else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y) {
    mx = (float)(scale * (x - SCREEN_X / 2));
    my = -(float)(scale * (y - SCREEN_Y / 2));
}

void processNormalKeys(unsigned char key, int x, int y) {
    if (key == 27) {
        clean();
        exit(0);
    }
    else if (key == '1') toggleMode(CPU_MODE);
    else if (key == '2') toggleMode(GPU_MODE);
    else if (key == '3') toggleMode(GPU2_MODE);
    else if (key == 43) precision *= 2;
    else if (key == 45) if (precision > 1) precision /= 2;

}

void processSpecialKeys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP:
        break;
    case GLUT_KEY_DOWN:
        break;
    }
}

void initGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(SCREEN_X, SCREEN_Y);
    glutCreateWindow(TITLE);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.375, 0.375, 0);
}

int main(int argc, char** argv) {
    initGL(argc, argv);
    initCPU();

    glutDisplayFunc(render);
    glutIdleFunc(idle);
    glutMotionFunc(mouseMotion);
    glutMouseFunc(mouse);
    glutKeyboardFunc(processNormalKeys);
    glutSpecialFunc(processSpecialKeys);

    glutMainLoop();

    cleanCPU();
    cleanGPU();
    return 1;
}



