#include <stdio.h>
#include <stdlib.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <stdio.h>

#define CPU_MODE 1
#define GPU_MODE 2
#define GPU2_MODE 3

int mode = CPU_MODE;

#define SCREEN_X 1024
#define SCREEN_Y 768
#define FPS_UPDATE 500
#define TITLE "Bugs Cellular Automaton"

// Parameters for the "Bugs" automaton
#define RANGE 5
#define SURVIVELO 34
#define SURVIVEHI 58
#define BIRTHLO 34
#define BIRTHHI 45

GLuint imageTex;
GLuint imageBuffer;
int grid[SCREEN_X][SCREEN_Y];
int nextGrid[SCREEN_X][SCREEN_Y];
float scale = 0.003f;
int frame = 0;
int timebase = 0;

int* d_grid1;
int* d_grid2;

__global__ void updateGridGPU2(int* grid1, int* grid2, int screenX, int screenY, int range, int surviveLo, int surviveHi, int birthLo, int birthHi) {
    __shared__ int tile[16 + 2 * RANGE][16 + 2 * RANGE]; 

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + RANGE; 
    int sharedY = threadIdx.y + RANGE;

    if (x < screenX && y < screenY) {
        tile[sharedX][sharedY] = grid1[x * screenY + y];
    }

    if (threadIdx.x < RANGE && x - RANGE >= 0) {
        tile[sharedX - RANGE][sharedY] = grid1[(x - RANGE) * screenY + y];
    }
    if (threadIdx.x >= blockDim.x - RANGE && x + RANGE < screenX) {
        tile[sharedX + RANGE][sharedY] = grid1[(x + RANGE) * screenY + y];
    }
    if (threadIdx.y < RANGE && y - RANGE >= 0) {
        tile[sharedX][sharedY - RANGE] = grid1[x * screenY + (y - RANGE)];
    }
    if (threadIdx.y >= blockDim.y - RANGE && y + RANGE < screenY) {
        tile[sharedX][sharedY + RANGE] = grid1[x * screenY + (y + RANGE)];
    }

    __syncthreads(); 

    if (x < screenX && y < screenY) {
        int liveNeighbors = 0;

        for (int dx = -range; dx <= range; dx++) {
            for (int dy = -range; dy <= range; dy++) {
                int nx = sharedX + dx;
                int ny = sharedY + dy;
                if (nx >= 0 && nx < blockDim.x + 2 * RANGE && ny >= 0 && ny < blockDim.y + 2 * RANGE) {
                    liveNeighbors += tile[nx][ny];
                }
            }
        }

        if (tile[sharedX][sharedY] == 1) { 
            if (liveNeighbors >= surviveLo && liveNeighbors <= surviveHi) {
                grid2[x * screenY + y] = 1; 
            }
            else {
                grid2[x * screenY + y] = 0; 
            }
        }
        else {
            if (liveNeighbors >= birthLo && liveNeighbors <= birthHi) {
                grid2[x * screenY + y] = 1; 
            }
            else {
                grid2[x * screenY + y] = 0;
            }
        }
    }
}

void updateGridGPU2Function() {
    dim3 blockDim(16, 16);  
    dim3 gridDim((SCREEN_X + blockDim.x - 1) / blockDim.x, (SCREEN_Y + blockDim.y - 1) / blockDim.y);

    updateGridGPU2 << <gridDim, blockDim >> > (d_grid1, d_grid2, SCREEN_X, SCREEN_Y, RANGE, SURVIVELO, SURVIVEHI, BIRTHLO, BIRTHHI);

    cudaDeviceSynchronize();

    // Swap the grids
    int* temp = d_grid1;
    d_grid1 = d_grid2;
    d_grid2 = temp;
}


__global__ void updateGridGPU(int* grid1, int* grid2, int screenX, int screenY, int range, int surviveLo, int surviveHi, int birthLo, int birthHi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < screenX && y < screenY) {
        int liveNeighbors = 0;

        for (int dx = -range; dx <= range; dx++) {
            for (int dy = -range; dy <= range; dy++) {
                int nx = (x + dx + screenX) % screenX;
                int ny = (y + dy + screenY) % screenY;
                if (grid1[nx * screenY + ny] == 1) {
                    liveNeighbors++;
                }
            }
        }

        if (grid1[x * screenY + y] == 1) { // Cell is alive
            if (liveNeighbors >= surviveLo && liveNeighbors <= surviveHi) {
                grid2[x * screenY + y] = 1; // Cell survives
            }
            else {
                grid2[x * screenY + y] = 0; // Cell dies
            }
        }
        else { // Cell is dead
            if (liveNeighbors >= birthLo && liveNeighbors <= birthHi) {
                grid2[x * screenY + y] = 1; // Cell is born
            }
            else {
                grid2[x * screenY + y] = 0; // Cell remains dead
            }
        }
    }
}

void updateGridGPUFunction() {
    dim3 blockDim(16, 16);  
    dim3 gridDim((SCREEN_X + blockDim.x - 1) / blockDim.x, (SCREEN_Y + blockDim.y - 1) / blockDim.y);

    updateGridGPU << <gridDim, blockDim >> > (d_grid1, d_grid2, SCREEN_X, SCREEN_Y, RANGE, SURVIVELO, SURVIVEHI, BIRTHLO, BIRTHHI);

    cudaDeviceSynchronize();

    // Swap the grids
    int* temp = d_grid1;
    d_grid1 = d_grid2;
    d_grid2 = temp;
}

void cleanCPU() {
    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            grid[i][j] = 0;
            nextGrid[i][j] = 0;
        }
    }
}

void cleanGPU() {
    cudaFree(d_grid1);
    cudaFree(d_grid2);
}

void clean() {
    if (mode == CPU_MODE) {
        cleanCPU();
    }
    else {
        cleanGPU();
    }
}


void initCPU() {
    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            grid[i][j] = rand() % 2; // Randomly initialize cells (alive or dead)
        }
    }
}

void initGPU() {
    cudaMalloc((void**)&d_grid1, SCREEN_X * SCREEN_Y * sizeof(int));
    cudaMalloc((void**)&d_grid2, SCREEN_X * SCREEN_Y * sizeof(int));

    int* h_grid = (int*)malloc(SCREEN_X * SCREEN_Y * sizeof(int));
    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            h_grid[i * SCREEN_Y + j] = rand() % 2;
        }
    }

    cudaMemcpy(d_grid1, h_grid, SCREEN_X * SCREEN_Y * sizeof(int), cudaMemcpyHostToDevice);

    free(h_grid); 
}

void init() {
    if (mode == CPU_MODE) {
        initCPU();
    }
    else {
        initGPU();
    }
}

void toggleMode(int m) {
    clean();
    mode = m;
    init();
}

void processNormalKeys(unsigned char key, int x, int y) {
    if (key == 27) {
        clean();
        exit(0);
    }
    else if (key == '1') toggleMode(CPU_MODE);
    else if (key == '2') toggleMode(GPU_MODE);
    else if (key == '3') toggleMode(GPU2_MODE);
}

void countNeighbors(int x, int y, int* liveNeighbors) {
    *liveNeighbors = 0;
    for (int dx = -RANGE; dx <= RANGE; dx++) {
        for (int dy = -RANGE; dy <= RANGE; dy++) {
            int nx = (x + dx + SCREEN_X) % SCREEN_X; 
            int ny = (y + dy + SCREEN_Y) % SCREEN_Y;
            if (grid[nx][ny] == 1) {
                (*liveNeighbors)++;
            }
        }
    }
}

void updateGridCPU() {
    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            int liveNeighbors = 0;
            countNeighbors(i, j, &liveNeighbors);

            if (grid[i][j] == 1) { // Cell is alive
                if (liveNeighbors >= SURVIVELO && liveNeighbors <= SURVIVEHI) {
                    nextGrid[i][j] = 1; // Cell survives
                }
                else {
                    nextGrid[i][j] = 0; // Cell dies
                }
            }
            else { // Cell is dead
                if (liveNeighbors >= BIRTHLO && liveNeighbors <= BIRTHHI) {
                    nextGrid[i][j] = 1; // Cell is born
                }
                else {
                    nextGrid[i][j] = 0; // Cell remains dead
                }
            }
        }
    }

    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            grid[i][j] = nextGrid[i][j];
        }
    }
}

void renderCPU() {
    int timecur = glutGet(GLUT_ELAPSED_TIME);

    if (timecur - timebase > FPS_UPDATE) {
        char t[200];
        sprintf(t, "%s, %.2f FPS", TITLE, frame * 1000 / (float)(timecur - timebase));
        glutSetWindowTitle(t);
        timebase = timecur;
        frame = 0;
    }

    updateGridCPU();

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            if (grid[i][j] == 1) {
                glColor3f(1.0f, 1.0f, 1.0f); // Alive cells in white
            }
            else {
                glColor3f(0.0f, 0.0f, 0.0f); // Dead cells in black
            }
            glVertex2i(i, j);
        }
    }

    glEnd();
    glutSwapBuffers();
    frame++;
}

void renderGPU() {
    int timecur = glutGet(GLUT_ELAPSED_TIME);

    if (timecur - timebase > FPS_UPDATE) {
        char t[200];
        sprintf(t, "%s, %.2f FPS", TITLE, frame * 1000 / (float)(timecur - timebase));
        glutSetWindowTitle(t);
        timebase = timecur;
        frame = 0;
    }

    if (mode == GPU2_MODE) {
        updateGridGPU2Function(); 
    }
    else {
        updateGridGPUFunction(); 
    }

    int* h_grid = (int*)malloc(SCREEN_X * SCREEN_Y * sizeof(int));
    cudaMemcpy(h_grid, d_grid1, SCREEN_X * SCREEN_Y * sizeof(int), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    for (int i = 0; i < SCREEN_X; i++) {
        for (int j = 0; j < SCREEN_Y; j++) {
            if (h_grid[i * SCREEN_Y + j] == 1) {
                glColor3f(1.0f, 1.0f, 1.0f); // Alive cells in white
            }
            else {
                glColor3f(0.0f, 0.0f, 0.0f); // Dead cells in black
            }
            glVertex2i(i, j);
        }
    }

    glEnd();
    glutSwapBuffers();
    frame++;

    free(h_grid);
}



void render() {
    if (mode == CPU_MODE) {
        renderCPU();
    }
    else {
        renderGPU();
    }
}

void idle() {
    glutPostRedisplay();
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
}

int main(int argc, char** argv) {
    initGL(argc, argv);
    init();

    glutDisplayFunc(render);
    glutIdleFunc(idle);
    glutKeyboardFunc(processNormalKeys);

    glutMainLoop();
    clean();

    return 0;
}

