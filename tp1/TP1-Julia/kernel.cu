
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

GLuint imageTex;
GLuint imageBuffer;
float* debug;

/* Globals */
float scale = 0.003f;
float mx, my;
int mode = CPU_MODE;
int frame = 0;
int timebase = 0;

float4 *pixels;

void initCPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanCPU()
{
	free(pixels);
}

void initGPU()
{
	pixels = (float4*)malloc(SCREEN_X*SCREEN_Y*sizeof(float4));
}

void cleanGPU()
{
	free(pixels);
}

void exampleCPU()
{
	int i, j;
	for (i = 0; i<SCREEN_Y; i++)
	for (j = 0; j<SCREEN_X; j++)
	{
		float x = (float)(scale*(j - SCREEN_X / 2));
		float y = (float)(scale*(i - SCREEN_Y / 2));
		float4* p = pixels + (i*SCREEN_X + j);
		// default: black
		p->x = 0.0f;
		p->y = 0.0f;
		p->z = 0.0f;
		p->w = 1.0f;
		if (sqrt((x - mx)*(x - mx) + (y - my)*(y - my))<0.01)
			p->x = 1.0f;
		else if ((i == SCREEN_Y / 2) || (j == SCREEN_X / 2))
		{
			p->x = 1.0f;
			p->y = 1.0f;
			p->z = 1.0f;
		}
	}
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
		}
		sprintf(t, "%s:  %s, %.2f FPS", TITLE, m, frame * 1000 / (float)(timecur - timebase));
		glutSetWindowTitle(t);
		timebase = timecur;
		frame = 0;
	}

	switch (mode)
	{
	case CPU_MODE: exampleCPU(); break;
	case GPU_MODE: exampleCPU(); break;
	}
}

void idle()
{
	glutPostRedisplay();
}


void render()
{
	calculate();
	switch (mode)
	{
	case CPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	case GPU_MODE: glDrawPixels(SCREEN_X, SCREEN_Y, GL_RGBA, GL_FLOAT, pixels); break;
	}
	glutSwapBuffers();
}

void clean()
{
	switch (mode)
	{
	case CPU_MODE: cleanCPU(); break;
	case GPU_MODE: cleanGPU(); break;
	}
}

void init()
{
	switch (mode)
	{
	case CPU_MODE: initCPU(); break;
	case GPU_MODE: initGPU(); break;
	}

}

void toggleMode(int m)
{
	clean();
	mode = m;
	init();
}

void mouse(int button, int state, int x, int y)
{
	if (button <= 2)
	{
		mx = (float)(scale*(x - SCREEN_X / 2));
		my = -(float)(scale*(y - SCREEN_Y / 2));
	}
	// Wheel reports as button 3 (scroll up) and button 4 (scroll down)
	if (button == 3) scale /= 1.05f;
	else if (button == 4) scale *= 1.05f;
}

void mouseMotion(int x, int y)
{
	mx = (float)(scale*(x - SCREEN_X / 2));
	my = -(float)(scale*(y - SCREEN_Y / 2));
}

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27) { clean(); exit(0); }
	else if (key == '1') toggleMode(CPU_MODE);
	else if (key == '2') toggleMode(GPU_MODE);
}

void processSpecialKeys(int key, int x, int y) {
	// other keys (F1, F2, arrows, home, etc.)
	switch (key) {
	case GLUT_KEY_UP: break;
	case GLUT_KEY_DOWN: break;
	}
}

void initGL(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(SCREEN_X, SCREEN_Y);
	glutCreateWindow(TITLE);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glDisable(GL_DEPTH_TEST);

	// View Ortho
	// Sets up the OpenGL window so that (0,0) corresponds to the top left corner, 
	// and (SCREEN_X,SCREEN_Y) corresponds to the bottom right hand corner.  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_X, SCREEN_Y, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.375, 0.375, 0); // Displacement trick for exact pixelization
}


int main(int argc, char **argv) {

	initGL(argc, argv);

	init();

	glutDisplayFunc(render);
	glutIdleFunc(idle);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);

	// enter GLUT event processing cycle
	glutMainLoop();

	clean();

	return 1;
}
