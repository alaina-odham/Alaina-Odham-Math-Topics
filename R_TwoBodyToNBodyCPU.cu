// Name:Alaina Odham
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
// To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
//#define MASS 10.0  	                                                          //not used
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

#define NUMBER_OF_SPHERES 5                                                       //add editable sphere number

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

/*
float px1, py1, pz1, vx1, vy1, vz1, fx1, fy1, fz1, mass1;                        //change this
float px2, py2, pz2, vx2, vy2, vz2, fx2, fy2, fz2, mass2;
*/

float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES];       //do this in order to cycle through spheres
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];       //p position, v velocity, and f force
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
float mass[NUMBER_OF_SPHERES];                                                   //mass for each sphere

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int i, j, yeahBuddy;                        //add i and j
	float dx, dy, dz, seperation;

	for(i = 0; i < NUMBER_OF_SPHERES; i++)      //put in a for loop (cycle through spheres)
	{
		/*
		px1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		py1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		pz1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		*/
	
		yeahBuddy = 0;                         //0 while not ok at 1 when ok

		while(yeahBuddy == 0)                  //place sphere randomly
		{
			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;    //change px2s to px[i]s etc
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
			yeahBuddy = 1;                            //assume position is ok

			for(j = 0; j < i; j++)                    //add a for loop to check vs other spheres
			{
				dx = px[i] - px[j];                   //change 1s and 2s to [i]s and [j]s
				dy = py[i] - py[j];
				dz = pz[i] - pz[j];
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				/*
				yeahBuddy = 1;
				*/
				if(seperation < DIAMETER) yeahBuddy = 0;
			}
		}
	
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;   //change 1s to [i]s (cycle through spheres)
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;

		/*
		vx2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;     //remove original second sphere version
		vy2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		*/
	
		mass[i] = 1.0;    //change to mass [i]
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)  //add for loop (cycle through spheres)
	{
		glColor3d(1.0,0.5,1.0);                 //color each sphere

		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);      //change 1s to is (cycle through spheres)
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}

	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)         //add for loop to cycle through spheres
	{
		if(px[i] > halfBoxLength)                      //change 1s to [i]s (cycle through spheres) and remove old second sphere version
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
	
		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
			
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

void get_forces()
{
	float dx, dy, dz, r, r2, dvx, dvy, dvz, forceMag, inout;
	int i, j;                                                 //add i and j
	
	for(i = 0; i < NUMBER_OF_SPHERES; i++)                    //add for loop to start all forces at 0
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
	
	for(i = 0; i < NUMBER_OF_SPHERES - 1; i++)               //need forces between all two spheres, add for loops for i sphere and j sphere
	{
		for(j = i+1; j < NUMBER_OF_SPHERES; j++)             //make i and j be different spheres (j = i+1), has no repeats
		{
			dx = px[j] - px[i];                              //change 2 to j and 1 to i
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];
				
			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);

			forceMag =  mass[i]*mass[j]*GRAVITY/r2;
			
			if (r < DIAMETER)
			{
				dvx = vx[j] - vx[i];                        //change 2 to j and 1 to i   
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];

				inout = dx*dvx + dy*dvy + dz*dvz;

				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

			fx[i] += forceMag*dx/r;                         //change 2 to j and 1 to i, change = to += to add up forces
			fy[i] += forceMag*dy/r;
			fz[i] += forceMag*dz/r;
			fx[j] += -forceMag*dx/r;
			fy[j] += -forceMag*dy/r;
			fz[j] += -forceMag*dz/r;
		}
	}
}

void move_bodies(float time)
{  
	float dtf;                                             //make a dt factor so we start at 0.5 and leapfrog for accuracy

	if(time == 0.0)
	{
		dtf = 0.5*DT;
	}
	else
	{
		dtf = DT;
	}

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)            //add for loop to cycle through all spheres [i]
	{
		vx[i] += dtf*(fx[i] - DAMP*vx[i])/mass[i];
		vy[i] += dtf*(fy[i] - DAMP*vy[i])/mass[i];
		vz[i] += dtf*(fz[i] - DAMP*vz[i])/mass[i];
		
		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");                                     //change to N body
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


