/*
Raylib example file.
This is an example main file for a simple raylib project.
Use this as a starting point or replace it with your code.

by Jeffery Myers is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/

*/

// #include "resource_dir.h"	// utility header for SearchAndSetResourceDir
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define PI 3.14159365258979

struct Color {
	int ColorRGBA[4];
};

// Graphing Utilities
// // // // //

struct Color HSVtoRGB(float h, float s, float v) {
    float c = v * s;                   // chroma
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
    float m = v - c;

    float r, g, b;
    if      (h < 60)  { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else              { r = c; g = 0; b = x; }

    unsigned char R = (unsigned char)((r + m) * 255);
    unsigned char G = (unsigned char)((g + m) * 255);
    unsigned char B = (unsigned char)((b + m) * 255);

	struct Color myclr;
	
	myclr.ColorRGBA[0] = R;
	myclr.ColorRGBA[1] = G;
	myclr.ColorRGBA[2] = B;
	myclr.ColorRGBA[3] = 255;

    return myclr;
}

// Angle in radians, returns rainbow color wheel smoothly wrapping
struct Color AngleToRainbow(double angle) {
    // Normalize angle into [0, 2π)
    while (angle < 0) angle += 2*PI;
    while (angle >= 2*PI) angle -= 2*PI;

    // Map [0, 2π) → [0, 360)
    float hue = (float)(angle * 180.0 / PI);
    return HSVtoRGB(hue, 1.0f, 1.0f);
}

struct Color PhiToColor(double phi) {
    // Clamp φ
    if (phi < -1.0) phi = -1.0;
    if (phi >  1.0) phi =  1.0;

    float h; // hue
    if (phi < -0.3333) {
        // [-1, -1/3] : blue → purple
        h = 240.0f + (phi + 1.0f) / (2.0f/3.0f) * (270.0f - 240.0f);
    } else if (phi < 0.3333) {
        // [-1/3, 1/3] : purple → magenta
        h = 270.0f + (phi + 0.3333f) / (2.0f/3.0f) * (300.0f - 270.0f);
    } else {
        // [1/3, 1] : magenta → red
        h = 300.0f + (phi - 0.3333f) / (2.0f/3.0f) * (360.0f - 300.0f);
        if (h >= 360.0f) h -= 360.0f; // wrap around
    }

    float s = 1.0f; // full saturation
    float v = 1.0f; // full brightness

    return HSVtoRGB(h, s, v);
}

double log_scale(double input) {
    if (input >= 1.0) return 1.0;
    if (input <= 0.0) return 0.0;

    double log_val = log10(input);  // log10 of input
    double scaled = (log_val + 9.0) / 9.0;  // map -9.0 → 0..1
    return scaled;
}



struct Color GetRainbowColor(float t, double tr) {
    // Clamp t between 0 and 1
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    // Map t to HSV hue from 0 (red) to 300 (purple), skipping magenta
    float hue = t * 300.0f;   // hue in degrees (0-360)
    float s = 1.0f;           // full saturation
    float v = 1.0f;           // full value

    // Convert HSV to RGB
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float rf, gf, bf;

    if (hue < 60)      { rf = c; gf = x; bf = 0; }
    else if (hue < 120){ rf = x; gf = c; bf = 0; }
    else if (hue < 180){ rf = 0; gf = c; bf = x; }
    else if (hue < 240){ rf = 0; gf = x; bf = c; }
    else if (hue < 300){ rf = x; gf = 0; bf = c; }
    else               { rf = c; gf = 0; bf = x; }

    // Convert to 0-255 range
    unsigned char r = (unsigned char)((rf + m) * 255*tr);
    unsigned char g = (unsigned char)((gf + m) * 255*tr);
    unsigned char b = (unsigned char)((bf + m) * 255*tr);
    unsigned char a = (int)(255);

	struct Color myclr;
	
	myclr.ColorRGBA[0] = r;
	myclr.ColorRGBA[1] = g;
	myclr.ColorRGBA[2] = b;
	myclr.ColorRGBA[3] = a;

    return myclr;
}

// EARLY DEFINITIONS
// // // // //

#define gridsize 100

struct Field2D {
	double field[gridsize][gridsize][9];
};

struct Field2D fieldo;

struct Field2D next_field;

int DiscreteVelociyVectors[9][2] = {
	{-1,1},
	{0,1},
	{1,1},
	{-1,0},
	{0,0},
	{1,0},
	{-1,-1},
	{0,-1},
	{1,-1}
};

struct Field2D df;

// calculate order parameter:
double phiplusphi3[gridsize][gridsize];
double lapphiplusphi3[gridsize][gridsize];
double GradOrderParamField[gridsize][gridsize][2];
double vdotgrad[gridsize][gridsize];
double velocityField[gridsize][gridsize][2];
double DensityField[gridsize][gridsize];
int SolidField[gridsize][gridsize];
double SolidVelocityField[gridsize][gridsize][2];
double OrderParamField[gridsize][gridsize];
double OrderCubedParamField[gridsize][gridsize];
double LapOrderParamField[gridsize][gridsize];
double LapLapOrderParamField[gridsize][gridsize];
double chempotfield[gridsize][gridsize];
double lapchempotfield[gridsize][gridsize];
#define SpeedOfSound (1.0/1.7320508075688772)
#define TimeRelaxationConstant 0.9
#define defaultSpeedX 0.0
#define defaultSpeedY 0.0
#define PeriodicBCX 1
#define PeriodicBCY 1
#define MFluid 0.1
#define fluida -0.1
#define fluidb 0.1
#define KFluid 0.1

double getMomentum(struct Field2D exfield, int x, int y) {
	double xmom = 0;
	double ymom = 0;
	xmom = velocityField[x][y][0] * DensityField[x][y];
	ymom = velocityField[x][y][1] * DensityField[x][y];
	return pow(pow(xmom,2) + pow(ymom,2), 0.5);
}

double atan3(double x, double y) {
    double angle = atan2(y, x);   // range (-π, π]
    if (angle < 0) {
        angle += 2.0 * PI;      // shift into [0, 2π)
    }
    return angle;
}

int opposite[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};

double Weights[9] = {1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36};
double BodyForceTable[gridsize][gridsize][2];

// Field Updating Functions
// // // // //



double getVal(double exfield[gridsize][gridsize], int x, int y) {
	int xi, yi;
    if (PeriodicBCX == 0) {
        if (x < 0) xi = 0;
        if (x >= gridsize) xi = gridsize - 1;
    } else {
        xi = (x + gridsize) % gridsize;
    }
	if (PeriodicBCY == 0) {
        if (y < 0) yi = 0;
        if (y >= gridsize) yi = gridsize - 1;
    } else {
        yi = (y + gridsize) % gridsize;
    }
	return exfield[xi][yi];
}

// 9-point Laplacian
double computeLaplacianEl(double exfield[gridsize][gridsize], int x, int y, int useGhosts) {
    double c  = getVal(exfield, x, y);
    double n  = getVal(exfield, x, y+1);
    double s  = getVal(exfield, x, y-1);
    double e  = getVal(exfield, x+1, y);
    double w  = getVal(exfield, x-1, y);
    double ne = getVal(exfield, x+1, y+1);
    double nw = getVal(exfield, x-1, y+1);
    double se = getVal(exfield, x+1, y-1);
    double sw = getVal(exfield, x-1, y-1);

    return (-20.0*c + 4.0*(n+s+e+w) + (ne+nw+se+sw)) / 6.0;
}

// Compute gradient field
void gradientField(double exfield[gridsize][gridsize], double gradexfield[gridsize][gridsize][2], int useGhosts) {
    for (int x = 0; x < gridsize; x++) {
        for (int y = 0; y < gridsize; y++) {
            double dx = (getVal(exfield, x+1, y) - getVal(exfield, x-1, y)) * 0.5;
            double dy = (getVal(exfield, x, y+1) - getVal(exfield, x, y-1)) * 0.5;

            gradexfield[x][y][0] = dx; // ∂φ/∂x
            gradexfield[x][y][1] = dy; // ∂φ/∂y
        }
    }
}

// Compute laplacian field
void laplacianField(double exfield[gridsize][gridsize], double lap[gridsize][gridsize], int useGhosts) {
    for (int x = 0; x < gridsize; x++) {
        for (int y = 0; y < gridsize; y++) {
            lap[x][y] = computeLaplacianEl(exfield, x, y, useGhosts);
        }
    }
}

// Creating Geometries
// // // // //

void MakeSolidDisc(double r, double x0, double y0) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( pow((pow((double)x-x0, 2) + pow((double)y-y0, 2)), 0.5) < r) {
				SolidField[x][y] = 1;
			}
		}
	}
}

void MakeLiquidDisc(double r, double x0, double y0, double orderParam) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( pow((pow((double)x-x0, 2) + pow((double)y-y0, 2)), 0.5) < r) {
				SolidField[x][y] = 0;
				OrderParamField[x][y] = orderParam;
			}
		}
	}
}

void MakeSolidRectangle(double x0, double y0, double x1, double y1) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( (double)x-x0 < x1 && (double)x-x0 > 0 && (double)y-y0 < y1 && (double)y-y0 > 0) {
				SolidField[x][y] = 1;
			}
		}
	}
}

void MakeLiquidRectangle(double x0, double y0, double x1, double y1) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( (double)x-x0 < x1 && (double)x-x0 > 0 && (double)y-y0 < y1 && (double)y-y0 > 0) {
				SolidField[x][y] = 0;
			}
		}
	}
}



int main ()
{

	setbuf(stdout, NULL);

	srand(time(NULL));

	int x,y;


	// double BodyForce[2] = {0.0,0.0};

	for(x=0;x<gridsize;x++) {
		for(y=0;y<gridsize;y++) {
			DensityField[x][y] = 1.0;
		}
	}

	for(x=0;x<gridsize;x++) {
		for(y=0;y<gridsize;y++) {
			BodyForceTable[x][y][0] = 0.0;
			BodyForceTable[x][y][1] = 0.0;
		}
	}

	for(x=0;x<gridsize;x++) {
		for(y=0;y<gridsize;y++) {
			SolidField[x][y] = 0;
		}
	}

	for(x=0;x<gridsize;x++) {
    	for(y=0;y<gridsize;y++) {
        	// OrderParamField[x][y] = -1.0;
			// Uniform noise in [-noiseAmplitude, +noiseAmplitude]
            double rndom = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            OrderParamField[x][y] = 0.0 + 0.05 * rndom;
    	}
	}

	// MakeLiquidDisc(20,20,20, 1.0);
	// MakeLiquidDisc(20,80,80, -1.0);
	// MakeLiquidDisc(8,70,75);
	// MakeLiquidDisc
	// MakeLiquidDisc(10,75,75);

	// MakeSolidRectangle(60,60,10,10);

	// MakeSolidRectangle(75, 100, 10, 10);
	// MakeSolidDisc(5,90,75);
	
	// DensityField[25][25] = 40.0;

	// for(x=6;x<=9;x++) {
	// 	for(y=6;y<=9;y++) {
	// 		DensityField[x][y] = 5.0;
	// 	}
	// }

	// for(x=43;x<=51;x++) {
	// 	for(y=43;y<=51;y++) {
	// 		DensityField[x][y] = 5;
	// 	}
	// }
	// velocityField[25][25][1] = 0.1;
	double default_val[9] = {0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0};
	for(x=0;x<gridsize;x++) {
    	for(y=0;y<gridsize;y++) {
        	for(int k=0; k<9; k++) {
            	fieldo.field[x][y][k] = default_val[k];
        	}
    	}
	}

	for(x=0;x<gridsize;x++) {
		for(y=0;y<gridsize;y++) {
			if(SolidField[x][y]==1) {continue;}
			velocityField[x][y][0] = defaultSpeedX;
			velocityField[x][y][1] = defaultSpeedY;
		}
	}

	// collision step
	int i,j;
	double gradchempotfield[gridsize][gridsize][2];
	
	int pos = 10;

	int phi_timesteps = 20;
	
	int rhythm = 0;

	FILE *log = fopen("debug.log", "w");

	int l;
	int time_steps_l = 5000;
	// double OrderParamTable[time_steps_l][gridsize][gridsize];
	double *OrderParamTable = malloc(time_steps_l * gridsize * gridsize * sizeof(double));
	double *MomentumTable = malloc(time_steps_l * gridsize * gridsize * sizeof(double));
	#define O(l,x,y) OrderParamTable[((l) * gridsize + (x)) * gridsize + (y)]
	#define Mtbl(l,x,y) MomentumTable[((l) * gridsize + (x)) * gridsize + (y)]

	for (l=0;l<time_steps_l;l++) {
		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				for(int k=0; k<9; k++) {
					df.field[x][y][k] = default_val[k];
				}
			}
		}

		// double FlowVelocity[2] = {(amp/pow(freq,2))*sin(((double)rhythm)/freq), 0.0};

		// double BodyForce[2] = {(amp/pow(freq,2))*sin(((double)rhythm)/freq), 0.0};

		for (i=0;i<phi_timesteps;i++) {
			for(x=0;x<gridsize;x++) {
				for(y=0;y<gridsize;y++) {
					OrderCubedParamField[x][y] = pow(OrderParamField[x][y], 3.0);
				}
			}
			laplacianField(OrderParamField, LapOrderParamField, 0);
			
			for(x=0;x<gridsize;x++) {
				for(y=0;y<gridsize;y++) {
					phiplusphi3[x][y] = fluida*OrderParamField[x][y] + fluidb*OrderCubedParamField[x][y];
				}	
			}
			
			laplacianField(phiplusphi3, lapphiplusphi3, 0);
			for(x=0;x<gridsize;x++) {
				for(y=0;y<gridsize;y++) {
					chempotfield[x][y] = phiplusphi3[x][y] - KFluid*LapOrderParamField[x][y];
				}
			}

			// update order parameter
			
			gradientField(OrderParamField, GradOrderParamField, 0);
			laplacianField(chempotfield, lapchempotfield, 0);
			for(x=0;x<gridsize;x++) {
				for(y=0;y<gridsize;y++) {
					vdotgrad[x][y] = velocityField[x][y][0]*GradOrderParamField[x][y][0]
					+ velocityField[x][y][1]*GradOrderParamField[x][y][1];
					// vdotgrad[x][y] = 0.0;
				}
			}
			double dt_phi = 1.0/((double)phi_timesteps);
			for(x=0;x<gridsize;x++) {
				for(y=0;y<gridsize;y++) {
					OrderParamField[x][y] = OrderParamField[x][y] + dt_phi*(MFluid*lapchempotfield[x][y] 
					- vdotgrad[x][y]);
				}
			}
		}

		// calculate body force

		
		gradientField(chempotfield, gradchempotfield, 0);
		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				BodyForceTable[x][y][0] = -OrderParamField[x][y]*gradchempotfield[x][y][0];
				BodyForceTable[x][y][1] = -OrderParamField[x][y]*gradchempotfield[x][y][1];
				double max_force = 0.2;
				if (BodyForceTable[x][y][0] > max_force) BodyForceTable[x][y][0] = max_force;
				if (BodyForceTable[x][y][0] < -max_force) BodyForceTable[x][y][0] = -max_force;
				if (BodyForceTable[x][y][1] > max_force) BodyForceTable[x][y][1] = max_force;
				if (BodyForceTable[x][y][1] < -max_force) BodyForceTable[x][y][1] = -max_force;

			}
		}

		

		// collision step:

		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				if (SolidField[x][y] == 1) {
					continue;
				}
				double BodyForce[2] = {0.0,0.0};
				BodyForce[0]=BodyForceTable[x][y][0];
				BodyForce[1]=BodyForceTable[x][y][1];
				for(int v=0; v<9; v++) {
					double Velocity = fieldo.field[x][y][v];
					double FirstTerm = Velocity;
					double FlowVelocity[2] = {velocityField[x][y][0], velocityField[x][y][1]};
					double Dotted = (
						FlowVelocity[0]*(double)DiscreteVelociyVectors[v][0]
						+ FlowVelocity[1]*(double)DiscreteVelociyVectors[v][1]
					);
					double taylor = (
						1
						+ ((Dotted)/pow(SpeedOfSound, 2))
						+ (pow(Dotted,2) / (2*pow(SpeedOfSound, 4)))
						- (
							(pow(FlowVelocity[0],2) + pow(FlowVelocity[1],2))
							/ (2*pow(SpeedOfSound,2))
						)
					);
					double density = DensityField[x][y];
					double equilibrium = density * taylor * Weights[v];

					double dotcF = (double)DiscreteVelociyVectors[v][0]*BodyForce[0] + 
								(double)DiscreteVelociyVectors[v][1]*BodyForce[1];
					double dotcu = (double)DiscreteVelociyVectors[v][0]*FlowVelocity[0] + 
								(double)DiscreteVelociyVectors[v][1]*FlowVelocity[1];

					double ThirdTerm = (1.0 - 0.5/TimeRelaxationConstant) * Weights[v] * (
						(((double)DiscreteVelociyVectors[v][0] - FlowVelocity[0]) * BodyForce[0] +
						((double)DiscreteVelociyVectors[v][1] - FlowVelocity[1]) * BodyForce[1]) / (SpeedOfSound*SpeedOfSound)
					+ (dotcu * dotcF) / (SpeedOfSound*SpeedOfSound*SpeedOfSound*SpeedOfSound)
					);


					double SecondTerm = (equilibrium - Velocity) / TimeRelaxationConstant;
					df.field[x][y][v] = FirstTerm + SecondTerm + ThirdTerm;

				}
			}
		}

		// streaming step:

		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				if(SolidField[x][y] == 1) {
					continue;
				}
				for(int v=0; v<9; v++) {

					int tx = x + DiscreteVelociyVectors[v][0];
					int ty = y + DiscreteVelociyVectors[v][1];

					if (tx >= 0 && tx < gridsize && ty >= 0 && ty < gridsize) {
						// do normal stuff
						if (SolidField[tx][ty] == 1) {
							fieldo.field[x][y][opposite[v]] = df.field[x][y][v];
						}
						else {
							fieldo.field[tx][ty][v] = df.field[x][y][v];
						}
						
					}
					else {
						// boundaries!
						// periodic
						
						// neighbor outside → bounce back into opposite direction
						// fieldo.field[x][y][opposite[v]] = df.field[x][y][v];

						if (PeriodicBCX == 1 && PeriodicBCY == 0) {
							if(ty==-1 || ty == gridsize) {
								fieldo.field[x][y][opposite[v]] = df.field[x][y][v];
							}
							else {
								tx = (tx+gridsize) % gridsize;
								ty = (ty+gridsize) % gridsize;
								fieldo.field[tx][ty][v] = df.field[x][y][v];
							}
						}
						else if (PeriodicBCX == 0 && PeriodicBCY == 1) {
							if(tx==-1 || tx == gridsize) {
								fieldo.field[x][y][opposite[v]] = df.field[x][y][v];
							}
							else {
								tx = (tx+gridsize) % gridsize;
								ty = (ty+gridsize) % gridsize;
								fieldo.field[tx][ty][v] = df.field[x][y][v];
							}
						}
						else if (PeriodicBCX == 1 && PeriodicBCY == 1) {
							tx = (tx+gridsize) % gridsize;
							ty = (ty+gridsize) % gridsize;
							fieldo.field[tx][ty][v] = df.field[x][y][v];
						}
						else {
							fieldo.field[x][y][opposite[v]] = df.field[x][y][v];
						}
					}
				}
			}
		}

		// update variables

		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				double sum = 0;
				int v;
				for (v=0; v < 9; v++) {
					sum += fieldo.field[x][y][v];
				}
				
				double BodyForce[2] = {0.0,0.0};
				BodyForce[0]=BodyForceTable[x][y][0];
				BodyForce[1]=BodyForceTable[x][y][1];
				
				DensityField[x][y] = sum;
				double FlowVelocity[2] = {0.0, 0.0};
				for (v=0; v < 9; v++) {
					FlowVelocity[0] = (
						FlowVelocity[0] 
						+ (double)DiscreteVelociyVectors[v][0]
						* fieldo.field[x][y][v]
					);
				}
				for (v=0; v < 9; v++) {
					FlowVelocity[1] = (
						FlowVelocity[1] 
						+ (double)DiscreteVelociyVectors[v][1]
						* fieldo.field[x][y][v]
					);
				}
				if (DensityField[x][y] < 0.5) {
					DensityField[x][y] = 0.5;
				} 
				FlowVelocity[0] = (FlowVelocity[0] + 0.5*BodyForce[0]) / DensityField[x][y];
				FlowVelocity[1] = (FlowVelocity[1] + 0.5*BodyForce[1]) / DensityField[x][y];
				double vmag = pow(pow(FlowVelocity[0], 2) + pow(FlowVelocity[1], 2) ,0.5);

				if (vmag > SpeedOfSound - 0.01) {
					FlowVelocity[0] = (SpeedOfSound - 0.01)*FlowVelocity[0]/vmag;
					FlowVelocity[1] = (SpeedOfSound - 0.01)*FlowVelocity[1]/vmag;
				}

				velocityField[x][y][0] = FlowVelocity[0];
				velocityField[x][y][1] = FlowVelocity[1];
			}
		}

		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				O(l, x, y) = OrderParamField[x][y];
				Mtbl(l, x, y) = getMomentum(fieldo,x,y);
			}
		}
		if (l % 10 == 1) {
			if (log) {
				fprintf(log, "Step %d complete\n", l);
				fflush(log);  // ensures it’s written immediately
			}
		}
	}

	l = 0;

	#define WindowSize 200

	// Create folder
	mkdir("frames", 0777);

	// Allocate image buffer
	unsigned char *image = malloc(WindowSize * WindowSize * 3);
	int jindex=0;
	// Loop over saved timesteps
	for (int l = 0; l < time_steps_l; l++) {
		if((l%40)!= 1){continue;}
		// Fill image from stored order parameter
		jindex=jindex+1;
		for (int y = 0; y < WindowSize; y++) {
			for (int x = 0; x < WindowSize; x++) {

				int gx = (int)(x * gridsize / WindowSize);
				int gy = (int)(y * gridsize / WindowSize);

				double phi = O(l, gx, gy);

				struct Color c = PhiToColor(phi);

				int index = (y * WindowSize + x) * 3;
				image[index + 0] = c.ColorRGBA[0];
				image[index + 1] = c.ColorRGBA[1];
				image[index + 2] = c.ColorRGBA[2];
			}
		}

		char filename[64];
		sprintf(filename, "frames/frame_%05d.png", jindex);

		stbi_write_png(filename, WindowSize, WindowSize, 3,
					image, WindowSize * 3);

		printf("Wrote %s\n", filename);
	}

    system("ffmpeg -framerate 6 "
           "-i frames/frame_%05d.png "
           "-c:v libx264 -pix_fmt yuv420p output.mp4");

	free(image);
	return 0;
}

