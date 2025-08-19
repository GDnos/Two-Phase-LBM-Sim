/*
Raylib example file.
This is an example main file for a simple raylib project.
Use this as a starting point or replace it with your code.

by Jeffery Myers is marked with CC0 1.0. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/

*/

#include "raylib.h"
#include "raymath.h"

#include "resource_dir.h"	// utility header for SearchAndSetResourceDir

#include "math.h"




#define gridsize 150
#define n 2

struct Field2D {
	double field[gridsize][gridsize][9];
};

struct Field2D fieldo;

struct Field2D next_field;

Color HSVtoRGB(float h, float s, float v) {
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

    return (Color){ R, G, B, 255 };
}

// Angle in radians, returns rainbow color wheel smoothly wrapping
Color AngleToRainbow(double angle) {
    // Normalize angle into [0, 2π)
    while (angle < 0) angle += 2*PI;
    while (angle >= 2*PI) angle -= 2*PI;

    // Map [0, 2π) → [0, 360)
    float hue = (float)(angle * 180.0 / PI);
    return HSVtoRGB(hue, 1.0f, 1.0f);
}

double log_scale(double input) {
    if (input >= 1.0) return 1.0;
    if (input <= 0.0) return 0.0;

    double log_val = log10(input);  // log10 of input
    double scaled = (log_val + 9.0) / 9.0;  // map -9.0 → 0..1
    return scaled;
}



Color GetRainbowColor(float t, double tr) {
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

    return CLITERAL(Color){ r, g, b, a };
}

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


double velocityField[gridsize][gridsize][2];
double DensityField[gridsize][gridsize];
int SolidField[gridsize][gridsize];
double SolidVelocityField[gridsize][gridsize][2];
#define SpeedOfSound (1.0/1.7320508075688772)
#define TimeRelaxationConstant 0.9
#define defaultSpeedX 0.0
#define defaultSpeedY 0.0

double getMomentum(struct Field2D exfield, int x, int y) {
	double xmom = 0;
	double ymom = 0;
	xmom = velocityField[x][y][0] * DensityField[x][y];
	ymom = velocityField[x][y][1] * DensityField[x][y];
	return pow(pow(xmom,2) + pow(ymom,2), 0.5);
}



void MakeSolidDisc(double r, double x0, double y0) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( pow((pow((double)x-x0, 2) + pow((double)y-y0, 2)), 0.5) < r) {
				SolidField[x][y] = 1;
			}
		}
	}
}

void MakeLiquidDisc(double r, double x0, double y0) {
	for (int x=0; x < gridsize; x++) {
		for (int y=0; y < gridsize; y++) {
			if( pow((pow((double)x-x0, 2) + pow((double)y-y0, 2)), 0.5) < r) {
				SolidField[x][y] = 0;
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

int main ()
{
	#define WindowSize 1200
	InitWindow(WindowSize,WindowSize,"basic window");
	SetTargetFPS(24);

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

	MakeSolidDisc(10,75,75);
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
	int i,j,k;

	
	int pos = 10;
	
	int rhythm = 0;

	while (!WindowShouldClose())
	{

		BeginDrawing();
		ClearBackground(BLACK);

		// rhythm += 1;
		// rhythm = rhythm % 5;
		// if(rhythm == 0) {
		// 	pos += 1;
		// 	pos = pos % gridsize;
		// }

		rhythm += 1;
		
		double freq = 10.0;
		double amp = 0.4;
		// double FlowVelocity[2] = {(amp/pow(freq,2))*sin(((double)rhythm)/freq), 0.0};

		// double BodyForce[2] = {(amp/pow(freq,2))*sin(((double)rhythm)/freq), 0.0};

		double BodyForce[2] = {0.000,0.000};

		struct Field2D df;

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
						// tx = (tx+gridsize) % gridsize;
						// ty = (ty+gridsize) % gridsize;
						// fieldo.field[tx][ty][v] = df.field[x][y][v];
						// neighbor outside → bounce back into opposite direction
						// fieldo.field[x][y][opposite[v]] = df.field[x][y][v];

						if(ty==-1 || ty == gridsize) {
							fieldo.field[x][y][opposite[v]] = df.field[x][y][v];
						}
						else {
							tx = (tx+gridsize) % gridsize;
							ty = (ty+gridsize) % gridsize;
							fieldo.field[tx][ty][v] = df.field[x][y][v];
						}

					}
				}
			}
		}

		for(x=0;x<gridsize;x++) {
			for(y=0;y<gridsize;y++) {
				double sum = 0;
				int v;
				for (v=0; v < 9; v++) {
					sum += fieldo.field[x][y][v];
				}
				
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
				// if (DensityField[x][y] < 0.5) {
				// 	DensityField[x][y] = 0.5;
				// } 
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
		// DrawLine(100,100,120,120,BLUE);

		

		for (x=0; x < gridsize; x++) {
			for (y=0; y < gridsize; y++) {
				// if (x <= 1 || x >= gridsize - 2 || y <= 1 || y >= gridsize - 2) {
				// 	DensityField[x][y]=1.0;
				// }
				Color clr;
				if (SolidField[x][y] == 1) {
					clr = WHITE;
				}
				else {

					clr = GetRainbowColor(5*getMomentum(fieldo,x,y)/SpeedOfSound, 1);
					// clr = AngleToRainbow(atan3(velocityField[x][y][0],velocityField[x][y][1]));
					// clr = GetRainbowColor(DensityField[x][y]/4, 1);
				}
				DrawRectangle((int)(((double)x/(double)gridsize)*1200), (int)(((double)y/(double)gridsize)*1200), (int)(1200/(double)gridsize) + 1, (int)(1200/(double)gridsize) + 1, clr);
			}
		}

		EndDrawing();
	}
	CloseWindow();
	return 0;
}
