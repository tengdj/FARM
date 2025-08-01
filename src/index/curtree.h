#pragma once

struct BBOX
{
	int sz;
	double* xmin;
	double* ymin;
	double* xmax;
	double* ymax;
	int *id;
};

struct RTREE
{
	int sz;
	int fanout;
	double* xmin;
	double* ymin;
	double* xmax;
	double* ymax;
	int *pos;
	int *len;
};

struct IDPAIR
{
	int sz;
	int *fid;
	int *tid;
};

