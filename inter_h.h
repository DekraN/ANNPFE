enum
{
	ANN_FLOAT,
	ANN_DOUBLE
};

#ifndef PROGRAM_PRECISION
	#define PROGRAM_PRECISION ANN_FLOAT
#endif

#if PROGRAM_PRECISION == ANN_FLOAT
	typedef float atyp;
#else
	typedef double atyp;
#endif
