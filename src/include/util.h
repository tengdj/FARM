#ifndef TENG_UTIL_H_
#define TENG_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <vector>
#include <thread>
#include <iostream>
#include <stdarg.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <time.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

using namespace std;

namespace{

#define TENG_RANDOM_NUMBER 0315
#define OSM_SRID 4326
const double PI = 3.14159265;

// some utility function

// round a number to a specified number of significant digits.
// for example:
// roundToSignificantDigits(12345.6789, 3) -> 12300
// roundToSignificantDigits(0.0045678, 2) -> 0.0046
inline double roundToSignificantDigits(double num, int n) {
    if (num == 0) {
        return 0;
    }
    
    double d = std::ceil(std::log10(num < 0 ? -num : num));
    int power = n - static_cast<int>(d);
    
    double magnitude = std::pow(10.0, power);
    double shifted = std::round(num * magnitude);
    return shifted / magnitude;
}

const double degree_per_kilometer_latitude = 360.0/40076.0;
const double degree_per_kilometer_longitude_arr[] = {
		0.008983,0.008983,0.008983,0.008983,0.008983,0.008983,0.008983,0.008984,0.008984,0.008984
		,0.008984,0.008985,0.008985,0.008985,0.008986,0.008986,0.008986,0.008987,0.008987,0.008988
		,0.008988,0.008989,0.008990,0.008990,0.008991,0.008991,0.008992,0.008993,0.008994,0.008994
		,0.008995,0.008996,0.008997,0.008998,0.008999,0.009000,0.009001,0.009002,0.009003,0.009004
		,0.009005,0.009006,0.009007,0.009008,0.009009,0.009011,0.009012,0.009013,0.009015,0.009016
		,0.009017,0.009019,0.009020,0.009022,0.009023,0.009024,0.009026,0.009028,0.009029,0.009031
		,0.009032,0.009034,0.009036,0.009038,0.009039,0.009041,0.009043,0.009045,0.009047,0.009048
		,0.009050,0.009052,0.009054,0.009056,0.009058,0.009060,0.009063,0.009065,0.009067,0.009069
		,0.009071,0.009073,0.009076,0.009078,0.009080,0.009083,0.009085,0.009087,0.009090,0.009092
		,0.009095,0.009097,0.009100,0.009103,0.009105,0.009108,0.009111,0.009113,0.009116,0.009119
		,0.009122,0.009124,0.009127,0.009130,0.009133,0.009136,0.009139,0.009142,0.009145,0.009148
		,0.009151,0.009154,0.009157,0.009161,0.009164,0.009167,0.009170,0.009174,0.009177,0.009180
		,0.009184,0.009187,0.009190,0.009194,0.009197,0.009201,0.009205,0.009208,0.009212,0.009216
		,0.009219,0.009223,0.009227,0.009231,0.009234,0.009238,0.009242,0.009246,0.009250,0.009254
		,0.009258,0.009262,0.009266,0.009270,0.009274,0.009278,0.009283,0.009287,0.009291,0.009295
		,0.009300,0.009304,0.009309,0.009313,0.009317,0.009322,0.009326,0.009331,0.009336,0.009340
		,0.009345,0.009350,0.009354,0.009359,0.009364,0.009369,0.009374,0.009378,0.009383,0.009388
		,0.009393,0.009398,0.009403,0.009409,0.009414,0.009419,0.009424,0.009429,0.009435,0.009440
		,0.009445,0.009451,0.009456,0.009461,0.009467,0.009472,0.009478,0.009484,0.009489,0.009495
		,0.009501,0.009506,0.009512,0.009518,0.009524,0.009530,0.009535,0.009541,0.009547,0.009553
		,0.009559,0.009566,0.009572,0.009578,0.009584,0.009590,0.009597,0.009603,0.009609,0.009616
		,0.009622,0.009628,0.009635,0.009642,0.009648,0.009655,0.009661,0.009668,0.009675,0.009682
		,0.009688,0.009695,0.009702,0.009709,0.009716,0.009723,0.009730,0.009737,0.009744,0.009751
		,0.009759,0.009766,0.009773,0.009781,0.009788,0.009795,0.009803,0.009810,0.009818,0.009825
		,0.009833,0.009841,0.009848,0.009856,0.009864,0.009872,0.009880,0.009888,0.009896,0.009904
		,0.009912,0.009920,0.009928,0.009936,0.009944,0.009952,0.009961,0.009969,0.009978,0.009986
		,0.009994,0.010003,0.010012,0.010020,0.010029,0.010038,0.010046,0.010055,0.010064,0.010073
		,0.010082,0.010091,0.010100,0.010109,0.010118,0.010127,0.010136,0.010146,0.010155,0.010164
		,0.010174,0.010183,0.010193,0.010202,0.010212,0.010222,0.010231,0.010241,0.010251,0.010261
		,0.010271,0.010281,0.010291,0.010301,0.010311,0.010321,0.010331,0.010341,0.010352,0.010362
		,0.010373,0.010383,0.010394,0.010404,0.010415,0.010426,0.010436,0.010447,0.010458,0.010469
		,0.010480,0.010491,0.010502,0.010513,0.010524,0.010535,0.010547,0.010558,0.010569,0.010581
		,0.010592,0.010604,0.010616,0.010627,0.010639,0.010651,0.010663,0.010675,0.010687,0.010699
		,0.010711,0.010723,0.010735,0.010748,0.010760,0.010772,0.010785,0.010797,0.010810,0.010823
		,0.010835,0.010848,0.010861,0.010874,0.010887,0.010900,0.010913,0.010926,0.010939,0.010953
		,0.010966,0.010980,0.010993,0.011007,0.011020,0.011034,0.011048,0.011062,0.011075,0.011089
		,0.011104,0.011118,0.011132,0.011146,0.011160,0.011175,0.011189,0.011204,0.011218,0.011233
		,0.011248,0.011263,0.011278,0.011293,0.011308,0.011323,0.011338,0.011353,0.011369,0.011384
		,0.011400,0.011415,0.011431,0.011446,0.011462,0.011478,0.011494,0.011510,0.011526,0.011543
		,0.011559,0.011575,0.011592,0.011608,0.011625,0.011642,0.011658,0.011675,0.011692,0.011709
		,0.011726,0.011744,0.011761,0.011778,0.011796,0.011813,0.011831,0.011849,0.011867,0.011884
		,0.011903,0.011921,0.011939,0.011957,0.011975,0.011994,0.012013,0.012031,0.012050,0.012069
		,0.012088,0.012107,0.012126,0.012145,0.012164,0.012184,0.012203,0.012223,0.012243,0.012263
		,0.012283,0.012303,0.012323,0.012343,0.012363,0.012384,0.012404,0.012425,0.012446,0.012467
		,0.012488,0.012509,0.012530,0.012551,0.012573,0.012594,0.012616,0.012638,0.012660,0.012682
		,0.012704,0.012726,0.012748,0.012771,0.012793,0.012816,0.012839,0.012862,0.012885,0.012908
		,0.012931,0.012955,0.012978,0.013002,0.013026,0.013050,0.013074,0.013098,0.013122,0.013147
		,0.013171,0.013196,0.013221,0.013246,0.013271,0.013296,0.013322,0.013347,0.013373,0.013399
		,0.013425,0.013451,0.013477,0.013503,0.013530,0.013557,0.013584,0.013610,0.013638,0.013665
		,0.013692,0.013720,0.013748,0.013775,0.013803,0.013832,0.013860,0.013888,0.013917,0.013946
		,0.013975,0.014004,0.014033,0.014063,0.014093,0.014122,0.014152,0.014183,0.014213,0.014243
		,0.014274,0.014305,0.014336,0.014367,0.014399,0.014430,0.014462,0.014494,0.014526,0.014558
		,0.014591,0.014623,0.014656,0.014689,0.014723,0.014756,0.014790,0.014824,0.014858,0.014892
		,0.014926,0.014961,0.014996,0.015031,0.015066,0.015102,0.015138,0.015174,0.015210,0.015246
		,0.015283,0.015320,0.015357,0.015394,0.015431,0.015469,0.015507,0.015545,0.015584,0.015622
		,0.015661,0.015700,0.015740,0.015779,0.015819,0.015860,0.015900,0.015941,0.015981,0.016023
		,0.016064,0.016106,0.016148,0.016190,0.016233,0.016275,0.016318,0.016362,0.016405,0.016449
		,0.016493,0.016538,0.016583,0.016628,0.016673,0.016719,0.016765,0.016811,0.016857,0.016904
		,0.016952,0.016999,0.017047,0.017095,0.017143,0.017192,0.017241,0.017291,0.017341,0.017391
		,0.017441,0.017492,0.017543,0.017595,0.017647,0.017699,0.017752,0.017805,0.017858,0.017912
		,0.017966,0.018020,0.018075,0.018131,0.018186,0.018242,0.018299,0.018356,0.018413,0.018471
		,0.018529,0.018587,0.018646,0.018706,0.018766,0.018826,0.018887,0.018948,0.019009,0.019072
		,0.019134,0.019197,0.019261,0.019325,0.019389,0.019454,0.019520,0.019586,0.019652,0.019719
		,0.019787,0.019855,0.019923,0.019992,0.020062,0.020132,0.020203,0.020274,0.020346,0.020419
		,0.020492,0.020565,0.020639,0.020714,0.020790,0.020866,0.020942,0.021020,0.021098,0.021176
		,0.021255,0.021335,0.021416,0.021497,0.021579,0.021662,0.021745,0.021829,0.021914,0.021999
		,0.022085,0.022172,0.022260,0.022349,0.022438,0.022528,0.022619,0.022710,0.022803,0.022896
		,0.022990,0.023085,0.023181,0.023278,0.023375,0.023474,0.023573,0.023673,0.023774,0.023877
		,0.023980,0.024084,0.024189,0.024295,0.024402,0.024510,0.024619,0.024729,0.024840,0.024953
		,0.025066,0.025181,0.025296,0.025413,0.025531,0.025650,0.025771,0.025892,0.026015,0.026139
		,0.026264,0.026391,0.026519,0.026648,0.026779,0.026911,0.027044,0.027179,0.027315,0.027452
		,0.027592,0.027732,0.027874,0.028018,0.028163,0.028310,0.028459,0.028609,0.028761,0.028914
		,0.029069,0.029226,0.029385,0.029546,0.029708,0.029873,0.030039,0.030207,0.030378,0.030550
		,0.030724,0.030901,0.031079,0.031260,0.031443,0.031628,0.031816,0.032006,0.032198,0.032393
		,0.032590,0.032789,0.032991,0.033196,0.033404,0.033614,0.033827,0.034043,0.034261,0.034483
		,0.034707,0.034935,0.035166,0.035400,0.035637,0.035877,0.036121,0.036368,0.036619,0.036873
		,0.037132,0.037393,0.037659,0.037929,0.038202,0.038480,0.038762,0.039048,0.039338,0.039633
		,0.039933,0.040237,0.040546,0.040860,0.041179,0.041503,0.041833,0.042167,0.042508,0.042854
		,0.043206,0.043563,0.043927,0.044297,0.044674,0.045057,0.045447,0.045844,0.046248,0.046659
		,0.047078,0.047505,0.047939,0.048382,0.048833,0.049293,0.049762,0.050239,0.050727,0.051224
		,0.051731,0.052248,0.052776,0.053315,0.053865,0.054426,0.055000,0.055586,0.056185,0.056797
		,0.057423,0.058063,0.058717,0.059387,0.060072,0.060774,0.061492,0.062228,0.062981,0.063753
		,0.064545,0.065357,0.066189,0.067044,0.067921,0.068821,0.069746,0.070696,0.071672,0.072677
		,0.073710,0.074773,0.075867,0.076994,0.078155,0.079352,0.080587,0.081861,0.083176,0.084534
		,0.085938,0.087389,0.088890,0.090445,0.092054,0.093723,0.095453,0.097249,0.099114,0.101052
		,0.103068,0.105166,0.107351,0.109630,0.112008,0.114492,0.117089,0.119806,0.122654,0.125640
		,0.128776,0.132072,0.135543,0.139201,0.143062,0.147144,0.151467,0.156051,0.160922,0.166108
		,0.171640,0.177553,0.183889,0.190694,0.198023,0.205939,0.214514,0.223836,0.234005,0.245143
		,0.257394,0.270936,0.285983,0.302800,0.321719,0.343162,0.367668,0.395945,0.428935,0.467923
		,0.514710,0.571895,0.643376,0.735281,0.857823,1.029381,1.286721,1.715622,2.573426,5.146844
};

inline double degree_per_kilometer_longitude_calculate(double latitude){
	double absla = abs(latitude);
	assert(absla<=90);
	if(absla==90){
		absla = 89;
	}
	return 360.0/(sin((90-absla)*PI/180)*40076);
}

CUDA_HOSTDEV inline double degree_per_kilometer_longitude(double latitude){
	double absla = abs(latitude);
	assert(absla<=90);
	if(absla==90){
		absla = 89.9;
	}
	return degree_per_kilometer_longitude_arr[(int)(absla*10)];
}

inline int double_to_int(double val){
	int vi = (int)val;
	if(abs(1.0*(vi+1)-val)<0.00000001){
		vi++;
	}
	return vi;
}

inline bool is_number(char ch){
	return ch=='-'||ch=='.'||(ch<='9'&&ch>='0')||ch=='e';
}

inline double read_double(const char *input, size_t &offset){
	char tmp[100];
	while(!is_number(input[offset])){
		offset++;
	}
	int index = 0;
	while(is_number(input[offset])){
		tmp[index++] = input[offset++];
	}
	tmp[index] = '\0';
	return atof(tmp);
}
inline void skip_space(const char *input, size_t &offset){
	while(input[offset]==' '||input[offset]=='\t'||input[offset]=='\n'){
		offset++;
	}
}

inline struct timeval get_cur_time(){
	struct timeval t1;
	gettimeofday(&t1, NULL);
	return t1;
}
inline double get_time_elapsed(struct timeval &t1, bool update_start = false){
	struct timeval t2;
    double elapsedTime;
	gettimeofday(&t2, NULL);
	// compute and print the elapsed time in millisec
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	if(update_start){
		t1 = get_cur_time();
	}
	return elapsedTime;
}

inline string time_string(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	struct tm *nowtm;
	char tmbuf[100];
	char buf[256];
	nowtm = localtime(&tv.tv_sec);
	strftime(tmbuf, sizeof tmbuf, "%H:%M:%S", nowtm);
	sprintf(buf,"%s.%04ld", tmbuf, tv.tv_usec/1000);
	return string(buf);
}

static pthread_mutex_t general_lock;
inline void lock(){
	pthread_mutex_lock(&general_lock);
}

inline void unlock(){
	pthread_mutex_unlock(&general_lock);
}

static pthread_mutex_t print_lock;
inline double logt(const char *format, struct timeval &start, ...){
	pthread_mutex_lock(&print_lock);
	fprintf(stderr, "\r                                                                              ");
	va_list args;
	va_start(args, start);
	char sprint_buf[200];
	int n = vsprintf(sprint_buf, format, args);
	va_end(args);
	fprintf(stderr,"\r%s thread %ld:\t%s", time_string().c_str(), syscall(__NR_gettid),sprint_buf);

	double mstime = get_time_elapsed(start, true);
	if(mstime>1000){
		fprintf(stderr," takes %f s\n", mstime/1000);
	}else{
		fprintf(stderr," takes %f ms\n", mstime);
	}
	fflush(stderr);

	pthread_mutex_unlock(&print_lock);

	return mstime;
}

inline void log(const char *format, ...){
	pthread_mutex_lock(&print_lock);
	va_list args;
	va_start(args, format);
	char sprint_buf[200];
	int n = vsprintf(sprint_buf, format, args);
	va_end(args);
	fprintf(stderr,"\r%s thread %ld:\t%s\n", time_string().c_str(), syscall(__NR_gettid),sprint_buf);
	fflush(stderr);
	pthread_mutex_unlock(&print_lock);
}

inline double logt_refresh(const char *format, struct timeval &start, ...){
	pthread_mutex_lock(&print_lock);
	va_list args;
	va_start(args, start);
	char sprint_buf[200];
	int n = vsprintf(sprint_buf, format, args);
	va_end(args);
	fprintf(stderr,"\r%s thread %ld:\t%s", time_string().c_str(), syscall(__NR_gettid),sprint_buf);

	double mstime = get_time_elapsed(start, true);
	if(mstime>1000){
		fprintf(stderr," takes %f s", mstime/1000);
	}else{
		fprintf(stderr," takes %f ms", mstime);
	}
	fflush(stderr);

	pthread_mutex_unlock(&print_lock);
	return mstime;
}
inline void log_refresh(const char *format, ...){
	pthread_mutex_lock(&print_lock);
	va_list args;
	va_start(args, format);
	char sprint_buf[200];
	int n = vsprintf(sprint_buf, format, args);
	va_end(args);
	fprintf(stderr,"\r%s thread %ld:\t%s", time_string().c_str(), syscall(__NR_gettid),sprint_buf);
	fflush(stderr);
	pthread_mutex_unlock(&print_lock);
}

inline void log_stdout(const char *format, ...){
	pthread_mutex_lock(&print_lock);
	va_list args;
	va_start(args, format);
	char sprint_buf[200];
	int n = vsprintf(sprint_buf, format, args);
	va_end(args);
	fprintf(stdout,"\r%s thread %ld:\t%s\n", time_string().c_str(), syscall(__NR_gettid),sprint_buf);
	fflush(stdout);
	pthread_mutex_unlock(&print_lock);
}

inline void log(){
	pthread_mutex_lock(&print_lock);
	fprintf(stdout,"\r%s thread %ld:\tterry is good\n", time_string().c_str(),syscall(__NR_gettid));
	fflush(stdout);
	pthread_mutex_unlock(&print_lock);
}

inline int get_rand_number(int max_value){
	return rand()%max_value+1;
}

inline double get_rand_double(){
	return rand()/(double)RAND_MAX;
}

inline bool get_rand_sample(int rate){
	return rand()%100<rate;
}

inline bool tryluck(float possibility){
	assert(possibility>=0);
	return possibility>=1.0||(rand()*1.0)/RAND_MAX<possibility;
}

inline bool is_dir(const char* path) {
    struct stat buf;
    stat(path, &buf);
    return S_ISDIR(buf.st_mode);
}

inline bool is_file(const char* path) {
    struct stat buf;
    stat(path, &buf);
    return S_ISREG(buf.st_mode);
}

inline void list_files(const char *path, std::vector<string> &f_list){
	if(is_file(path)){
		f_list.push_back(std::string(path));
		return;
	}
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (path)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			if(strcmp(ent->d_name,"..")==0||
					strcmp(ent->d_name,".")==0){
				continue;
			}
			std::stringstream ss;
			ss<<path<<"/"<<ent->d_name;
			string spath;
			ss>>spath;
			list_files(spath.c_str(), f_list);
		}
		closedir (dir);
	}
}

inline bool write_file(const char *path, char *content, size_t data_size){
	ofstream os;
	os.open(path, ios::out | ios::binary |ios::trunc);
	assert(os.is_open());
	os.write(content, data_size);
	os.close();
	return true;
}

inline long file_size(const char *file){
	struct stat stat_buf;
	int rc = stat(file, &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

inline long file_size(std::vector<string> &f_list){
	long size = 0;
	for(string s:f_list){
		long ls = file_size(s.c_str());
		if(ls>0){
			size += ls;
		}
	}
	return size;
}

inline bool file_exist(const char *path) {
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

inline int get_num_threads(){
	return std::thread::hardware_concurrency();
}

inline string read_line(){
	string input_line;
	getline(std::cin, input_line);
	return input_line;
}


inline void tokenize( const std::string& str, std::vector<std::string>& result,
	const std::string& delimiters = " ,;:\t",
	const bool keepBlankFields=false,
	const std::string& quote="\"\'"
	){
    // clear the vector
    if (!result.empty()){
    	result.clear();
    }

    // you must be kidding
    if (delimiters.empty())
	return ;

    std::string::size_type pos = 0; // the current position (char) in the string
    char ch = 0; // buffer for the current character

    char current_quote = 0; // the char of the current open quote
    bool quoted = false; // indicator if there is an open quote
    std::string token;  // string buffer for the token
    bool token_complete = false; // indicates if the current token is
    // read to be added to the result vector
    std::string::size_type len = str.length();  // length of the input-string

    // for every char in the input-string
	while(len > pos){
		// get the character of the string and reset the delimiter buffer
		ch = str.at(pos);

		bool add_char = true;
		if ( false == quote.empty()){
			// if quote chars are provided and the char isn't protected
			if (std::string::npos != quote.find_first_of(ch)){
				if (!quoted){
					quoted = true;
					current_quote = ch;
					add_char = false;
				} else {
					if (current_quote == ch){
						quoted = false;
						current_quote = 0;
						add_char = false;
					}
				}
			}
		}

		if (!delimiters.empty()&&!quoted){
			// if ch is delemiter
			if (std::string::npos != delimiters.find_first_of(ch)){
				token_complete = true;
				// don't add the delimiter to the token
				add_char = false;
			}
		}

		// add the character to the token
		if (add_char){
			token.push_back(ch);
		}

		// add the token if it is complete
		// if ( true == token_complete && false == token.empty() )
		if (token_complete){
			if (token.empty())
			{
			if (keepBlankFields)
				result.push_back("");
			}
			else
			result.push_back( token );
			token.clear();
			token_complete = false;
		}
		++pos;
    } // while
    // add the final token
    if ( false == token.empty() ) {
    	result.push_back( token );
    } else if(keepBlankFields && std::string::npos != delimiters.find_first_of(ch) ){
    	result.push_back("");
    }
}


inline void remove_slash(string &str){
	if(str.at(str.size() - 1) == '/'){
		str = str.substr(0, str.size() - 1);
	}
}

#define min_equal 1e-12
inline bool double_equal(double d1, double d2){
	return fabs(d1-d2)<min_equal;
}

inline bool double_zero(double d, double epsilon = min_equal){
	return fabs(d)<epsilon;
}

template<class T>
void parallel_sort(T* data, size_t len, size_t grainsize, bool (*comp)(T, T)){
    if(len <= grainsize){
        std::sort(data, data + len, comp);
    } else {
        auto future = std::async(parallel_sort<T>, data, len/2, grainsize, comp);
        parallel_sort(data + len/2, len - len/2, grainsize, comp);
        future.wait();
        std::inplace_merge(data, data + len/2, data + len, comp);
    }
}

// generate y = a*x+b
inline void linear_regression(vector<double> &X, vector<double> &Y, double &a, double &b){

	double N = X.size();

	assert(X.size()==Y.size()&&X.size()>0);

	double sum_x = 0.0;
	double sum_y = 0.0;
	double sum_xy = 0.0;
	double sum_xx = 0.0;
	double sum_yy = 0.0;

	for (size_t i = 0; i < X.size(); i++) {
		double xi = X[i];
		double yi = Y[i];
		sum_x += xi;
		sum_y += yi;
		sum_xx += xi * xi;
		sum_yy += yi * yi;
		sum_xy += xi * yi;
	}


	a = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x * sum_x);
	b = (sum_y * sum_xx - sum_x * sum_xy)/(N * sum_xx - sum_x * sum_x);
}

}
#endif
