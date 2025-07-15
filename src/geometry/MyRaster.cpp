#include "MyRaster.h"
#include "UniversalGrid.h"

MyRaster::~MyRaster(){
	if(status) delete []status;
}

void MyRaster::init_raster(int num_pixels){
    assert(num_pixels >= 0);

	double multi = abs((mbr->high[1]-mbr->low[1])/(mbr->high[0]-mbr->low[0]));
	dimx = std::pow(num_pixels/multi,0.5);
	dimy = dimx*multi;

	if(dimx==0){
		dimx = 1;
	}
	if(dimy==0){
		dimy = 1;
	}

	step_x = (mbr->high[0]-mbr->low[0])/dimx;
	step_y = (mbr->high[1]-mbr->low[1])/dimy;

	if(step_x<0.00001){
		step_x = 0.00001;
		dimx = (mbr->high[0]-mbr->low[0])/step_x+1;
	}
	if(step_y<0.00001){
		step_y = 0.00001;
		dimy = (mbr->high[1]-mbr->low[1])/step_y+1;
	}
}

void MyRaster::init_raster(int dx, int dy){
	dimx = dx;
	dimy = dy;

	step_x = (mbr->high[0]-mbr->low[0])/dimx;
	step_y = (mbr->high[1]-mbr->low[1])/dimy;

	if(step_x<0.00001){
		step_x = 0.00001;
		dimx = (mbr->high[0]-mbr->low[0])/step_x+1;
	}
	if(step_y<0.00001){
		step_y = 0.00001;
		dimy = (mbr->high[1]-mbr->low[1])/step_y+1;
	}
}

void MyRaster::print(){
	MyMultiPolygon *inpolys = new MyMultiPolygon();
	MyMultiPolygon *borderpolys = new MyMultiPolygon();
	MyMultiPolygon *outpolys = new MyMultiPolygon();

	for(int i=0;i<dimx;i++){
		for(int j=0;j<dimy;j++){
			box bx = get_pixel_box(i, j);
			MyPolygon *m = MyPolygon::gen_box(bx);
			if(show_status(get_id(i, j)) == BORDER){
				borderpolys->insert_polygon(m);
			}else if(show_status(get_id(i, j)) == IN){
				inpolys->insert_polygon(m);
			}else if(show_status(get_id(i, j)) == OUT){
				outpolys->insert_polygon(m);
			}
		}
	}

	cout<<"border:" << borderpolys->num_polygons() <<endl;
	borderpolys->print();
	cout<<"in:"<< inpolys->num_polygons() << endl;
	inpolys->print();
	cout<<"out:"<< outpolys->num_polygons() << endl;
	outpolys->print();
	cout << endl;


	delete borderpolys;
	delete inpolys;
	delete outpolys;
}

box *MyRaster::extractMER(int starter){
	assert(show_status(starter) == IN);
	int cx = get_x(starter);
	int cy = get_y(starter);
	box *curmer = new box();
	int shift[4] = {0,0,0,0};
	bool limit[4] = {false,false,false,false};
	int index = 0;
	while(!limit[0]||!limit[1]||!limit[2]||!limit[3]){
		//left
		if(!limit[0]){
			shift[0]++;
			if(cx-shift[0]<0){
				limit[0] = true;
				shift[0]--;
			}else{
				for(int i=cy-shift[1];i<=cy+shift[3];i++){
					//log("%d %d %d %d", cx-shift[0], dimx, i, dimy);
					if(show_status(get_id(cx-shift[0], i)) != IN){
						limit[0] = true;
						shift[0]--;
						break;
					}
				}
			}
		}
		//bottom
		if(!limit[1]){
			shift[1]++;
			if(cy-shift[1]<0){
				limit[1] = true;
				shift[1]--;
			}else{
				for(int i=cx-shift[0];i<=cx+shift[2];i++){
					if(show_status(get_id(i, cy-shift[1])) != IN){
						limit[1] = true;
						shift[1]--;
						break;
					}
				}
			}
		}
		//right
		if(!limit[2]){
			shift[2]++;
			if(cx+shift[2]>=(dimx+1)){
				limit[2] = true;
				shift[2]--;
			}else{
				for(int i=cy-shift[1];i<=cy+shift[3];i++){
					if(show_status(get_id(cx+shift[2], i)) != IN){
						limit[2] = true;
						shift[2]--;
						break;
					}
				}
			}
		}
		//top
		if(!limit[3]){
			shift[3]++;
			if(cy+shift[3]>=(dimy+1)){
				limit[3] = true;
				shift[3]--;
			}else{
				for(int i=cx-shift[0];i<=cx+shift[2];i++){
					// log("%d %d %d %d", i, dimx, cy+shift[3], dimy);
					if(show_status(get_id(i, cy+shift[3])) != IN){
						limit[3] = true;
						shift[3]--;
						break;
					}
				}
			}
		}
	}

	curmer->low[0] = get_pixel_box(cx-shift[0], cy-shift[1]).low[0];
	curmer->low[1] = get_pixel_box(cx-shift[0], cy-shift[1]).low[1];
	curmer->high[0] = get_pixel_box(cx+shift[2], cy+shift[3]).high[0];
	curmer->high[1] = get_pixel_box(cx+shift[2], cy+shift[3]).high[1];

	return curmer;
}

int MyRaster::get_id(int x, int y){
	assert(x>=0&&x<dimx);
	assert(y>=0&&y<dimy);
	return y * dimx + x;
}

// from id to pixel x
int MyRaster::get_x(int id){
	return id % dimx;
}

// from id to pixel y
int MyRaster::get_y(int id){
	assert((id / dimx) < dimy);
	return id / dimx;
}

// the range must be [0, dimx]
int MyRaster::get_offset_x(double xval){
	assert(mbr);
	assert(step_x>0.000000000001 && "the width per pixel must not be 0");
	int x = double_to_int((xval-mbr->low[0])/step_x);
	return min(max(x, 0), dimx - 1);
}
// the range must be [0, dimy]
int MyRaster::get_offset_y(double yval){
	assert(mbr);
	assert(step_y>0.000000000001 && "the hight per pixel must not be 0");
	int y = double_to_int((yval-mbr->low[1])/step_y);
	return min(max(y, 0), dimy - 1);
}

// void MyRaster::set_status(int id, PartitionStatus state){
// 	int pos = id % 4 * 2;   // The multiplication by 2 is because each status occupies 2 bits.
// 	if(state == OUT){
// 		status[id / 4] &= ~((uint8_t)3 << pos);
// 	}else if(state == IN){
// 		status[id / 4] |= ((uint8_t)3 << pos);
// 	}else{
// 		status[id / 4] &= ~((uint8_t)1 << pos);
// 		status[id / 4] |= ((uint8_t)1 << (pos + 1));
// 	}
// }

void MyRaster::set_status(int id, uint8_t state){   // The multiplication by 2 is because each status occupies 2 bits.
	status[id] = state;
}

// PartitionStatus MyRaster::show_status(int id){
// 	uint8_t st = status[id / 4];
// 	int pos = id % 4 * 2;   // The multiplication by 2 is because each status occupies 2 bits.	
// 	st &= ((uint8_t)3 << pos);
// 	st >>= pos;
// 	if(st == 0) return OUT;
// 	if(st == 3) return IN;
// 	return BORDER;
// }

PartitionStatus MyRaster::show_status(int id){
	if (status[id] == 0)
		return OUT;
	else if (status[id] == category_count - 1)
		return IN;
	else
		return BORDER;
}

void MyRaster::set_status_size()
{
	status_size = dimx * dimy;
}

vector<int> MyRaster::get_intersect_pixels(box *b){

	// test all the pixels
	int txstart = get_offset_x(b->low[0]);
	int tystart = get_offset_y(b->low[1]);

	double width_d = (b->high[0]-b->low[0]+step_x*0.9999999)/step_x;
	int width = double_to_int(width_d);

	double height_d = (b->high[1]-b->low[1]+step_y*0.9999999)/step_y;
	int height = double_to_int(height_d);

	vector<int> ret;
	for(int i=txstart;i<txstart+width;i++){
		for(int j=tystart;j<tystart+height;j++){
			ret.push_back(get_id(i, j));
		}
	}
	return ret;
}

vector<int> MyRaster::get_closest_pixels(box &target){

	// note that at 0 or dimx/dimy will be returned if
	// the range of target is beyound this, as expected
	int txstart = get_offset_x(target.low[0]);
	int txend = get_offset_x(target.high[0]);
	int tystart = get_offset_y(target.low[1]);
	int tyend = get_offset_y(target.high[1]);
	vector<int> ret;
	for(int i=txstart;i<=txend;i++){
		for(int j=tystart;j<=tyend;j++){
			ret.push_back(get_id(i, j));
		}
	}
	return ret;
}

int MyRaster::get_closest_pixel(Point &p){
	int pixx = get_offset_x(p.x);
	int pixy = get_offset_y(p.y);
	if(pixx < 0){
		pixx = 0;
	}
	if(pixx > dimx){
		pixx = dimx;
	}
	if(pixy < 0){
		pixy = 0;
	}
	if(pixy > dimy){
		pixy = dimy;
	}
	return get_id(pixx, pixy);
}

vector<int> MyRaster::get_pixels(PartitionStatus status){
	vector<int> ret;
	for(int id = 0; id < get_num_pixels(); id ++){
		if(show_status(id) == status){
			ret.push_back(id);
		}
	}
	return ret;
}

box MyRaster::get_pixel_box(int x, int y){
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	double lowx = start_x + x * step_x;
	double lowy = start_y + y * step_y;
	double highx = start_x + (x + 1) * step_x;
	double highy = start_y + (y + 1) * step_y;

	return box(lowx, lowy, highx, highy);
}

int MyRaster::get_pixel_id(Point &p){
	int xoff = get_offset_x(p.x);
	int yoff = get_offset_y(p.y);
	assert(xoff < dimx);
	assert(yoff < dimy);
	return get_id(xoff, yoff);
}

vector<int> MyRaster::retrieve_pixels(box *target){
	vector<int> ret;
	int start_x = get_offset_x(target->low[0] + 1e-6);
	int start_y = get_offset_y(target->low[1] + 1e-6);
	int end_x = get_offset_x(target->high[0] - 1e-6);
	int end_y = get_offset_y(target->high[1] - 1e-6);

	//log("%d %d %d %d %d %d",dimx,dimy,start_x,end_x,start_y,end_y);
	for(int i=start_x;i<=end_x;i++){
		for(int j=start_y;j<=end_y;j++){
			ret.push_back(get_id(i , j));
		}
	}
	return ret;
}

bool MyRaster::contain(box *b, bool &contained){
	if(!mbr->contain(*b)){
		contained = false;
		return true;
	}
	// test all the pixels that intersects b
	vector<int> covered = get_intersect_pixels(b);

	int incount = 0;
	int outcount = 0;
	for(auto pix : covered){
		if(show_status(pix) == OUT){
			outcount++;
		}
		if(show_status(pix) == IN){
			incount++;
		}
	}
	int total = covered.size();
	covered.clear();
	// all in/out
	if(incount==total){
		contained = true;
		return true;
	}else if(outcount==total){
		contained = false;
		return true;
	}

	// not determined by checking pixels only
	return false;
}

vector<int> MyRaster::expand_radius(int core_x_low, int core_x_high, int core_y_low, int core_y_high, int step){

	vector<int> needprocess;
	int ymin = max(0,core_y_low-step);
	int ymax = min(dimy,core_y_high+step);

	//left scan
	if(core_x_low-step>=0){
		for(int y=ymin;y<=ymax;y++){
			needprocess.push_back(get_id(core_x_low-step,y));
		}
	}
	//right scan
	if(core_x_high+step<=get_dimx()){
		for(int y=ymin;y<=ymax;y++){
			needprocess.push_back(get_id(core_x_high+step,y));
		}
	}

	// skip the first if there is left scan
	int xmin = max(0,core_x_low-step+(core_x_low-step>=0));
	// skip the last if there is right scan
	int xmax = min(dimx,core_x_high+step-(core_x_high+step<=dimx));
	//bottom scan
	if(core_y_low-step>=0){
		for(int x=xmin;x<=xmax;x++){
			needprocess.push_back(get_id(x,core_y_low-step));
		}
	}
	//top scan
	if(core_y_high+step<=get_dimy()){
		for(int x=xmin;x<=xmax;x++){
			needprocess.push_back(get_id(x,core_y_high+step));
		}
	}

	return needprocess;
}

vector<int> MyRaster::expand_radius(int center, int step){

	int lowx = get_x(center);
	int highx = get_x(center);
	int lowy = get_y(center);
	int highy = get_y(center);

	return expand_radius(lowx,highx,lowy,highy,step);
}

size_t MyRaster::get_num_pixels(){
	return dimx * dimy;
}

size_t MyRaster::get_num_pixels(PartitionStatus status){
	size_t num = 0;
	for(int i = 0; i < get_num_pixels();i ++){
		if(show_status(i) == status){
			num++;
		}
	}
	return num;	
}

void MyRaster::grid_align(){
	UniversalGrid &space = UniversalGrid::getInstance();

	bool flag1 = false, flag2 = false;
	auto x = space.get_step_x();
	auto y = space.get_step_y();
	if (step_x <= x)
		step_x = x, flag1 = true;
	if (step_y <= y)
		step_y = y, flag2 = true;

	for (int i = space.get_max_layers(); i > 0; i--){
		if (!flag1 && step_x > x && step_x < x * 2){
			flag1 = true;
			step_x = abs(step_x - x) < abs(step_x - x * 2) ? x : x * 2;
		}
		if (!flag2 && step_y > y && step_y < y * 2){
			flag2 = true;
			step_y = abs(step_y - y) < abs(step_y - y * 2) ? y : y * 2;
		}
		if (flag1 && flag2)
			break;
		x *= 2;
		y *= 2;
	}
	assert(flag1 && flag2);

	step_x = min(step_x, step_y);
	step_y = step_x;

	mbr->low[0] = floor(mbr->low[0] / step_x) * step_x;
	mbr->low[1] = floor(mbr->low[1] / step_y) * step_y;
	mbr->high[0] = ceil(mbr->high[0] / step_x) * step_x;
	mbr->high[1] = ceil(mbr->high[1] / step_y) * step_y;

	dimx = static_cast<int>(round((mbr->high[0] - mbr->low[0]) / step_x));
	dimy = static_cast<int>(round((mbr->high[1] - mbr->low[1]) / step_y));


	// printf("dimx = %d, dimy = %d\n", dimx, dimy);
	// printf("step_x = %lf, step_y = %lf\n", step_x, step_y);
	// printf("MBR:\n");
	// printf("POLYGON((%lf %lf, %lf %lf, %lf %lf, %lf %lf, %lf %lf))\n", mbr->low[0], mbr->low[1], mbr->low[0], mbr->high[1], mbr->high[0], mbr->high[1], mbr->high[0], mbr->low[1], mbr->low[0], mbr->low[1]);
}