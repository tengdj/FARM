#include <Hraster.h>

Hraster::~Hraster()
{
	if (!owns_status)
	{
		status = nullptr;
	}
	if (owns_areas && areas)
	{
		delete[] areas;
	}
	if (owns_mbr && mbr)
	{
		delete mbr;
	}
}

void Hraster::init(double _step_x, double _step_y, int &_dimx, int &_dimy, box *_mbr, bool is_base_layer){
	step_x = _step_x;
	step_y = _step_y;
	dimx = _dimx;
	dimy = _dimy;
	status = nullptr;
	areas = nullptr;
	owns_status = false;
	owns_areas = false;
	owns_mbr = false;

	if(is_base_layer){
		mbr = _mbr;
	}else{
		mbr = new box(_mbr);
		owns_mbr = true;

		status_size = static_cast<uint>(packed_status_bytes(
			static_cast<size_t>(dimx) * dimy, bitwidth));
		status = new uint8_t[status_size];
		memset(status, 0, status_size * sizeof(uint8_t));
		areas = new double[dimx * dimy]();
		owns_status = true;
		owns_areas = true;
	}

	status_size = static_cast<uint>(packed_status_bytes(
		static_cast<size_t>(dimx) * dimy, bitwidth));

	_dimx = dimx;
	_dimy = dimy;
}

void Hraster::attach_base_storage(uint8_t *_status, double *_areas)
{
	status = _status;
	areas = _areas;
	owns_status = false;
	owns_areas = false;
	status_size = static_cast<uint>(packed_status_bytes(
		static_cast<size_t>(dimx) * dimy, bitwidth));
}

void Hraster::print(){
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
	// allpolys->print();


	delete borderpolys;
	delete inpolys;
	delete outpolys;
}
