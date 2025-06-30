#include <Hraster.h>

void Hraster::init(double _step_x, double _step_y, int _dimx, int _dimy, box *_mbr, bool last_layer){
    step_x = _step_x;
    step_y = _step_y;
    dimx = _dimx;
    dimy = _dimy;
    
    if(last_layer){
        mbr = _mbr;
    }else{
		mbr = new box(_mbr);

		mbr->low[0] = floor(mbr->low[0] / step_x) * step_x;
		mbr->low[1] = floor(mbr->low[1] / step_y) * step_y;
		mbr->high[0] = ceil(mbr->high[0] / step_x) * step_x;
		mbr->high[1] = ceil(mbr->high[1] / step_y) * step_y;

		dimx = static_cast<int>(round((mbr->high[0] - mbr->low[0]) / step_x));
		dimy = static_cast<int>(round((mbr->high[1] - mbr->low[1]) / step_y));
        
		status = new uint8_t[dimx * dimy];
        memset(status, 0, dimx * dimy * sizeof(uint8_t));
    }
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