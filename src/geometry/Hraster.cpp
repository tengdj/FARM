#include <Hraster.h>

void Hraster::init(double _step_x, double _step_y, int _dimx, int _dimy, box *_mbr, bool last_layer){
    step_x = _step_x;
    step_y = _step_y;
    dimx = _dimx;
    dimy = _dimy;
    
    if(last_layer){
        mbr = _mbr;
    }else{
        mbr = new box(_mbr->low[0], _mbr->low[1], _mbr->low[0] + step_x * dimx, _mbr->low[1] + step_y * dimy);
        status = new uint8_t[(dimx+1)*(dimy+1) / 4 + 1];
        memset(status, 0, ((dimx+1)*(dimy+1) / 4 + 1) * sizeof(uint8_t));
    }
}

PartitionStatus Hraster::merge_status(box target){
	int start_x = get_offset_x(target.low[0]);
	int start_y = get_offset_y(target.low[1]);
	int end_x = get_offset_x(target.high[0]) - 1;
	int end_y = get_offset_y(target.high[1]) - 1;

	assert(start_x >= 0 && start_y >=0 && end_x < dimx && end_y < dimy);

	if(end_y < start_y) end_y = start_y;
	if(end_x < start_x) end_x = start_x;

	bool etn = true, itn = true;

	for(int i=start_x;i<=end_x;i++){
		for(int j=start_y;j<=end_y;j++){
			int id = get_id(i ,j);
			if(show_status(id) == OUT) itn = false;
			if(show_status(id) == IN) etn = false;
			if(show_status(id) == BORDER) itn = false, etn = false;
		}
	}

	if(etn)	return OUT;
	if(itn) return IN;
	return BORDER;	
}

void Hraster::merge(Hraster &r){
	for(int x = 0; x <= dimx; x ++){
		for(int y = 0; y <= dimy; y ++){
			PartitionStatus st = r.merge_status(get_pixel_box(x, y));
			MyRaster::set_status(get_id(x, y), st); 
		}
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