#include "../include/Ideal.h"

Ideal::~Ideal(){
	if(offset) delete []offset;
	if(edge_sequences) delete []edge_sequences;
	if(vertical) delete vertical;
	if(horizontal) delete horizontal;
}

void Ideal::add_edge(int idx, int start, int end){
	edge_sequences[idx] = make_pair(start, end - start  + 1);
}

uint32_t Ideal::get_num_sequences(int id){
	if(show_status(id) != BORDER) return 0;
	return offset[id + 1] - offset[id];
}

void Ideal::init_edge_sequences(int num_edge_seqs){
	len_edge_sequences = num_edge_seqs;
	edge_sequences = new pair<uint32_t, uint32_t>[num_edge_seqs];
	assert(len_edge_sequences < 65536);
}

void Ideal::process_pixels_null(int x, int y){
	offset[(x+1)*(y+1)] = len_edge_sequences;
	for(int i = (x+1)*(y+1)-1; i >= 0; i --){
		if(show_status(i) != BORDER){
			offset[i] = offset[i + 1]; 
		}
	}
}

void Ideal::process_crosses(map<int, vector<cross_info>> edges_info){
	int num_edge_seqs = 0;

	for(auto info : edges_info){
		auto pix = info.first;
		auto crosses = info.second; 
		if(crosses.size() == 0) return;

		if(crosses.size()%2==1){
			crosses.push_back(cross_info((cross_type)!crosses[crosses.size()-1].type,crosses[crosses.size()-1].edge_id));
		}

		int start = 0;
		int end = crosses.size() - 1;
		if(crosses[0].type == LEAVE){
			assert(crosses[end].type == ENTER);
			num_edge_seqs += 2;
			start ++;
			end --;
		}

		for(int i = start; i <= end; i++){
			assert(crosses[i].type == ENTER);
			//special case, an ENTER has no pair LEAVE,
			//happens when one edge crosses the pair
			if(i == end || crosses[i + 1].type == ENTER){
				num_edge_seqs ++;
			}else{
				num_edge_seqs ++;
				i++;
			}
		}

	}
	init_edge_sequences(num_edge_seqs);

	int idx = 0;
	int edge_count = 0;
	for(auto info : edges_info){
		auto pix = info.first;
		auto crosses = info.second;
		if(crosses.size() == 0) return;
		
		if(crosses.size() % 2 == 1){
			crosses.push_back(cross_info((cross_type)!crosses[crosses.size()-1].type, crosses[crosses.size()-1].edge_id));
		}
		
		assert(crosses.size()%2==0);

		// Initialize based on crosses.size().
		int start = 0;
		int end = crosses.size() - 1;
		set_offset(pix, idx);

		if(crosses[0].type == LEAVE){
			assert(crosses[end].type == ENTER);
			add_edge(idx ++, 0, crosses[0].edge_id);
			add_edge(idx ++, crosses[end].edge_id, boundary->num_vertices - 2);
			start ++;
			end --;
		}

		for(int i = start; i <= end; i++){
			assert(crosses[i].type == ENTER);
			//special case, an ENTER has no pair LEAVE,
			//happens when one edge crosses the pair
			if(i == end || crosses[i + 1].type == ENTER){
				add_edge(idx ++, crosses[i].edge_id, crosses[i].edge_id);
			}else{
				add_edge(idx ++, crosses[i].edge_id, crosses[i+1].edge_id);
				i++;
			}
		}
	}
}

void Ideal::process_intersection(map<int, vector<double>> intersection_info, Direction direction){
	int num_nodes = 0;
	for(auto i : intersection_info){
		num_nodes += i.second.size();
	}
	if(direction == HORIZONTAL){
		horizontal->init_intersection_node(num_nodes);
		horizontal->set_num_crosses(num_nodes);
		int idx = 0;
		for(auto info : intersection_info){
			auto h = info.first;
			auto nodes = info.second;
			
			sort(nodes.begin(), nodes.end());

			horizontal->set_offset(h, idx);

			for(auto node : nodes){
				horizontal->add_node(idx, node);
				idx ++;
			}
		}
		horizontal->set_offset(dimy, idx);
	}else{
		vertical->init_intersection_node(num_nodes);
		vertical->set_num_crosses(num_nodes);
		vertical->set_offset(dimx + 1, num_nodes);

		int idx = 0;
		for(auto info : intersection_info){
			auto h = info.first;
			auto nodes = info.second;
			
			sort(nodes.begin(), nodes.end());

			vertical->set_offset(h, idx);

			for(auto node : nodes){
				vertical->add_node(idx, node);
				idx ++;
			}
		}
		vertical->set_offset(dimx, idx);		
	}

}

void Ideal::init_pixels(){
	assert(mbr);
	status_size = get_num_pixels();
	if(use_hierachy){
		layer_offset = new uint32_t[num_layers + 1];
		layer_info = new RasterInfo[num_layers + 1];

		int _dimx = dimx, _dimy = dimy;
		for(int i = num_layers - 1; i >= 0; i --){
			layer_offset[i] = status_size;
			_dimx = static_cast<int>(ceil(_dimx / 2.0));
			_dimy = static_cast<int>(ceil(_dimy / 2.0));
			status_size += (_dimx + 1) * (_dimy + 1);
		}
		assert(_dimx == 1 && _dimy == 1);
	}

	status = new uint8_t[status_size];
	memset(status, 0, status_size * sizeof(uint8_t));
    offset = new uint32_t[(dimx+1)*(dimy+1) + 1];    // +1 here is to ensure that pointer[num_pixels] equals len_edge_sequences, so we don't need to make a special case for the last pointer.
	horizontal = new Grid_line(dimy);
	vertical = new Grid_line(dimx);
}

void Ideal::evaluate_edges(){
	map<int, vector<double>> horizontal_intersect_info;
	map<int, vector<double>> vertical_intersect_info;
	map<int, vector<cross_info>> edges_info;
	
	// normalize
	assert(mbr);
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	for(int i=0;i<boundary->num_vertices-1;i++){
		double x1 = boundary->p[i].x;
		double y1 = boundary->p[i].y;
		double x2 = boundary->p[i+1].x;
		double y2 = boundary->p[i+1].y;

		int cur_startx = (x1-start_x)/step_x;
		int cur_endx = (x2-start_x)/step_x;
		int cur_starty = (y1-start_y)/step_y;
		int cur_endy = (y2-start_y)/step_y;

		if(cur_startx==dimx+1){
			cur_startx--;
		}
		if(cur_endx==dimx+1){
			cur_endx--;
		}

		int minx = min(cur_startx,cur_endx);
		int maxx = max(cur_startx,cur_endx);

		if(cur_starty==dimy+1){
			cur_starty--;
		}
		if(cur_endy==dimy+1){
			cur_endy--;
		}
		// todo should not happen for normal cases
		if(cur_startx>dimx||cur_endx>dimx||cur_starty>dimy||cur_endy>dimy){
			cout<<"xrange\t"<<cur_startx<<" "<<cur_endx<<endl;
			cout<<"yrange\t"<<cur_starty<<" "<<cur_endy<<endl;
			printf("xrange_val\t%f %f\n",(x1-start_x)/step_x, (x2-start_x)/step_x);
			printf("yrange_val\t%f %f\n",(y1-start_y)/step_y, (y2-start_y)/step_y);
			assert(false);
		}
		assert(cur_startx<=dimx);
		assert(cur_endx<=dimx);
		assert(cur_starty<=dimy);
		assert(cur_endy<=dimy);

		set_status(get_id(cur_startx, cur_starty), BORDER);
		set_status(get_id(cur_endx, cur_endy), BORDER);

		//in the same pixel
		if(cur_startx==cur_endx&&cur_starty==cur_endy){
			continue;
		}

		if(y1==y2){
			//left to right
			if(cur_startx<cur_endx){
				for(int x=cur_startx;x<cur_endx;x++){
					vertical_intersect_info[x + 1].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x+1, cur_starty)].push_back(cross_info(ENTER, i));
					set_status(get_id(x, cur_starty), BORDER);
					set_status(get_id(x+1, cur_starty), BORDER);
				}
			}else { // right to left
				for(int x=cur_startx;x>cur_endx;x--){
					vertical_intersect_info[x].push_back(y1);
					edges_info[get_id(x, cur_starty)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(x-1, cur_starty)].push_back(cross_info(ENTER, i));
					set_status(get_id(x, cur_starty), BORDER);
					set_status(get_id(x-1, cur_starty), BORDER);
				}
			}
		}else if(x1==x2){
			//bottom up
			if(cur_starty<cur_endy){
				for(int y=cur_starty;y<cur_endy;y++){
					horizontal_intersect_info[y + 1].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y+1)].push_back(cross_info(ENTER, i));
					set_status(get_id(cur_startx, y), BORDER);
					set_status(get_id(cur_startx, y+1), BORDER);
				}
			}else { //border[bottom] down
				for(int y=cur_starty;y>cur_endy;y--){
					horizontal_intersect_info[y].push_back(x1);
					edges_info[get_id(cur_startx, y)].push_back(cross_info(LEAVE, i));
					edges_info[get_id(cur_startx, y-1)].push_back(cross_info(ENTER, i));
					set_status(get_id(cur_startx, y), BORDER);
					set_status(get_id(cur_startx, y-1), BORDER);
				}
			}
		}else{
			// solve the line function
			double a = (y1-y2)/(x1-x2);
			double b = (x1*y2-x2*y1)/(x1-x2);

			int x = cur_startx;
			int y = cur_starty;
			while(x!=cur_endx||y!=cur_endy){
				bool passed = false;
				double yval = 0;
				double xval = 0;
				int cur_x = 0;
				int cur_y = 0;
				//check horizontally
				if(x!=cur_endx){
					if(cur_startx<cur_endx){
						xval = ((double)x+1)*step_x+start_x;
					}else{
						xval = (double)x*step_x+start_x;
					}
					yval = xval*a+b;
					cur_y = (yval-start_y)/step_y;
					//printf("y %f %d\n",(yval-start_y)/step_y,cur_y);
					if(cur_y>max(cur_endy, cur_starty)){
						cur_y=max(cur_endy, cur_starty);
					}
					if(cur_y<min(cur_endy, cur_starty)){
						cur_y=min(cur_endy, cur_starty);
					}
					if(cur_y==y){
						passed = true;
						// left to right
						if(cur_startx<cur_endx){
							vertical_intersect_info[x + 1].push_back(yval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x ++, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}else{//right to left
							vertical_intersect_info[x].push_back(yval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x --, y)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
					}
				}
				//check vertically
				if(y!=cur_endy){
					if(cur_starty<cur_endy){
						yval = (y+1)*step_y+start_y;
					}else{
						yval = y*step_y+start_y;
					}
					xval = (yval-b)/a;
					int cur_x = (xval-start_x)/step_x;
					//printf("x %f %d\n",(xval-start_x)/step_x,cur_x);
					if(cur_x>max(cur_endx, cur_startx)){
						cur_x=max(cur_endx, cur_startx);
					}
					if(cur_x<min(cur_endx, cur_startx)){
						cur_x=min(cur_endx, cur_startx);
					}
					if(cur_x==x){
						passed = true;
						if(cur_starty<cur_endy){// bottom up
							horizontal_intersect_info[y + 1].push_back(xval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y ++)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}else{// top down
							horizontal_intersect_info[y].push_back(xval);
							set_status(get_id(x, y), BORDER);
							edges_info[get_id(x, y --)].push_back(cross_info(LEAVE, i));
							edges_info[get_id(x, y)].push_back(cross_info(ENTER, i));
							set_status(get_id(x, y), BORDER);
						}
					}
				}
				// for debugging, should never happen
				if(!passed){
					boundary->print();
					cout<<"dim\t"<<dimx<<" "<<dimy<<endl;
					printf("val\t%f %f\n",(xval-start_x)/step_x, (yval-start_y)/step_y);
					cout<<"curxy\t"<<x<<" "<<y<<endl;
					cout<<"calxy\t"<<cur_x<<" "<<cur_y<<endl;
					cout<<"xrange\t"<<cur_startx<<" "<<cur_endx<<endl;
					cout<<"yrange\t"<<cur_starty<<" "<<cur_endy<<endl;
					printf("xrange_val\t%f %f\n",(x1-start_x)/step_x, (x2-start_x)/step_x);
					printf("yrange_val\t%f %f\n",(y1-start_y)/step_y, (y2-start_y)/step_y);
				}
				assert(passed);
			}
		}
	}

	// special case
	if(edges_info.size() == 0 && boundary->num_vertices > 0){
		init_edge_sequences(1);
		set_offset(0, 0);
		add_edge(0, 0, boundary->num_vertices - 1);
	}else{
		process_crosses(edges_info);
	}

	// for(int i = 0; i <= dimx; i ++) {
	// 	set_status(get_id(i, dimy), OUT);
	// }

	// for(int i = 0; i <= dimy; i ++) {
	// 	set_status(get_id(dimx, i), OUT);
	// }


	// process_crosses(edges_info);
	process_intersection(horizontal_intersect_info, HORIZONTAL);
	process_intersection(vertical_intersect_info, VERTICAL);
	process_pixels_null(dimx, dimy);
}

void Ideal::scanline_reandering(){
	const double start_x = mbr->low[0];
	const double start_y = mbr->low[1];

	for(int y = 1; y < dimy; y ++){
		bool isin = false;
		uint32_t i = horizontal->get_offset(y), j = horizontal->get_offset(y + 1);
		for(int x = 0; x < dimx; x ++){
			if(show_status(get_id(x, y)) != BORDER){
				if(isin){
					set_status(get_id(x, y), IN);
				}else{
					set_status(get_id(x, y), OUT);
				}
				continue;
			}
			int pass = 0;
			while(i < j && horizontal->get_intersection_nodes(i) <= start_x + step_x * (x + 1)){
				pass ++;
				i ++;
			}
			if(pass % 2 == 1) isin = !isin;

		}
	}
}

void Ideal::rasterization(){

	//1. create space for the pixels
	init_pixels();

	//2. edge crossing to identify BORDER pixels
	evaluate_edges();

	//3. determine the status of rest pixels with scanline rendering
	scanline_reandering();
}

void Ideal::rasterization(int vpr){
	assert(vpr > 0);
	pthread_mutex_lock(&ideal_partition_lock);
	if(use_hierachy)
		num_layers = static_cast<int>(ceil(max(log(dimx + 1) / log(2.0), log(dimy + 1) / log(2.0))));
    else 
		init_raster(boundary->num_vertices / vpr);

    rasterization();

	if(use_hierachy){
		assert(num_layers != 0);
		layers = new Hraster[num_layers + 1];

		// process the last layer
		layers[num_layers].init(step_x, step_y, dimx, dimy, getMBB(), true);
		layers[num_layers].set_status(status);
		layer_info[num_layers] = {*mbr, dimx, dimy, step_x, step_y};
		layer_offset[num_layers] = 0;

		for(int i = num_layers - 1; i >= 0; i --){
			double _step_x = layers[i + 1].get_step_x() * 2, _step_y = layers[i + 1].get_step_y() * 2;
			int _dimx = static_cast<int>(ceil(layers[i + 1].get_dimx() / 2.0)), _dimy = static_cast<int>(ceil(layers[i + 1].get_dimy() / 2.0));

			layers[i].init(_step_x, _step_y, _dimx, _dimy, getMBB(), false);
			layer_info[i] = {*layers[i].mbr, _dimx, _dimy, _step_x, _step_y};
			
			layers[i].merge(layers[i + 1]);
			memcpy(status + layer_offset[i], layers[i].get_status(), (_dimx+1) * (_dimy+1) * sizeof(uint8_t));	
		}
	}

	pthread_mutex_unlock(&ideal_partition_lock);
}

int Ideal::num_edges_covered(int id){
	int c = 0;
	for(int i = 0; i < get_num_sequences(id); i ++){
		auto r = edge_sequences[offset[id] + i];
		c += r.second;
	}
	return c;
}

int Ideal::get_num_border_edge(){
	int num = 0;
	for(int i = 0; i < get_num_pixels(); i ++){
		if(show_status(i) == BORDER){
			num += num_edges_covered(i);
		}
	}
	return num;
}

size_t Ideal::get_num_crosses(){
	size_t num = 0;
	num = horizontal->get_num_crosses() + vertical->get_num_crosses();
	return num;
}

int Ideal::count_intersection_nodes(Point &p){
	// here we assume the point inside one of the pixel
	int pix_id = get_pixel_id(p);
	assert(show_status(pix_id) == BORDER);
	int count = 0;
	int x = get_x(pix_id) + 1;
	uint32_t i = vertical->get_offset(x), j;
	if(x < dimx) j = vertical->get_offset(x + 1);
	else j = vertical->get_num_crosses();
	while(i < j && vertical->get_intersection_nodes(i) <= p.y){
		count ++;
		i ++;
	}
	return count;
}

bool Ideal::contain(Point &p, query_context *ctx, bool profile){

	// the MBB may not be checked for within query
	if(!mbr->contain(p)){
		return false;
	}


	struct timeval start = get_cur_time();
	// todo adjust the lower bound of pixel number when the raster model is usable
    start = get_cur_time();
    int target = get_pixel_id(p);
    box bx = get_pixel_box(get_x(target), get_y(target));
    double bx_high = bx.high[0];
    if(show_status(target) == IN) {
        return true;
    }
    if(show_status(target) == OUT){
        return false;
    }

    start = get_cur_time();
    bool ret = false;

    // checking the intersection edges in the target pixel
    for(uint32_t e = 0; e < get_num_sequences(target); e ++){    
        auto edges = get_edge_sequence(get_offset(target) + e);
        auto pos = edges.first;
        for(int k = 0; k < edges.second; k ++){
            int i = pos + k;
            int j = i + 1;  //ATTENTION
            if(((boundary->p[i].y >= p.y) != (boundary->p[j].y >= p.y))){
                double int_x = (boundary->p[j].x - boundary->p[i].x) * (p.y - boundary->p[i].y) / (boundary->p[j].y - boundary->p[i].y) + boundary->p[i].x;
                if(p.x <= int_x && int_x <= bx_high){
                    ret = !ret;
                }
            }
        }
    }
    // check the crossing nodes on the right bar
    // swap the state of ret if odd number of intersection
    // nodes encountered at the right side of the border
    struct timeval tstart = get_cur_time();
    int nc = count_intersection_nodes(p);
    if(nc%2==1){
        ret = !ret;
    }
    return ret;
}

PartitionStatus Ideal::segment_contain(Point &p){
    int target = get_pixel_id(p);
    box bx = get_pixel_box(get_x(target), get_y(target));
    double bx_high = bx.high[0];
    if(show_status(target) == IN) {
        return IN;
    }
    if(show_status(target) == OUT){
        return OUT;
    }

    bool ret = false;

    // checking the intersection edges in the target pixel
    for(uint32_t e = 0; e < get_num_sequences(target); e ++){    
        auto edges = get_edge_sequence(get_offset(target) + e);
        auto pos = edges.first;
        for(int k = 0; k < edges.second; k ++){
			Point v1 = boundary->p[pos + k];
			Point v2 = boundary->p[pos + k + 1];
			if(p == v1 || p == v2){
				return BORDER;
			}
            if ((v1.y >= p.y) != (v2.y >= p.y))
			{

				const double dx = v2.x - v1.x;
				const double dy = v2.y - v1.y;
				const double py_diff = p.y - v1.y;

				if (dy != 0.0)
				{
					const double int_x = dx * py_diff / dy + v1.x;
					if(fabs(p.x - int_x) < eps) {
						return BORDER;
					}
					if (p.x < int_x && int_x <= bx.high[0])
					{
						ret = !ret;
					}
				}
			}
        }
    }
    // check the crossing nodes on the right bar
    // swap the state of ret if odd number of intersection
    // nodes encountered at the right side of the border
    int nc = count_intersection_nodes(p);
    if(nc%2==1){
        ret = !ret;
    }
	if(ret) return IN;
	else return OUT;
}


bool Ideal::contain(Ideal *target, query_context *ctx, bool profile){
	if(!getMBB()->contain(*target->getMBB())){
		//log("mbb do not contain");
		return false;
	}
	vector<int> pxs = retrieve_pixels(target->getMBB());
	int etn = 0;
	int itn = 0;
	for(auto p : pxs){
		if(show_status(p) == OUT){
			etn++;
		}else if(show_status(p) == IN){
			itn++;
		}
	}
	if(etn == pxs.size()){
		return false;
	}
	if(itn == pxs.size()){
		return true;
	}

	vector<int> tpxs;

	for(auto p : pxs){
		box bx =  get_pixel_box(get_x(p), get_y(p));
		tpxs = target->retrieve_pixels(&bx);
		for(auto p2 : tpxs){
			// an external pixel of the container intersects an internal
			// pixel of the containee, which means the containment must be false
			if(show_status(p) == IN) continue;
			if(show_status(p) == OUT && target->show_status(p2) == IN){
				return false;
			}
			if (show_status(p) == OUT && target->show_status(p2) == BORDER){
				Point pix_border[5];
				pix_border[0].x = bx.low[0]; pix_border[0].y = bx.low[1];
				pix_border[1].x = bx.low[0]; pix_border[1].y = bx.high[1];
				pix_border[2].x = bx.high[0]; pix_border[2].y = bx.high[1];
				pix_border[3].x = bx.high[0]; pix_border[3].y = bx.low[1];
				pix_border[4].x = bx.low[0]; pix_border[4].y = bx.low[1];
				for (int e = 0; e < target->get_num_sequences(p2); e++){
					auto edges = target->get_edge_sequence(target->get_offset(p2) + e);
					auto pos = edges.first;
					auto size = edges.second;
					if (segment_intersect_batch(target->boundary->p + pos, pix_border, size, 4)){
						return false;
					}
				}
			}
			// evaluate the state
			if(show_status(p) == BORDER && target->show_status(p2) == BORDER){
				for(int i = 0; i < get_num_sequences(p); i ++){
					auto r = get_edge_sequence(get_offset(p) + i);
					for(int j = 0; j < target->get_num_sequences(p2); j ++){
						auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
						if(segment_intersect_batch(boundary->p+r.first, target->boundary->p+r2.first, r.second, r2.second)){
							return false;
						}
					}
				}
			}
		}
		tpxs.clear();
	}
	pxs.clear();

	// this is the last step for all the cases, when no intersection segment is identified
	// pick one point from the target and it must be contained by this polygon
	Point p(target->getx(0),target->gety(0));
	return contain(p, ctx,false);
}

void Ideal::intersection(Ideal *target, query_context *ctx){
	vector<int> pxs = retrieve_pixels(target->getMBB());
	int etn = 0;
	int itn = 0;
	for(auto p : pxs){
		if(show_status(p) == OUT){
			etn++;
		}else if(show_status(p) == IN){
			itn++;
		}
	}
	if(etn == pxs.size() || itn == pxs.size()){
		return;
	}

	vector<int> tpxs;
	vector<Intersection> inters;

	for(auto p : pxs){
		box bx =  get_pixel_box(get_x(p), get_y(p));
		tpxs = target->retrieve_pixels(&bx);
		for(auto p2 : tpxs){
			if(show_status(p) == BORDER && target->show_status(p2) == BORDER){
				for(int i = 0; i < get_num_sequences(p); i ++){
					auto r = get_edge_sequence(get_offset(p) + i);
					for(int j = 0; j < target->get_num_sequences(p2); j ++){
						auto r2 = target->get_edge_sequence(target->get_offset(p2) + j);
						assert(r.second != 0 && r2.second != 0);
						segment_intersect_batch(boundary->p, target->boundary->p, r.first, r2.first, r.first + r.second, r2.first + r2.second, inters);
					}
				}
			}
		}
		tpxs.clear();
	}
	pxs.clear();

	std::sort(inters.begin(), inters.end(),
		[](const Intersection &a, const Intersection &b){
			return a.p < b.p;
		});

	auto new_end = std::unique(inters.begin(), inters.end(),
        [](const Intersection& a, const Intersection& b) {
            return a.p == b.p;
        });

	inters.erase(new_end, inters.end());

	int num_inters = inters.size();

	vector<Segment> candidates;
	vector<PartitionStatus> temp;

	std::sort(inters.begin(), inters.end(),
		[](const Intersection &a, const Intersection &b){
        	if (a.edge_source_id != b.edge_source_id){
				return a.edge_source_id < b.edge_source_id;
			}else if(a.t != b.t){
        		return a.t < b.t;
			}
		});

	for(int i = 0; i < num_inters; i ++){
		// Intersection a = inters[i];
		// Intersection b = inters[(i + 1) % num_inters];
		
		// int a_edge_id = a.edge_source_id;
		// int b_edge_id = b.edge_source_id;
		
		// // 调整边界参数
		// if(fabs(a.t - 1) < eps) a_edge_id++;
		// if(fabs(b.t) < eps) b_edge_id--;
		
		// // 判断是否需要特殊处理edge_id
		// bool use_edge_range = (i + 1 >= num_inters) ? 
		// 					true : (a_edge_id != b_edge_id && a_edge_id + 1 <= b_edge_id);
		
		// if(use_edge_range){
		// 	auto p = boundary->p[a_edge_id + 1];
		// 	candidates.push_back({true, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
		// 	status.push_back(target->segment_contain(p));
		// }else{
		// 	auto p = (a.p + b.p) * 0.5;
		// 	candidates.push_back({true, a.p, b.p, -1, -1, 0});
		// 	status.push_back(target->segment_contain(p));
		// }
		
		
		if(i + 1 >= num_inters){
			Intersection a = inters[i];
			Intersection b = inters[0];
			int a_edge_id = a.edge_source_id;
			int b_edge_id = b.edge_source_id;
			double a_param = a.t;
			double b_param = b.t;
			if(fabs(a_param - 1.0) < eps) a_edge_id ++;
			if(fabs(b_param) < eps) b_edge_id --;
			if(a_edge_id == b_edge_id){
				candidates.push_back({true, a.p, b.p, -1, -1, 0});
				auto p = (a.p + b.p) * 0.5;
				temp.push_back(target->segment_contain(p));
			}else{
				candidates.push_back({true, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
				auto p = boundary->p[a_edge_id + 1];
				temp.push_back(target->segment_contain(p));				
			}
		}else{
			Intersection a = inters[i];
			Intersection b = inters[i + 1];
			int a_edge_id = a.edge_source_id;
			int b_edge_id = b.edge_source_id;
			double a_param = a.t;
			double b_param = b.t;
			if(a_edge_id != b_edge_id){
				if(fabs(a_param - 1.0) < eps) a_edge_id ++;
				if(fabs(b_param) < eps) b_edge_id --;
				if(a_edge_id == b_edge_id){
					candidates.push_back({true, a.p, b.p, -1, -1, 0});
					auto p = (a.p + b.p) * 0.5;
					temp.push_back(target->segment_contain(p));
				}else{
					candidates.push_back({true, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
					auto p = boundary->p[a_edge_id + 1];
					temp.push_back(target->segment_contain(p));
				}
			}else{	
					candidates.push_back({true, a.p, b.p, -1, -1, 0});
					auto p = (a.p + b.p) * 0.5;
					temp.push_back(target->segment_contain(p));
			}	
		}
	}

	std::sort(inters.begin(), inters.end(),
		[](const Intersection &a, const Intersection &b){
			if (a.edge_target_id != b.edge_target_id){
				return a.edge_target_id < b.edge_target_id;
			}else if(a.u != b.u){
				return a.u < b.u;
			} 
		});

	for(int i = 0; i < num_inters; i ++){
		// Intersection a = inters[i];
		// Intersection b = inters[(i + 1) % num_inters];
		
		// int a_edge_id = a.edge_target_id;
		// int b_edge_id = b.edge_target_id;
		
		// // 调整边界参数
		// if(fabs(a.u - 1) < eps) a_edge_id++;
		// if(fabs(b.u) < eps) b_edge_id--;
		
		// // 判断是否需要特殊处理edge_id
		// bool use_edge_range = (i + 1 >= num_inters) ? 
		// 					true : (a_edge_id != b_edge_id && a_edge_id + 1 <= b_edge_id);
		
		// if(use_edge_range){
		// 	auto p = target->boundary->p[a_edge_id + 1];
		// 	candidates.push_back({false, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
		// 	status.push_back(segment_contain(p));
		// }else{
		// 	auto p = (a.p + b.p) * 0.5;
		// 	candidates.push_back({false, a.p, b.p, -1, -1, 0});
		// 	status.push_back(segment_contain(p));
		// }

		if(i + 1 >= num_inters){
			Intersection a = inters[i];
			Intersection b = inters[0];
			int a_edge_id = a.edge_target_id;
			int b_edge_id = b.edge_target_id;
			double a_param = a.u;
			double b_param = b.u;
			if(fabs(a_param - 1.0) < eps) a_edge_id ++;
			if(fabs(b_param) < eps) b_edge_id --;
			if(a_edge_id == b_edge_id){
				candidates.push_back({false, a.p, b.p, -1, -1, 0});
				auto p = (a.p + b.p) * 0.5;
				temp.push_back(segment_contain(p));
			}else{
				candidates.push_back({false, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
				auto p = target->boundary->p[a_edge_id + 1];
				temp.push_back(segment_contain(p));				
			}
		}else{
			Intersection a = inters[i];
			Intersection b = inters[i + 1];
			int a_edge_id = a.edge_target_id;
			int b_edge_id = b.edge_target_id;
			double a_param = a.u;
			double b_param = b.u;
			if(a_edge_id != b_edge_id){
				if(fabs(a_param - 1.0) < eps) a_edge_id ++;
				if(fabs(b_param) < eps) b_edge_id --;
				if(a_edge_id == b_edge_id){
					candidates.push_back({false, a.p, b.p, -1, -1, 0});
					auto p = (a.p + b.p) * 0.5;
					temp.push_back(segment_contain(p));
				}else{
					candidates.push_back({false, a.p, b.p, a_edge_id + 1, b_edge_id, 0});
					auto p = target->boundary->p[a_edge_id + 1];
					temp.push_back(segment_contain(p));
				}
			}else{	
					candidates.push_back({false, a.p, b.p, -1, -1, 0});
					auto p = (a.p + b.p) * 0.5;
					temp.push_back(segment_contain(p));
			}	
		}
	}

	vector<Segment> segments;
	vector<bool> status;

  	for (int i = 0; i < candidates.size(); i ++){
		if(temp[i] == IN){
			auto& seg = candidates[i];
			segments.push_back(seg);
			status.push_back(true);
		}
	}

  	for (int i = 0; i < candidates.size(); i ++){
		if(temp[i] == BORDER){
			auto& seg = candidates[i];
			segments.push_back(seg);
			status.push_back(false);
		}
	}

	auto num_segments = segments.size();

	std::unordered_map<std::string, std::vector<size_t>> adjacencyList;
	std::vector<bool> used(num_segments, false);

    auto pointToKey = [](const Point& p) {
        return std::to_string(p.x) + ":" + std::to_string(p.y);
    };

	for (size_t i = 0; i < num_segments; ++i) {
		adjacencyList[pointToKey(segments[i].start)].push_back(i);
    }

 	for (size_t startIdx = 0; startIdx < segments.size(); ++startIdx) {
        if (used[startIdx] || !status[startIdx]) continue;
        
        vector<Point> currentVertices;
        
        size_t currentSegIdx = startIdx;
        Point currentPoint = segments[startIdx].start;
        Point startPoint = currentPoint;
        
        bool foundCycle = false;
  
        while (!used[currentSegIdx]) {
            used[currentSegIdx] = true;
            
            currentVertices.push_back(currentPoint);
			const Segment& seg = segments[currentSegIdx];
			Point *vertices = nullptr;
			size_t num_vertices;
			if(seg.is_source) {
				vertices = boundary->p;
				num_vertices = get_num_vertices();
			}else{
				vertices = target->boundary->p;
				num_vertices = target->get_num_vertices();
			}
			if(seg.edge_start != -1){
                if(seg.edge_start <= seg.edge_end){
                    for(int verId = seg.edge_start; verId <= seg.edge_end; verId ++){
						currentVertices.push_back(vertices[verId]);
                    }
                }else{
                    for(int verId = seg.edge_start; verId < num_vertices - 1; verId ++){
                        currentVertices.push_back(vertices[verId]);
                    }
                    for(int verId = 0; verId <= seg.edge_end; verId ++){
                        currentVertices.push_back(vertices[verId]);
                    }
                }
			}
           
            Point nextPoint = currentPoint == seg.start ? seg.end : seg.start;
            
            bool foundNext = false;
            std::string nextKey = pointToKey(nextPoint);
            
            if (adjacencyList.find(nextKey) != adjacencyList.end()) {
                for (size_t idx : adjacencyList[nextKey]) {
                    if (!used[idx]) {
                        currentSegIdx = idx;
                        currentPoint = nextPoint;
                        foundNext = true;
                        break;
                    }
                }
            }
            
            // 如果回到起点，我们找到了一个闭合的多边形
            if (!foundNext && nextPoint == startPoint) {
                currentVertices.push_back(nextPoint); // 添加最后一个点闭合多边形
                foundCycle = true;
                break;
            }
            
            // 如果没有找到下一个segment，则路径不能闭合
            if (!foundNext) break;
        }
        
        // 如果找到了闭合的多边形并且至少有3个点，添加到结果中
        if (foundCycle && currentVertices.size() >= 3) {
			VertexSequence* vs = new VertexSequence(currentVertices.size(), currentVertices.data());
			MyPolygon *currentPolygon = new MyPolygon();
			currentPolygon->set_boundary(vs);
            ctx->intersection_polygons.push_back(currentPolygon);
        } 
    }
	return;
}

double Ideal::get_possible_min(Point &p, int center, int step, bool geography){
	int core_x_low = get_x(center);
	int core_x_high = get_x(center);
	int core_y_low = get_y(center);
	int core_y_high = get_y(center);

	vector<int> needprocess;

	int ymin = max(0,core_y_low-step);
	int ymax = min(dimy,core_y_high+step);

	double mindist = DBL_MAX;
	//left scan
	if(core_x_low-step>=0){
		double x = get_pixel_box(core_x_low-step,ymin).high[0];
		double y1 = get_pixel_box(core_x_low-step,ymin).low[1];
		double y2 = get_pixel_box(core_x_low-step,ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	//right scan
	if(core_x_high+step<=get_dimx()){
		double x = get_pixel_box(core_x_high+step,ymin).low[0];
		double y1 = get_pixel_box(core_x_high+step,ymin).low[1];
		double y2 = get_pixel_box(core_x_high+step,ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0,core_x_low-step+(core_x_low-step>=0));
	// skip the last if there is right scan
	int xmax = min(dimx,core_x_high+step-(core_x_high+step<=dimx));
	//bottom scan
	if(core_y_low-step>=0){
		double y = get_pixel_box(xmin,core_y_low-step).high[1];
		double x1 = get_pixel_box(xmin,core_y_low-step).low[0];
		double x2 = get_pixel_box(xmax,core_y_low-step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	//top scan
	if(core_y_high+step<=get_dimy()){
		double y = get_pixel_box(xmin,core_y_low+step).low[1];
		double x1 = get_pixel_box(xmin,core_y_low+step).low[0];
		double x2 = get_pixel_box(xmax,core_y_low+step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = point_to_segment_distance(p, p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::get_possible_min(box *t_mbr, int core_x_low, int core_y_low, int core_x_high, int core_y_high, int step, bool geography)
{

	vector<int> needprocess;

	int ymin = max(0, core_y_low - step);
	int ymax = min(dimy, core_y_high + step);

	double mindist = DBL_MAX;
	// left scan
	if (core_x_low - step >= 0)
	{
		double x = get_pixel_box(core_x_low - step, ymin).high[0];
		double y1 = get_pixel_box(core_x_low - step, ymin).low[1];
		double y2 = get_pixel_box(core_x_low - step, ymax).high[1];

		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// right scan
	if (core_x_high + step <= get_dimx())
	{
		double x = get_pixel_box(core_x_high + step, ymin).low[0];
		double y1 = get_pixel_box(core_x_high + step, ymin).low[1];
		double y2 = get_pixel_box(core_x_high + step, ymax).high[1];
		Point p1 = Point(x, y1);
		Point p2 = Point(x, y2);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}

	// skip the first if there is left scan
	int xmin = max(0, core_x_low - step + (core_x_low - step >= 0));
	// skip the last if there is right scan
	int xmax = min(dimx, core_x_high + step - (core_x_high + step <= dimx));
	// bottom scan
	if (core_y_low - step >= 0)
	{
		double y = get_pixel_box(xmin, core_y_low - step).high[1];
		double x1 = get_pixel_box(xmin, core_y_low - step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low - step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	// top scan
	if (core_y_high + step <= get_dimy())
	{
		double y = get_pixel_box(xmin, core_y_low + step).low[1];
		double x1 = get_pixel_box(xmin, core_y_low + step).low[0];
		double x2 = get_pixel_box(xmax, core_y_low + step).high[0];
		Point p1 = Point(x1, y);
		Point p2 = Point(x2, y);
		double dist = t_mbr->distance(p1, p2, geography);
		mindist = min(dist, mindist);
	}
	return mindist;
}

double Ideal::distance(Point &p, query_context *ctx, bool profile){
	// distance is 0 if contained by the polygon
	double mindist = getMBB()->max_distance(p, ctx->geography);

	bool contained = contain(p, ctx, profile);
	if(contained){
		return 0;
	}
	
	double mbrdist = mbr->distance(p,ctx->geography);

	//initialize the starting pixel
	int closest = get_closest_pixel(p);
	
	int step = 0;
	double step_size = get_step(ctx->geography);
	vector<int> needprocess;

	while(true){
		if(step==0){
			needprocess.push_back(closest);
		}else{
			needprocess = expand_radius(closest, step);
		}
		// should never happen
		// all the boxes are scanned
		if(needprocess.size()==0){
			assert(false&&"should not evaluated all boxes");
			return boundary->distance(p, ctx->geography);
		}
		for(auto cur : needprocess){
			//printf("checking pixel %d %d %d\n",cur->id[0],cur->id[1],cur->status);
			if(show_status(cur) == BORDER){
				box cur_box = get_pixel_box(get_x(cur), get_y(cur));
				// printf("BOX: lowx=%lf, lowy=%lf, highx=%lf, highy=%lf\n", cur_box.low[0], cur_box.low[1], cur_box.high[0], cur_box.high[1]);
				double mbr_dist = cur_box.distance(p, ctx->geography);
				// skip the pixels that is further than the current minimum
				if(mbr_dist >= mindist){
					continue;
				}

				// the vector model need be checked.

				for(int i = 0; i < get_num_sequences(cur); i ++){
					auto rg = get_edge_sequence(get_offset(cur) + i);
					for(int j = 0; j < rg.second; j ++){
						auto r = rg.first + j;
						double dist = point_to_segment_distance(p, *get_point(r), *get_point(r+1), ctx->geography);
						mindist = min(mindist, dist);
						if(ctx->within(mindist)){
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();

		// for within query, return if the current minimum is close enough
		if(ctx->within(mindist)){
			return mindist;
		}
		step++;
		double minrasterdist = get_possible_min(p, closest, step, ctx->geography);
		// close enough
		if(mindist < minrasterdist){
			break;
		}
	}
	// IDEAL return
	return mindist;
}

// get the distance from pixel pix to polygon target
double Ideal::distance(Ideal *target, int pix, query_context *ctx, bool profile)
{
	assert(show_status(pix) == BORDER);

	auto pix_x = get_x(pix);
	auto pix_y = get_y(pix);
	auto pix_box = get_pixel_box(pix_x, pix_y);
	double mindist = getMBB()->max_distance(pix_box, ctx->geography);
	double mbrdist = getMBB()->distance(pix_box, ctx->geography);
	int step = 0;

	// initialize the seed closest pixels
	vector<int> needprocess = target->get_closest_pixels(pix_box);
	assert(needprocess.size() > 0);
	unsigned short lowx = target->get_x(needprocess[0]);
	unsigned short highx = target->get_x(needprocess[0]);
	unsigned short lowy = target->get_y(needprocess[0]);
	unsigned short highy = target->get_y(needprocess[0]);
	for (auto p : needprocess)
	{
		lowx = min(lowx, (unsigned short)target->get_x(p));
		highx = max(highx, (unsigned short)target->get_x(p));
		lowy = min(lowy, (unsigned short)target->get_y(p));
		highy = max(highy, (unsigned short)target->get_y(p));
	}

	while (true)
	{
		// for later steps, expand the circle to involve more pixels
		if (step > 0)
		{
			needprocess = target->expand_radius(lowx, highx, lowy, highy, step);
		}

		// all the boxes are scanned (should never happen)
		if (needprocess.size() == 0)
		{
			return mindist;
		}

		for (auto cur : needprocess)
		{
			// note that there is no need to check the edges of
			// this pixel if it is too far from the target
			auto cur_x = target->get_x(cur);
			auto cur_y = target->get_y(cur);

			if (target->show_status(cur) == BORDER)
			{
				bool toofar = (target->get_pixel_box(cur_x, cur_y).distance(pix_box, ctx->geography) >= mindist);
				if (toofar)
				{
					continue;
				}
				// the vector model need be checked.
				for (int i = 0; i < get_num_sequences(pix); i++)
				{
					auto pix_er = get_edge_sequence(get_offset(pix) + i);
					for (int j = 0; j < target->get_num_sequences(cur); j++)
					{
						auto cur_er = target->get_edge_sequence(target->get_offset(cur) + j);
						if (cur_er.second < 2 || pix_er.second < 2)
							continue;
						double dist;
						if (ctx->is_within_query())
						{
							dist = segment_to_segment_within_batch(target->boundary->p + cur_er.first,
																   boundary->p + pix_er.first, cur_er.second, pix_er.second,
																   ctx->within_distance, ctx->geography, ctx->edge_checked.counter);
						}
						else
						{
							dist = segment_sequence_distance(target->boundary->p + cur_er.first,
															 boundary->p + pix_er.first, cur_er.second, pix_er.second, ctx->geography);
						}
						mindist = min(dist, mindist);
						if (ctx->within(mindist))
						{
							return mindist;
						}
					}
				}
			}
		}
		needprocess.clear();
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = target->get_possible_min(&pix_box, lowx, lowy, highx, highy, step, ctx->geography);
		if (mindist < minrasterdist)
			break;
	}

	return mindist;
}

double Ideal::distance(Ideal *target, query_context *ctx)
{
	timeval start = get_cur_time();
	// both polygons are rasterized and the pixel of the target is larger
	// then swap the role of source and target, and use the target as the host
	// one currently we put this optimization in the caller
	if (target->get_step(false) > get_step(false))
	{
		return target->distance(this, ctx);
	}

	double mindist = getMBB()->max_distance(*target->getMBB(), ctx->geography);
	const double mbrdist = getMBB()->distance(*target->getMBB(), ctx->geography);
	int step = 0;

	vector<int> needprocess = get_closest_pixels(*target->getMBB());
	assert(needprocess.size() > 0);
	unsigned short lowx = get_x(needprocess[0]);
	unsigned short highx = get_x(needprocess[0]);
	unsigned short lowy = get_y(needprocess[0]);
	unsigned short highy = get_y(needprocess[0]);
	for (auto p : needprocess)
	{
		lowx = min(lowx, (unsigned short)get_x(p));
		highx = max(highx, (unsigned short)get_x(p));
		lowy = min(lowy, (unsigned short)get_y(p));
		highy = max(highy, (unsigned short)get_y(p));
	}

	while (true)
	{
		struct timeval start = get_cur_time();

		// first of all, expand the circle to involve more pixels
		if (step > 0)
		{
			needprocess = expand_radius(lowx, highx, lowy, highy, step);
		}

		// all the boxes are scanned (should never happen)
		if (needprocess.size() == 0)
		{
			return mindist;
		}

		for (auto cur : needprocess)
		{
			// printf("checking pixel %d %d
			// %d\n",cur->id[0],cur->id[1],cur->status);
			//  note that there is no need to check the edges of
			//  this pixel if it is too far from the target
			auto cur_x = get_x(cur);
			auto cur_y = get_y(cur);
			if (show_status(cur) == BORDER && get_pixel_box(cur_x, cur_y).distance(*target->getMBB(), ctx->geography) < mindist)
			{
				// the vector model need be checked.
				// do a polygon--pixel distance calculation
				double dist = distance(target, cur, ctx, true);
				mindist = min(dist, mindist);
				if (ctx->within(mindist))
				{
					return mindist;
				}
			}
		}
		needprocess.clear();
		if (ctx->within(mindist))
		{
			return mindist;
		}
		step++;
		double minrasterdist = get_possible_min(target->getMBB(), lowx, lowy, highx, highy, step, ctx->geography);
		if (mindist < minrasterdist)
			break;
	}

	// iterate until the closest pair of edges are found
	// assert(false && "happens when there is no boundary pixel, check out the input");
	// return DBL_MAX;
	return mindist;
}